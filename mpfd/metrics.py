# -*- coding: utf-8 -*-
"""MPFD-Bench metrics. Fully runnable on CPU given a Session (gold) + SystemOutput (prediction).

Headline metric = **false-barge-in rate (FBR)**. See docs/METRICS.md for definitions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .cells import RESPONSE_HORIZON, gold_events
from .schema import Action, GoldEvent, Session, SystemOutput, first_onset_after, spoke_in


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def score_session(session: Session, out: SystemOutput) -> Dict[str, List]:
    """Return per-event raw outcomes for one session (accumulated by aggregate())."""
    ev = gold_events(session)
    seg = out.agent_segments
    raw = defaultdict(list)

    # Windows in which the agent is LEGITIMATELY allowed to be speaking a response, so onsets there
    # are not "answering someone else". Used to make wrong_addressee timing-robust: a correct
    # response to an addressed turn must never be misattributed to a nearby human-directed question.
    respond_windows = [(e.start, e.end) for e in ev if e.action == Action.RESPOND]

    def _covered(t: float, windows: List[Tuple[float, float]]) -> bool:
        return any(a <= t <= b for a, b in windows)

    for e in ev:
        agent_spoke = spoke_in(seg, (e.start, e.end))
        if e.action == Action.SILENT:
            raw["silent_total"].append(1)
            raw["false_bargein"].append(1 if agent_spoke else 0)          # agent wrongly spoke during it
            if e.dialogue_act in ("question", "request"):
                raw["qlike_silent_total"].append(1)
                # "answered someone else's question" = the agent STARTS speaking in the window after it
                # ends, and that onset is NOT explained by a legitimate response to an addressed turn.
                onsets_in = [s[0] for s in seg if e.end < s[0] <= e.end + RESPONSE_HORIZON]
                answered = any(not _covered(o, respond_windows) for o in onsets_in)
                raw["wrong_addressee"].append(1 if answered else 0)
        elif e.action == Action.RESPOND:
            raw["respond_total"].append(1)
            # Timing-robust: the agent responded iff it produces speech anywhere in the response
            # window (a fresh onset OR an already-running turn that spans into it). This avoids the
            # onset-only undercount when a real model's response span is merged across e.start.
            responded = spoke_in(seg, (e.start, e.end))
            raw["responded"].append(1 if responded else 0)
            onset = first_onset_after(seg, e.start, horizon=e.end - e.start)
            if onset is not None:
                raw["response_latency"].append(onset - e.start)
        elif e.action == Action.STOP:
            raw["stop_total"].append(1)
            # was the agent speaking at the barge-in onset, and did it stop within the window?
            covering = [s for s in seg if s[0] <= e.start <= s[1]]
            stopped = all(s[1] <= e.end for s in covering) if covering else True
            raw["stopped"].append(1 if stopped else 0)
            if covering:
                raw["stop_latency"].append(max(0.0, min(s[1] for s in covering) - e.start))
        elif e.action in (Action.BACKCHANNEL, Action.CONTINUE):
            # agent should keep speaking: it must still be talking at the END of the window
            # (a truncated/aborted turn overlapping only the window start does NOT count).
            still = any(s[0] <= e.end - 0.05 <= s[1] for s in seg)
            raw["continue_total"].append(1)
            raw["continued"].append(1 if still else 0)

    # addressee classification (only if the system predicted it)
    if out.addressee_pred:
        for u in session.utterances:
            if u.speaker == session.agent_id or u.utt_id not in out.addressee_pred:
                continue
            gold_pos = u.is_for_agent
            pred_pos = bool(out.addressee_pred[u.utt_id])
            raw["addr_tp"].append(1 if (gold_pos and pred_pos) else 0)
            raw["addr_fp"].append(1 if (not gold_pos and pred_pos) else 0)
            raw["addr_fn"].append(1 if (gold_pos and not pred_pos) else 0)
    return raw


def aggregate(sessions: List[Session], outs: Dict[str, SystemOutput]) -> Dict[str, float]:
    """Aggregate metrics across sessions. `outs` maps session_id -> SystemOutput."""
    acc = defaultdict(list)
    per_cell = defaultdict(lambda: defaultdict(list))
    for s in sessions:
        out = outs.get(s.session_id, SystemOutput(session_id=s.session_id))
        raw = score_session(s, out)
        for k, v in raw.items():
            acc[k].extend(v)
            per_cell[s.cell][k].extend(v)

    def rate(num_key, den_key):
        den = len(acc[den_key])
        return (sum(acc[num_key]) / den) if den else None

    def mean(key):
        return (sum(acc[key]) / len(acc[key])) if acc[key] else None

    m = {
        "false_bargein_rate": rate("false_bargein", "silent_total"),      # HEADLINE (lower better)
        "correct_silence_rate": (1 - rate("false_bargein", "silent_total")) if acc["silent_total"] else None,
        "correct_response_rate": rate("responded", "respond_total"),      # higher better
        "missed_response_rate": (1 - rate("responded", "respond_total")) if acc["respond_total"] else None,  # 2D under-respond axis (lower better)
        "wrong_addressee_rate": rate("wrong_addressee", "qlike_silent_total"),
        "stop_success_rate": rate("stopped", "stop_total"),
        "continue_when_should_rate": rate("continued", "continue_total"),
        "mean_response_latency_s": mean("response_latency"),
        "mean_stop_latency_s": mean("stop_latency"),
        "addressee_f1": _f1(sum(acc["addr_tp"]), sum(acc["addr_fp"]), sum(acc["addr_fn"]))
                        if acc["addr_tp"] or acc["addr_fp"] or acc["addr_fn"] else None,
        "n_sessions": len(sessions),
    }
    # per-cell false-barge-in / response for the breakdown table
    m["per_cell"] = {}
    for cell, d in per_cell.items():
        st, rt = len(d["silent_total"]), len(d["respond_total"])
        ct = len(d["continue_total"])
        m["per_cell"][cell] = {
            "n": st + rt + len(d["stop_total"]) + ct,
            "fbr": (sum(d["false_bargein"]) / st) if st else None,
            "resp_rate": (sum(d["responded"]) / rt) if rt else None,
            "continue_rate": (sum(d["continued"]) / ct) if ct else None,
        }
    return m


def simplified_der(session: Session, out: SystemOutput, frame: float = 0.1) -> Optional[float]:
    """Best-effort frame-based diarization error with greedy 1-1 label mapping.
    Returns None if the system produced no diarization. Not a substitute for dscore/md-eval."""
    if not out.diar_pred:
        return None
    import numpy as np
    T = max([u.end for u in session.utterances] + [e for _, e, _ in out.diar_pred] + [0.0])
    n = int(T / frame) + 1
    times = (np.arange(n) + 0.5) * frame

    def label_at(spans, t):
        for a, b, spk in spans:
            if a <= t < b:
                return spk
        return None

    gold_spans = [(u.start, u.end, u.speaker) for u in session.utterances if u.speaker != session.agent_id]
    g = [label_at(gold_spans, t) for t in times]
    p = [label_at([(a, b, s) for a, b, s in out.diar_pred], t) for t in times]
    # greedy label mapping by co-occurrence
    from collections import Counter
    co = Counter((gi, pi) for gi, pi in zip(g, p) if gi is not None and pi is not None)
    mapping = {}
    for (gi, pi), _ in co.most_common():
        if pi not in mapping.values() and gi not in mapping:
            mapping[gi] = pi
    err = sum(1 for gi, pi in zip(g, p) if gi is not None and (mapping.get(gi) != pi))
    tot = sum(1 for gi in g if gi is not None)
    return (err / tot) if tot else None
