# -*- coding: utf-8 -*-
"""Baseline "systems" for MPFD-Bench. A *system* is a callable `session -> SystemOutput`.

Two CPU-only reference systems let the harness run end-to-end without any model:
  • `oracle_system`  — behaves correctly (sanity check: should score ~perfectly).
  • `naive_dyadic_system` — simulates a dyadic full-duplex model fed the mixed multi-party
    stream: it cannot tell the addressee, so it answers *every* utterance and stops for *any*
    voice during its turn. This reproduces the failure MPFD-Bench is designed to expose.

Real-model baselines (server, GPU) implement the same callable signature:
  • dyadic frozen FD model (Freeze-Omni / MiniCPM-o 4.5) fed the mixed wav;
  • modular L0 = pyannote streaming diarization + heuristic addressee (name/wake-word +
    last-speaker) gating the FD model.
See docs and `mpfd/baselines/real.py` stubs.
"""
from __future__ import annotations

from ..cells import gold_events
from ..schema import Action, Session, SystemOutput


def oracle_system(session: Session) -> SystemOutput:
    seg = []
    ev = {e.ref_utt_id: e for e in gold_events(session)}
    # agent's own scripted turns, truncated at an addressed barge-in
    stops = [e for e in ev.values() if e.action == Action.STOP]
    for u in session.utterances:
        if u.speaker != session.agent_id:
            continue
        end = u.end
        for st in stops:
            if u.start <= st.start <= u.end:
                end = min(end, st.start + 0.2)
        seg.append((u.start, round(end, 2)))
    # responses to addressed turns
    for e in ev.values():
        if e.action == Action.RESPOND:
            seg.append((round(e.start + 0.3, 2), round(e.start + 0.3 + 1.6, 2)))
    addr = {u.utt_id: u.is_for_agent for u in session.utterances if u.speaker != session.agent_id}
    return SystemOutput(session.session_id, agent_segments=sorted(seg), addressee_pred=addr)


def naive_dyadic_system(session: Session) -> SystemOutput:
    """Cannot distinguish speakers or addressee: answers every human utterance, and stops its own
    turn whenever any human speaks during it. Predicts everything as addressed-to-agent."""
    seg = []
    humans = [u for u in session.utterances if u.speaker != session.agent_id]
    # its own turns, truncated at the FIRST human voice overlapping (any speaker)
    for u in session.utterances:
        if u.speaker != session.agent_id:
            continue
        end = u.end
        for h in humans:
            if u.start < h.start < u.end:
                end = min(end, h.start + 0.2)
        seg.append((u.start, round(end, 2)))
    # answers EVERY human utterance (regardless of addressee) shortly after it ends
    for h in humans:
        if h.dialogue_act in ("question", "request", "statement"):
            seg.append((round(h.end + 0.3, 2), round(h.end + 0.3 + 1.4, 2)))
    addr = {u.utt_id: True for u in humans}   # assumes everything is for it
    return SystemOutput(session.session_id, agent_segments=sorted(seg), addressee_pred=addr)


SYSTEMS = {"oracle": oracle_system, "naive_dyadic": naive_dyadic_system}
