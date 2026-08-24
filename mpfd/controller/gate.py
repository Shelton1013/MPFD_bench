# -*- coding: utf-8 -*-
"""Gating: keep a base speak-onset only if the utterance that triggered it is addressed to the
agent. This is what turns a dyadic model's speak decisions into multi-party-correct behavior."""
from __future__ import annotations

from typing import Dict, List

from ..schema import Session


def gate_onsets(session: Session, onsets: List[float], addressee_pred: Dict[str, bool],
                resp_dur: float = 1.5, trigger_window: float = 3.0) -> List[tuple]:
    """For each speak onset t, find the triggering human utterance (active at t, or the most
    recent one ending within `trigger_window` before t). Keep the onset (-> a speak span) only if
    that utterance is predicted addressed-to-agent; otherwise suppress it (no barge-in)."""
    humans = [u for u in session.utterances if u.speaker != session.agent_id]
    seg = []
    for t in onsets:
        trig = None
        for u in humans:
            active = u.start <= t <= u.end
            recent = u.end <= t <= u.end + trigger_window
            if active or recent:
                if trig is None or u.end > trig.end:
                    trig = u
        keep = True if trig is None else bool(addressee_pred.get(trig.utt_id, True))
        if keep:
            seg.append((round(t, 2), round(t + resp_dur, 2)))
    return sorted(seg)
