# -*- coding: utf-8 -*-
"""The MPFD controller system (P2): base speak-onsets + addressee gate -> gated SystemOutput.

Onset providers (where the "should I speak" moments come from):
  • gold_dyadic  : after every human turn (CPU, no audio) — models a dyadic base that answers
                   every turn; use to test the gate purely.
  • vad          : energy VAD on the rendered audio + dyadic policy (real perception, CPU).
  • fo_cache     : Freeze-Omni's cached speak onsets (the real base decisions).

Addressee source: 'oracle' (gold is_for_agent) or a trained AddresseeClassifier.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, List

from ..controller.features import session_examples
from ..controller.gate import gate_onsets
from ..schema import Session, SystemOutput


# ---------- onset providers ----------
def gold_dyadic_onsets(session: Session) -> List[float]:
    return [round(u.end + 0.3, 2) for u in session.utterances if u.speaker != session.agent_id]


def vad_onsets(session: Session) -> List[float]:
    from .real import vad_dyadic_baseline
    return [s[0] for s in vad_dyadic_baseline(session).agent_segments]


def fo_cache_onsets(cache_dir: str) -> Callable[[Session], List[float]]:
    def provider(session: Session) -> List[float]:
        p = os.path.join(cache_dir, f"{session.session_id}.json")
        return json.load(open(p)).get("onsets", []) if os.path.exists(p) else []
    return provider


ONSET_PROVIDERS = {"gold_dyadic": gold_dyadic_onsets, "vad": vad_onsets}


# ---------- addressee prediction ----------
def _addressee_pred(session: Session, addressee, use_wakeword: bool = True) -> Dict[str, bool]:
    if addressee == "oracle":
        return {u.utt_id: u.is_for_agent for u in session.utterances if u.speaker != session.agent_id}
    # a trained AddresseeClassifier
    pred = {}
    for x, _y, uid in session_examples(session, use_wakeword=use_wakeword):
        pred[uid] = bool(addressee.predict([x])[0])
    return pred


class MPFDController:
    def __init__(self, onset_provider: Callable[[Session], List[float]], addressee,
                 use_wakeword: bool = True, resp_dur: float = 1.5):
        self.onsets = onset_provider
        self.addressee = addressee            # "oracle" or an AddresseeClassifier
        self.use_wakeword = use_wakeword
        self.resp_dur = resp_dur

    def __call__(self, session: Session) -> SystemOutput:
        addr = _addressee_pred(session, self.addressee, self.use_wakeword)
        seg = gate_onsets(session, self.onsets(session), addr, resp_dur=self.resp_dur)
        return SystemOutput(session.session_id, agent_segments=seg, addressee_pred=addr)
