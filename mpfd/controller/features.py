# -*- coding: utf-8 -*-
"""Addressee features for one utterance in context. Lexical + context (+ optional prosody).

Kept deliberately simple/interpretable for v1 (logistic regression). Note: `has_wakeword` makes
the synthetic (wake-word) set trivial — evaluate the real claim on the implicit subset + AMI,
and report an ablation WITHOUT this feature.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..schema import Session, Utterance

WAKE = re.compile(r"\b(assistant|agent|hey assistant|ok assistant)\b", re.I)
WH = re.compile(r"\b(what|when|where|who|why|how|which|can you|could you|would you|please)\b", re.I)
SECOND = re.compile(r"\b(you|your|you're)\b", re.I)

FEATURE_NAMES = ["has_wakeword", "is_question", "wh_or_request", "second_person",
                 "len_words_norm", "speaker_changed", "prev_for_agent", "prev_was_agent", "bias"]


def utt_features(u: Utterance, prev: Optional[Utterance], prev_for_agent: float,
                 prev_was_agent: float = 0.0, use_wakeword: bool = True) -> List[float]:
    """Interpretable addressee features. `has_wakeword` is the lexical crutch that trivializes I0;
    `prev_was_agent` is the CONTEXT cue (a follow-up right after the agent spoke is likely for the
    agent) that must carry the harder I1/I2 tiers where no name/wake-word is present."""
    text = u.text or ""
    nwords = len(text.split())
    f = [
        1.0 if (use_wakeword and WAKE.search(text)) else 0.0,
        1.0 if (u.dialogue_act == "question" or text.strip().endswith("?")) else 0.0,
        1.0 if WH.search(text) else 0.0,
        1.0 if SECOND.search(text) else 0.0,
        min(nwords, 20) / 20.0,
        1.0 if (prev is not None and prev.speaker != u.speaker) else 0.0,
        float(prev_for_agent),
        float(prev_was_agent),
        1.0,  # bias
    ]
    return f


def session_examples(session: Session, use_wakeword: bool = True):
    """Yield (features, label, utt_id) for each non-agent utterance, threading context.
    Context `prev_for_agent` uses the GOLD previous addressee during training (teacher forcing);
    `prev_was_agent` marks that the immediately preceding turn was spoken by the agent."""
    prev = None
    prev_for_agent = 0.0
    for u in session.utterances:
        if u.speaker == session.agent_id:
            prev = u
            continue
        prev_was_agent = 1.0 if (prev is not None and prev.speaker == session.agent_id) else 0.0
        x = utt_features(u, prev, prev_for_agent, prev_was_agent, use_wakeword)
        yield x, (1 if u.is_for_agent else 0), u.utt_id
        prev = u
        prev_for_agent = 1.0 if u.is_for_agent else 0.0
