# -*- coding: utf-8 -*-
"""Core data structures for MPFD-Bench.

A *session* is one multi-party conversation (agent + K humans). Each *utterance* carries who
said it, when, its text, and — crucially — its **addressee** and whether it is addressed to the
agent. From the utterances we derive **gold events**: the expected agent behavior at each point
(RESPOND / SILENT / STOP / BACKCHANNEL). A **system** under test produces the intervals in which
the agent actually spoke (+ optional addressee/diarization predictions); metrics compare the two.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json


class Action(str, Enum):
    RESPOND = "respond"        # agent should start speaking (addressed, a gap is available)
    SILENT = "silent"          # agent should stay silent (inter-human / third-party, agent not already speaking)
    STOP = "stop"              # agent should stop (addressed speaker barged in while agent spoke)
    BACKCHANNEL = "backchannel"  # a listener backchannel during agent speech -> agent continues
    CONTINUE = "continue"      # non-addressing human speech while agent is speaking -> agent keeps going


@dataclass
class Utterance:
    utt_id: str
    speaker: str               # speaker label, e.g. "Alice" / "A"
    start: float               # seconds
    end: float
    text: str = ""
    addressee: Optional[str] = None   # who this utterance is directed at (speaker label / "agent" / "group")
    is_for_agent: bool = False        # convenience: addressee == the agent
    dialogue_act: str = "statement"   # statement / question / request / backchannel / ...


@dataclass
class Session:
    session_id: str
    cell: str                  # which test cell this session instantiates (see cells.py)
    speakers: List[str]        # human speaker labels
    agent_id: str              # the agent's label (e.g. "Agent")
    utterances: List[Utterance] = field(default_factory=list)
    sample_rate: int = 16000
    audio_path: Optional[str] = None   # rendered mixed far-field wav (optional, set after TTS)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Session":
        utts = [Utterance(**u) for u in d.get("utterances", [])]
        d = {k: v for k, v in d.items() if k != "utterances"}
        return Session(utterances=utts, **d)


@dataclass
class GoldEvent:
    """Expected agent behavior anchored to a reference utterance/window."""
    ref_utt_id: str
    start: float               # window in which the expectation is evaluated
    end: float
    action: Action
    is_for_agent: bool
    dialogue_act: str = "statement"


@dataclass
class SystemOutput:
    """What a system-under-test returns for one session."""
    session_id: str
    agent_segments: List[Tuple[float, float]] = field(default_factory=list)  # when the agent spoke
    addressee_pred: Dict[str, bool] = field(default_factory=dict)   # utt_id -> predicted is_for_agent
    diar_pred: List[Tuple[float, float, str]] = field(default_factory=list)  # (start,end,speaker) optional


# ---------- interval helpers ----------
def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def spoke_in(segments: List[Tuple[float, float]], window: Tuple[float, float], min_dur: float = 0.05) -> bool:
    """Did the agent produce >= min_dur of speech overlapping `window`?"""
    return sum(overlap(s, window) for s in segments) >= min_dur


def first_onset_after(segments: List[Tuple[float, float]], t: float,
                      horizon: float = 3.0) -> Optional[float]:
    """Earliest agent speech onset in (t, t+horizon]; None if it stayed silent."""
    cands = [s[0] for s in segments if t < s[0] <= t + horizon]
    return min(cands) if cands else None
