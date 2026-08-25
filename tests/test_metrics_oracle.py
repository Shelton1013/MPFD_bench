# -*- coding: utf-8 -*-
"""Regression tests: the two 2D headline axes (over-speak: FBR / wrong-addressee; under-respond:
missed-response) MUST read perfect for a correct system, and must stay correct under compressed /
real-model timing where the old onset-only logic misfired.

Run: python -m pytest tests/test_metrics_oracle.py   (or just execute this file)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpfd.baselines import naive_dyadic_system, oracle_system
from mpfd.cells import CELLS
from mpfd.metrics import aggregate, score_session
from mpfd.schema import Session, SystemOutput, Utterance
from mpfd.synth.generate_dialogues import make_session


def _agg(sessions, system):
    return aggregate(sessions, {s.session_id: system(s) for s in sessions})


def test_oracle_perfect_on_clean_synthetic():
    rng = random.Random(0)
    sessions = [make_session(c, i, rng) for c in CELLS for i in range(30)]
    m = _agg(sessions, oracle_system)
    assert m["false_bargein_rate"] == 0.0, m["false_bargein_rate"]
    assert m["wrong_addressee_rate"] == 0.0, m["wrong_addressee_rate"]
    assert m["missed_response_rate"] == 0.0, m["missed_response_rate"]
    assert m["correct_response_rate"] == 1.0
    assert m["addressee_f1"] == 1.0


def test_naive_dyadic_still_fails_loudly():
    """Sanity: the fixes must NOT make the broken dyadic baseline look good."""
    rng = random.Random(1)
    sessions = [make_session(c, i, rng) for c in CELLS for i in range(30)]
    m = _agg(sessions, naive_dyadic_system)
    assert m["false_bargein_rate"] > 0.2          # it barges in
    assert m["wrong_addressee_rate"] > 0.9         # it answers questions meant for others
    assert m["missed_response_rate"] == 0.0        # it over-responds, so it never MISSES


def test_wrong_addressee_not_misattributed_to_a_legit_response():
    """Compressed timing: a human->human question, then immediately a human->agent turn the agent
    correctly answers. The correct response onset lands inside the question's after-window. The
    OLD metric flagged this as wrong-addressee; the fixed metric must read 0."""
    utts = [
        Utterance("u0", "Alice", 0.0, 2.0, "did you send it", addressee="Bob",
                  is_for_agent=False, dialogue_act="question"),          # SILENT question -> Bob
        Utterance("u1", "Alice", 2.2, 3.8, "assistant, whats next", addressee="Agent",
                  is_for_agent=True, dialogue_act="question"),           # addressed -> Agent
    ]
    s = Session("adv", "addressee_switch", ["Alice", "Bob"], "Agent", utts)
    # a correct system: silent through u0, answers u1 at 4.1s (inside u0's (2.0, 5.0] after-window)
    out = SystemOutput("adv", agent_segments=[(4.1, 5.7)],
                       addressee_pred={"u0": False, "u1": True})
    raw = score_session(s, out)
    assert sum(raw["wrong_addressee"]) == 0, "legit response misattributed as wrong-addressee"
    assert sum(raw["responded"]) == len(raw["responded"]) == 1


def test_response_counted_when_agent_already_speaking_across_boundary():
    """Real models emit merged spans: a response turn can start just before the addressed turn ends
    and continue past it, leaving no onset strictly after e.start. The fixed metric counts it."""
    utts = [
        Utterance("u0", "Alice", 0.0, 2.0, "assistant, go", addressee="Agent",
                  is_for_agent=True, dialogue_act="request"),
    ]
    s = Session("rsp", "addressed_turn", ["Alice", "Bob"], "Agent", utts)
    # agent speech spans 1.9 -> 3.5: overlaps the response window (2.0, 5.0] but has no onset > 2.0
    out = SystemOutput("rsp", agent_segments=[(1.9, 3.5)], addressee_pred={"u0": True})
    raw = score_session(s, out)
    assert sum(raw["responded"]) == 1, "already-speaking response undercounted"


if __name__ == "__main__":
    test_oracle_perfect_on_clean_synthetic()
    test_naive_dyadic_still_fails_loudly()
    test_wrong_addressee_not_misattributed_to_a_legit_response()
    test_response_counted_when_agent_already_speaking_across_boundary()
    print("all metric regression tests passed")
