# -*- coding: utf-8 -*-
"""Template-based N-party dialogue generator: one Session per test cell, with gold addressee tags
and realistic timing. No LLM required (deterministic templates); an optional LLM hook can be
added for lexical variety. Output feeds the TTS/compose stage and the metrics harness.
"""
from __future__ import annotations

import random
from typing import List

from ..schema import Session, Utterance

# small content banks (kept simple; swap in an LLM for variety)
Q_TO_AGENT = ["assistant, summarize the last point", "hey assistant, what time is the meeting",
              "assistant, can you send the notes", "assistant, what did we decide"]
HUMAN_HUMAN = [("did you finish the report", "almost, one section left"),
               ("are you joining the call", "yes in five minutes"),
               ("did you see the email", "not yet, forward it to me")]
AGENT_LONG = ["sure, here is a quick summary of what we covered so far in the discussion",
              "let me walk you through the three action items from the meeting"]
BACKCH = ["uh-huh", "mm", "right", "okay"]
THIRD = ["boarding for flight 204 is now open", "coffee is ready in the kitchen"]


def _u(i, spk, t0, dur, text, addressee=None, agent="Agent", da="statement"):
    return Utterance(utt_id=f"u{i}", speaker=spk, start=round(t0, 2), end=round(t0 + dur, 2),
                     text=text, addressee=addressee, is_for_agent=(addressee == agent), dialogue_act=da)


# --- graded addressing banks (Track A). Each session carries one agent-addressed question
#     (POSITIVE) + one human-addressed question (NEGATIVE), realized at increasing implicitness so
#     addressee-F1 degrades along I0->I2 (see docs/BENCHMARK_SPEC.md §5.2):
#       I0 explicit   : explicit lexical marker — wake-word (agent) / vocative name (human). Regex-solvable.
#       I1 vocative-dropped : marker removed; conversational POSITION (prev speaker) still reveals it.
#       I2 contextual : HALF supportive-position, HALF misleading-position where the addressee is only
#                       recoverable from CONTENT (agent-answerable vs personal-to-a-human). Neither a
#                       wake-word regex nor a shallow "who spoke last" cue suffices -> real inference.
AGENT_Q = {  # agent-addressed, position-cued (follows the agent's own turn)
    "I0": ["assistant, summarize the last point", "assistant, what time is the sync",
           "hey assistant, send the notes"],
    "I1": ["could you summarize the last point", "what time is the sync", "please send the notes"],
    "I2": ["can you expand on that", "what about the second one", "why is that", "go on"],
}
HUMAN_Q = {  # human-addressed, position-cued (follows a human turn)
    "I0": ["Bob, did you send the report", "Bob, are you joining the call"],
    "I1": ["did you send the report", "are you joining the call"],
    "I2": ["and then what happened", "did you tell her yet"],
}
# I2 MISLEADING pool: addressee defensible by CONTENT alone, while conversational position lies.
AGENT_Q_SEM = ["what is on the agenda", "pull up the notes", "what did we decide last time",
               "how much time is left"]                                  # only the agent answers these
HUMAN_Q_SEM = ["did you finish your part", "are you feeling better", "how was your weekend"]  # personal to a human
AGENT_PRIOR = ["here is a quick summary of the three points we discussed",
               "the two options are both on the slide now"]
GRADED_TIERS = ("I0", "I1", "I2")


def make_graded_session(idx: int, rng: random.Random, tier: str = "I0",
                        speakers=("Alice", "Bob"), agent="Agent") -> Session:
    """One session for the addressee-difficulty CURVE. The agent-addressed (POSITIVE) and human-
    addressed (NEGATIVE) questions are tagged with `implicitness=tier` so 12_addressee_curve.py can
    bucket addressee-F1 by tier. utt_ids are made session-unique (prefixed) so downstream tagging
    never collides across sessions."""
    A, B = speakers[0], speakers[1]
    us: List[Utterance] = []
    t = [round(rng.uniform(0.3, 0.8), 2)]
    sid = f"addressee_graded_{tier}_{idx:04d}"

    def add(spk, dur, text, addr, da, tag=False):
        u = _u(len(us), spk, t[0], dur, text, addr, agent, da)
        u.utt_id = f"{sid}_u{len(us)}"           # session-unique id (avoid cross-session tag collisions)
        if tag:
            u.implicitness = tier
        us.append(u)
        t[0] = round(t[0] + dur + 0.35 + rng.uniform(-0.1, 0.2), 2)

    misleading = (tier == "I2" and rng.random() < 0.5)
    if not misleading:
        # supportive position: human topic, human-Q(neg, after human), agent turn, agent-Q(pos, after agent)
        add(A, 1.5, rng.choice(HUMAN_HUMAN)[0], B, "statement")
        add(B, 1.5, rng.choice(HUMAN_Q[tier]), A, "question", tag=True)          # NEGATIVE
        add(agent, 2.2, rng.choice(AGENT_PRIOR), A, "statement")
        add(A, 1.6, rng.choice(AGENT_Q[tier]), agent, "question", tag=True)       # POSITIVE
    else:
        # misleading position: neg follows the agent turn, pos follows a human turn; label from CONTENT
        add(agent, 2.2, rng.choice(AGENT_PRIOR), A, "statement")
        add(B, 1.6, rng.choice(HUMAN_Q_SEM), A, "question", tag=True)             # NEGATIVE, prev=agent (lies)
        add(A, 1.5, rng.choice(HUMAN_HUMAN)[1], B, "statement")
        add(B, 1.6, rng.choice(AGENT_Q_SEM), agent, "question", tag=True)         # POSITIVE, prev=human (lies)

    return Session(session_id=sid, cell="addressee_graded",
                   speakers=list(speakers), agent_id=agent, utterances=us)


def make_session(cell: str, idx: int, rng: random.Random,
                 speakers=("Alice", "Bob"), agent="Agent") -> Session:
    A, B = speakers[0], speakers[1]
    us: List[Utterance] = []
    t = round(rng.uniform(0.3, 0.8), 2)
    jit = lambda: rng.uniform(-0.1, 0.2)

    if cell == "inter_human":
        q, a = rng.choice(HUMAN_HUMAN)
        us += [_u(0, A, t, 1.6, q, B, agent, "question")]
        t += 1.6 + 0.3 + jit()
        us += [_u(1, B, t, 1.4, a, A, agent, "statement")]

    elif cell == "overlapping_humans":
        q, a = rng.choice(HUMAN_HUMAN)
        us += [_u(0, A, t, 1.8, q, B, agent, "question")]
        us += [_u(1, B, t + 1.0, 1.4, a, A, agent, "statement")]   # overlaps A

    elif cell == "addressed_turn":
        us += [_u(0, A, t, 1.7, rng.choice(Q_TO_AGENT), agent, agent, "question")]

    elif cell == "addressee_switch":
        q, a = rng.choice(HUMAN_HUMAN)
        us += [_u(0, A, t, 1.5, q, B, agent, "question")]; t += 1.5 + 0.3
        us += [_u(1, B, t, 1.3, a, A, agent, "statement")]; t += 1.3 + 0.4
        us += [_u(2, A, t, 1.6, rng.choice(Q_TO_AGENT), agent, agent, "question")]

    elif cell == "wrong_addressee":
        us += [_u(0, A, t, 1.8, rng.choice(HUMAN_HUMAN)[0], B, agent, "question")]  # Q, but to Bob

    elif cell == "addressed_bargein":
        us += [_u(0, agent, t, 3.5, rng.choice(AGENT_LONG), A, agent, "statement")]  # agent speaking
        us += [_u(1, A, t + 1.5, 1.2, "wait, stop", agent, agent, "bargein")]        # addressed barge-in

    elif cell == "nonaddressing_overlap":
        us += [_u(0, agent, t, 3.5, rng.choice(AGENT_LONG), A, agent, "statement")]
        us += [_u(1, B, t + 1.5, 1.2, rng.choice(HUMAN_HUMAN)[0], A, agent, "question")]  # B->A, not agent

    elif cell == "backchannel":
        us += [_u(0, agent, t, 3.5, rng.choice(AGENT_LONG), A, agent, "statement")]
        us += [_u(1, B, t + 1.5, 0.5, rng.choice(BACKCH), agent, agent, "backchannel")]  # listener BC
        us[-1].is_for_agent = False  # a backchannel is support, not an addressed request

    elif cell == "third_party":
        us += [_u(0, "TV", t, 2.2, rng.choice(THIRD), None, agent, "statement")]  # non-participant

    else:
        raise ValueError(f"unknown cell {cell}")

    return Session(session_id=f"{cell}_{idx:04d}", cell=cell, speakers=list(speakers),
                   agent_id=agent, utterances=us)
