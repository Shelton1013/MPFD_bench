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
