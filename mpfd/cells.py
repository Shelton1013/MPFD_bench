# -*- coding: utf-8 -*-
"""The MPFD-Bench test cells and the rule that turns a session into gold events.

Each cell is an interaction type that exposes a specific multi-party failure mode. The names and
the T/I/R codes follow the full-duplex interaction ontology (arXiv:2606.19453), extended from
"third-party = ignore" to genuine multi-party participation.
"""
from __future__ import annotations

from typing import Dict, List

from .schema import Action, GoldEvent, Session, Utterance

# cell_name -> (short description, probes)
CELLS: Dict[str, str] = {
    "inter_human":        "two humans converse with each other; agent must STAY SILENT (false-barge-in test)",
    "overlapping_humans": "two humans talk over each other; agent must attribute both and not misfire",
    "addressed_turn":     "a human addresses the agent with a clean gap; agent SHOULD respond",
    "addressee_switch":   "conversation moves from human-human to addressing the agent; respond only then",
    "wrong_addressee":    "a question-like utterance addressed to another human; agent must NOT answer",
    "addressed_bargein":  "agent is speaking; the addressed human interrupts; agent should STOP (<~200ms)",
    "nonaddressing_overlap": "agent speaking; a different human speaks to a third human; agent CONTINUE",
    "backchannel":        "a listener says 'uh-huh' while agent speaks; agent should CONTINUE",
    "third_party":        "a non-participant/background voice; agent should IGNORE",
}

# response tolerance: an addressed turn should be answered within this window after it ends (s)
RESPONSE_HORIZON = 3.0
# barge-in: after the addressed interrupt onset, the agent should stop within this (s)
STOP_HORIZON = 1.0


def gold_events(session: Session) -> List[GoldEvent]:
    """Derive the expected agent behavior for each human utterance in the session.

    Rule of thumb:
      - utterance addressed to the agent  -> RESPOND, evaluated in the window just after it ends
      - inter-human / third-party         -> SILENT, evaluated across the utterance span
      - the addressed speaker interrupting -> STOP (marked by dialogue_act == 'bargein')
      - a listener backchannel            -> the agent should CONTINUE (BACKCHANNEL expectation)
    """
    agent_turns = [(u.start, u.end) for u in session.utterances if u.speaker == session.agent_id]

    def during_agent_speech(u) -> bool:
        return any(a < u.end and u.start < b for a, b in agent_turns)

    events: List[GoldEvent] = []
    for u in session.utterances:
        if u.speaker == session.agent_id:
            continue  # the agent's own scripted turns are not gold *for the agent's decision*
        if u.dialogue_act == "bargein" and u.is_for_agent:
            events.append(GoldEvent(u.utt_id, u.start, u.start + STOP_HORIZON,
                                    Action.STOP, True, u.dialogue_act))
        elif u.dialogue_act == "backchannel" and not u.is_for_agent:
            events.append(GoldEvent(u.utt_id, u.start, u.end, Action.BACKCHANNEL, False, u.dialogue_act))
        elif u.is_for_agent:
            events.append(GoldEvent(u.utt_id, u.end, u.end + RESPONSE_HORIZON,
                                    Action.RESPOND, True, u.dialogue_act))
        elif during_agent_speech(u):
            # non-addressing human speech while the agent is legitimately speaking -> keep going,
            # NOT a false-barge-in opportunity (the agent is already correctly holding the floor)
            events.append(GoldEvent(u.utt_id, u.start, u.end, Action.CONTINUE, False, u.dialogue_act))
        else:
            events.append(GoldEvent(u.utt_id, u.start, u.end, Action.SILENT, False, u.dialogue_act))
    return events
