# -*- coding: utf-8 -*-
"""Expand data/paraphrases.json with a LOCAL Qwen3 model (no API, offline, reproducible).

The addressee label is a property of the CATEGORY, not the wording, so generating more surface
variants never changes gold labels — BUT we still enforce each category's lexical invariant with a
validator (reject, don't trust) so a stray generation can't silently flip a label:
  agent_q_i0  : must contain a wake-word (assistant/agent), no human vocative
  agent_q_i1/i2/sem : NO wake-word, no human vocative  (addressed to agent by position/content)
  human_q_i0  : must start with the literal template "{name}," (filled with the addressee at gen time)
  human_q_i1/i2/sem : NO wake-word, NO "{name}", no capitalized vocative (addressed to a human)

Needs transformers>=4.51 for Qwen3 (the freeze-omni env's 4.45 is too old — use a fresh env), e.g.:
  pip install "transformers>=4.51" accelerate torch
  python scripts/13_gen_paraphrases.py --model Qwen/Qwen3-8B --per_category 80 --out data/paraphrases.json
Faster alternative: serve with vLLM and point an OpenAI client at it (not required).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WAKE = re.compile(r"\b(assistant|agent)\b", re.I)
VOCATIVE = re.compile(r"^[a-z]+,\s")               # a name-comma opener (after lowercasing)
NAME_TOKEN = "{name}"

# category -> (instruction, validator-key). Instruction describes the SPEECH ACT precisely.
SPECS = {
    "agent_q_i0": ("short spoken questions or requests a person says TO AN AI ASSISTANT in a meeting, "
                   "each STARTING WITH a wake-word like 'assistant,' or 'hey assistant,'. no human names.",
                   "agent_wake"),
    "agent_q_i1": ("short polite questions/requests directed at an ai assistant, WITHOUT any wake-word "
                   "and WITHOUT any name (e.g. 'could you summarize the last point', 'what is the status'). "
                   "they read as directed to the assistant by their content.",
                   "agent_nomarker"),
    "agent_q_i2": ("very short follow-up questions someone asks right after the assistant just spoke, "
                   "with no name and no wake-word (e.g. 'can you expand on that', 'why is that', 'go on', "
                   "'what about the second one').",
                   "agent_nomarker"),
    "agent_q_sem": ("short requests that ONLY an ai assistant/system could fulfill, with no wake-word and "
                    "no name (e.g. 'what is on the agenda', 'pull up the notes', 'set a reminder for the demo', "
                    "'summarize the thread'). the assistant-directedness comes purely from the content.",
                    "agent_nomarker"),
    "human_q_i0": ("short questions one coworker asks ANOTHER PERSON BY NAME, each STARTING WITH the literal "
                   "token '{name},' exactly (do not invent a name) e.g. '{name}, did you send the report'.",
                   "human_named"),
    "human_q_i1": ("short casual questions one coworker asks another person, with NO name and NO wake-word "
                   "(e.g. 'did you send the report', 'are you joining the call').",
                   "human_nomarker"),
    "human_q_i2": ("very short topic-continuation questions between two coworkers, no name, no wake-word "
                   "(e.g. 'and then what happened', 'did you tell her yet', 'so what is the plan').",
                   "human_nomarker"),
    "human_q_sem": ("short PERSONAL questions one coworker asks another about that person's own life/work, "
                    "no name, no wake-word (e.g. 'did you finish your part', 'how was your weekend', "
                    "'are you feeling better'). clearly directed at a human by content.",
                    "human_nomarker"),
    "agent_prior": ("short things an ai assistant says when recapping a meeting (e.g. 'here is a quick summary "
                    "of the three points', 'the two options are on the slide').", "loose"),
    "human_stmt": ("short statements a coworker makes in a meeting, no question (e.g. 'i handed off my part "
                   "yesterday', 'the deadline moved to next month').", "loose"),
    "third_party": ("short public-address / background announcements unrelated to the meeting (e.g. "
                    "'boarding for flight 204 is now open', 'coffee is ready in the kitchen').", "loose"),
}


def _valid(text, key):
    t = text.strip().strip('"\'-').strip().lower()
    if not (2 <= len(t.split()) <= 14) or t.endswith((":",)) or t.startswith(("here are", "sure", "1.")):
        return None
    has_wake = bool(WAKE.search(t))
    if key == "agent_wake":
        return t if (has_wake and NAME_TOKEN not in t) else None
    if key == "agent_nomarker":
        return t if (not has_wake and NAME_TOKEN not in t and not VOCATIVE.match(t)) else None
    if key == "human_named":
        return t if t.startswith("{name},") and not has_wake else None
    if key == "human_nomarker":
        return t if (not has_wake and NAME_TOKEN not in t and not VOCATIVE.match(t)) else None
    return t if not has_wake else None       # loose


def generate(model, tok, instruction, n, temperature, max_new_tokens, device):
    import torch
    sys_p = ("you generate short spoken utterances for a multi-party meeting dataset. "
             "output ONLY the utterances, one per line, all lowercase, no numbering, no quotes, "
             "no commentary. keep the literal token {name} verbatim when the instruction uses it.")
    user = f"generate {n} diverse variants of: {instruction}\nremember: one per line, lowercase, nothing else."
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
    try:   # Qwen3 hybrid models accept enable_thinking; instruct-only (e.g. -Instruct-2507) do not
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    out = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=0.9,
                         max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return [ln for ln in (l.strip() for l in gen.splitlines()) if ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="data/paraphrases.json")
    ap.add_argument("--per_category", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--only", default=None, help="comma-separated subset of categories to (re)generate")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    device = model.device

    bank = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    cats = args.only.split(",") if args.only else list(SPECS)
    for cat in cats:
        instr, key = SPECS[cat]
        raw = generate(model, tok, instr, args.per_category, args.temperature, args.max_new_tokens, device)
        kept = [v for v in (_valid(r, key) for r in raw) if v]
        existing = [x for x in bank.get(cat, []) if isinstance(x, str)]
        merged = sorted(set(existing) | set(kept))
        print(f"  {cat:14s} gen={len(raw):3d} valid={len(kept):3d} rejected={len(raw)-len(kept):3d} "
              f"total={len(merged):3d}")
        bank[cat] = merged

    json.dump(bank, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote -> {args.out}. Regenerate the dataset to use it:")
    print("  python scripts/01_generate_synth.py --out data/synthetic --n_per_cell 50 --n_graded_per_tier 120")
    print("SPOT-CHECK a sample of each category by eye before trusting the curve.")


if __name__ == "__main__":
    main()
