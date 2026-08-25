# -*- coding: utf-8 -*-
"""Track D — inject agent-directed events into REAL multi-party audio (see docs/BENCHMARK_SPEC.md §5.1).

AMI sessions (from convert_ami.py) are real human meetings: real overlap, real room, and every
human utterance is_for_agent=False -> the agent must stay silent. That only tests the OVER-SPEAK
half. Track D splices TTS utterances into the real mix at chosen gaps, with clean labels:
  • agent-addressed  (from agent_q_* banks, is_for_agent=True)  -> gold RESPOND  (agent SHOULD answer)
  • human-addressed  (from human_q_* banks, is_for_agent=False) -> gold SILENT   (agent must NOT answer)
So we get REAL acoustics + CLEAN labels for BOTH halves (FBR on real cross-talk + correct-response on
injected agent events + wrong-addressee on injected human events).

Known limitation (BENCHMARK_SPEC §5.1): the TTS voice has an acoustic seam vs. real speakers; a model
could cheat by keying on "clean voice = respond". We MEASURE the seam: wrong-addressee on the injected
HUMAN-addressed events must stay high — if the model answers those too, it isn't cheating on voice.

Run in the piper env (offline TTS). Needs AMI sessions + piper voices.
  python scripts/20_inject_agent.py --ami data/ami --out data/injected \
      --model_dir ~/piper_voices --per_session_agent 2 --per_session_human 2
Then score the REAL model / controller on data/injected exactly like any rendered set.
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.schema import Session, Utterance
from mpfd.synth import generate_dialogues as G


def _gaps(utts, total, min_len):
    """Free intervals [a,b] on the timeline with b-a >= min_len (for clean injection)."""
    occ = sorted((u.start, u.end) for u in utts)
    gaps, cur = [], 0.0
    for a, b in occ:
        if a - cur >= min_len:
            gaps.append((cur, a))
        cur = max(cur, b)
    if total - cur >= min_len:
        gaps.append((cur, total))
    return gaps


def _agent_phrase(rng, tier):
    if tier == "sem":
        return rng.choice(G.AGENT_Q_SEM)
    return rng.choice(G.AGENT_Q.get(tier, G.AGENT_Q["I2"]))


def _human_phrase(rng, tier, addressee):
    if tier == "sem":
        t = rng.choice(G.HUMAN_Q_SEM)
    else:
        t = rng.choice(G.HUMAN_Q.get(tier, G.HUMAN_Q["I2"]))
    return t.replace("{name}", addressee)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami", default="data/ami", help="dir of AMI sessions from convert_ami.py")
    ap.add_argument("--out", default="data/injected")
    ap.add_argument("--model_dir", required=True, help="dir of piper *.onnx voices for the injected utterances")
    ap.add_argument("--piper_bin", default="piper", help="piper CLI path if not on PATH (from `which piper`)")
    ap.add_argument("--per_session_agent", type=int, default=2)
    ap.add_argument("--per_session_human", type=int, default=2)
    ap.add_argument("--tier", default="mix", help="I0|I1|I2|sem|mix — implicitness of injected addressing")
    ap.add_argument("--dur", type=float, default=1.8, help="nominal injected-utterance slot length (s)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    from mpfd.synth.piper_backend import PiperBackend
    G.load_paraphrases()                          # inject from the same (Qwen-expanded) phrase banks
    tts = PiperBackend(args.model_dir, piper_bin=args.piper_bin)
    rng = random.Random(args.seed)
    tiers = ["I0", "I1", "I2", "sem"]

    files = sorted(glob.glob(os.path.join(args.ami, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"no AMI sessions in {args.ami} — run scripts/convert_ami.py first")
    n_out = 0
    for f in files:
        s = Session.from_dict(json.load(open(f, encoding="utf-8")))
        if not s.audio_path or not os.path.exists(s.audio_path):
            continue
        wav, sr = sf.read(s.audio_path)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(1)
        wav = wav.astype(np.float32)
        total = len(wav) / sr
        gaps = _gaps(s.utterances, total, args.dur + 0.4)
        rng.shuffle(gaps)
        speakers = s.speakers or ["A", "B"]

        plan = ["agent"] * args.per_session_agent + ["human"] * args.per_session_human
        rng.shuffle(plan)
        k = 0
        for kind in plan:
            if k >= len(gaps):
                break
            g0, g1 = gaps[k]; k += 1
            start = round(g0 + 0.15, 2)
            spk = rng.choice(speakers)                       # a real participant utters the injected line
            tier = rng.choice(tiers) if args.tier == "mix" else args.tier
            if kind == "agent":
                text = _agent_phrase(rng, tier)
                addressee, is_agent = s.agent_id, True
            else:
                others = [x for x in speakers if x != spk] or speakers
                addressee = rng.choice(others)
                text = _human_phrase(rng, tier, addressee)
                is_agent = False
            audio = tts.synthesize(text, f"inj_{kind}", sr).astype(np.float32)
            dur = min(len(audio) / sr, g1 - start)
            i0 = int(start * sr)
            seg = audio[: max(0, min(len(audio), len(wav) - i0))]
            wav[i0: i0 + len(seg)] += 0.9 * seg              # splice into the real mix
            s.utterances.append(Utterance(
                utt_id=f"{s.session_id}_inj{k}", speaker=spk, start=start,
                end=round(start + dur, 2), text=text, addressee=addressee,
                is_for_agent=is_agent, dialogue_act="question", implicitness=(tier if tier in tiers else None)))

        peak = float(np.max(np.abs(wav)) + 1e-8)
        wav = 0.97 * wav / peak
        s.utterances.sort(key=lambda u: u.start)
        out_wav = os.path.join(args.out, "injected", f"{s.session_id}.wav")
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        sf.write(out_wav, wav, sr)
        s.audio_path = out_wav
        s.cell = "injected"
        json.dump(json.loads(s.to_json()),
                  open(os.path.join(args.out, "injected", f"{s.session_id}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        n_out += 1

    print(f"wrote {n_out} injected sessions -> {args.out}/injected  "
          f"(agent+{args.per_session_agent}/human+{args.per_session_human} per session, tier={args.tier})")
    print("score the real model / controller on them like any rendered set:")
    print(f"  python scripts/04_run_real.py --sessions {args.out}/injected --pred freeze_omni --fo_repo ... --model_path ... --llm_path ... --onset_cache fo_inj")
    print(f"  python scripts/11_run_controller.py --sessions {args.out}/injected --onsets fo_cache --fo_cache fo_inj --addressee addressee.json")
    print("SEAM CHECK: wrong_addressee on injected HUMAN events must stay high (else the model keys on the TTS voice).")


if __name__ == "__main__":
    main()
