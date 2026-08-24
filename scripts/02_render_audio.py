# -*- coding: utf-8 -*-
"""Render scripted sessions to real mixed far-field audio (server; needs a TTS backend).

For each session JSON, synthesize every utterance with a distinct voice, place on the shared
timeline, mix, and write a wav; save an updated session JSON (with audio_path) to --out.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.schema import Session
from mpfd.synth.compose import render_session


def get_backend(name: str, model_dir: str = None):
    if name == "cosyvoice2":
        from mpfd.synth.cosyvoice_backend import CosyVoice2Backend
        return CosyVoice2Backend(model_dir) if model_dir else CosyVoice2Backend()
    raise SystemExit(f"unknown tts backend '{name}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/synthetic")
    ap.add_argument("--out", default="data/rendered")
    ap.add_argument("--tts", default="cosyvoice2", help="backend NAME (not a path): cosyvoice2")
    ap.add_argument("--model_dir", default=None,
                    help="path to the TTS model, e.g. /abs/path/CosyVoice2-0.5B "
                         "(default: pretrained_models/CosyVoice2-0.5B relative to CWD)")
    ap.add_argument("--noise_wav", default=None, help="optional background noise wav")
    ap.add_argument("--snr_db", type=float, default=20.0)
    args = ap.parse_args()

    tts = get_backend(args.tts, args.model_dir)
    noise = None
    if args.noise_wav:
        import soundfile as sf
        noise, _ = sf.read(args.noise_wav)

    files = sorted(glob.glob(os.path.join(args.sessions, "**", "*.json"), recursive=True))
    n = 0
    for f in files:
        s = Session.from_dict(json.load(open(f, encoding="utf-8")))
        wav_path = os.path.join(args.out, s.cell, f"{s.session_id}.wav")
        render_session(s, tts, wav_path, noise=noise, snr_db=args.snr_db)
        os.makedirs(os.path.dirname(os.path.join(args.out, s.cell, "x")), exist_ok=True)
        json.dump(json.loads(s.to_json()),
                  open(os.path.join(args.out, s.cell, f"{s.session_id}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        n += 1
        if n % 50 == 0:
            print(f"rendered {n}/{len(files)}")
    print(f"rendered {n} sessions -> {args.out}")


if __name__ == "__main__":
    main()
