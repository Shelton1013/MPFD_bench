# -*- coding: utf-8 -*-
"""Run a REAL system on rendered audio and score. Systems read session.audio_path.

  --pred vad_dyadic   : runs now on rendered wavs (energy VAD + dyadic policy), no model
  --pred freeze_omni  : the real Freeze-Omni model (requires wiring FreezeOmniBaseline + --model_path)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.eval import load_sessions, run, print_report
from mpfd.baselines.real import REAL_SYSTEMS, FreezeOmniBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/rendered", help="dir of rendered sessions (with audio_path)")
    ap.add_argument("--pred", default="vad_dyadic")
    ap.add_argument("--model_path", default=None, help="for --pred freeze_omni")
    args = ap.parse_args()

    if args.pred == "freeze_omni":
        system = FreezeOmniBaseline(args.model_path)
    elif args.pred in REAL_SYSTEMS:
        system = REAL_SYSTEMS[args.pred]
    else:
        raise SystemExit(f"unknown real system '{args.pred}'; available: {list(REAL_SYSTEMS)} + freeze_omni")

    sessions = load_sessions(args.sessions)
    missing = [s.session_id for s in sessions if not s.audio_path]
    if missing:
        raise SystemExit(f"{len(missing)} sessions have no audio_path — run 02_render_audio.py first")
    print(f"loaded {len(sessions)} rendered sessions; running '{args.pred}'")
    print_report(run(sessions, system))


if __name__ == "__main__":
    main()
