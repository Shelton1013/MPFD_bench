# -*- coding: utf-8 -*-
"""Run the MPFD controller (base onsets + addressee gate) on sessions and score.

Examples:
  # gate gold-dyadic onsets by ORACLE addressee (upper bound of the approach, CPU):
  python scripts/11_run_controller.py --sessions data/synthetic --onsets gold_dyadic --addressee oracle
  # gate by a TRAINED classifier:
  python scripts/11_run_controller.py --sessions data/synthetic --onsets gold_dyadic --addressee addressee.json
  # gate Freeze-Omni's cached onsets by the trained classifier (the real headline system):
  python scripts/11_run_controller.py --sessions data/rendered --onsets fo_cache --fo_cache fo_onsets --addressee addressee.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.baselines.controller import MPFDController, ONSET_PROVIDERS, fo_cache_onsets
from mpfd.controller.addressee import AddresseeClassifier
from mpfd.eval import load_sessions, run, print_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--onsets", default="gold_dyadic", help="gold_dyadic | vad | fo_cache")
    ap.add_argument("--fo_cache", default=None, help="dir of Freeze-Omni cached onsets (for --onsets fo_cache)")
    ap.add_argument("--addressee", default="oracle", help="'oracle' or path to a trained addressee.json")
    ap.add_argument("--no_wakeword", action="store_true")
    args = ap.parse_args()

    if args.onsets == "fo_cache":
        if not args.fo_cache:
            raise SystemExit("--onsets fo_cache needs --fo_cache <dir>")
        provider = fo_cache_onsets(args.fo_cache)
    else:
        provider = ONSET_PROVIDERS[args.onsets]

    addressee = "oracle" if args.addressee == "oracle" else AddresseeClassifier.load(args.addressee)
    system = MPFDController(provider, addressee, use_wakeword=not args.no_wakeword)

    sessions = load_sessions(args.sessions)
    if not sessions:
        raise SystemExit(f"0 sessions found under {os.path.abspath(args.sessions)} — check the path")
    print(f"loaded {len(sessions)} sessions; controller onsets={args.onsets} addressee={args.addressee}")
    print_report(run(sessions, system))


if __name__ == "__main__":
    main()
