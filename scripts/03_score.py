# -*- coding: utf-8 -*-
"""Score a system on generated sessions. Demo systems: oracle, naive_dyadic (CPU, no models)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.baselines import SYSTEMS
from mpfd.eval import load_sessions, run, print_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/synthetic")
    ap.add_argument("--pred", default="naive_dyadic",
                    help="oracle | naive_dyadic (demo), or a real system registered elsewhere")
    args = ap.parse_args()

    if args.pred not in SYSTEMS:
        raise SystemExit(f"unknown system '{args.pred}'; available: {list(SYSTEMS)}")
    sessions = load_sessions(args.sessions)
    print(f"loaded {len(sessions)} sessions; running system '{args.pred}'")
    print_report(run(sessions, SYSTEMS[args.pred]))


if __name__ == "__main__":
    main()
