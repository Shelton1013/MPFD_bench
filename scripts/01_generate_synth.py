# -*- coding: utf-8 -*-
"""Generate scripted multi-party sessions (all cells) as JSON. CPU-only (no TTS)."""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.cells import CELLS
from mpfd.synth.generate_dialogues import make_session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n_per_cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = 0
    for cell in CELLS:
        d = os.path.join(args.out, cell)
        os.makedirs(d, exist_ok=True)
        for i in range(args.n_per_cell):
            s = make_session(cell, i, rng)
            open(os.path.join(d, f"{s.session_id}.json"), "w", encoding="utf-8").write(s.to_json())
            n += 1
    print(f"wrote {n} sessions across {len(CELLS)} cells to {args.out}")
    print("next: render audio with mpfd.synth.compose (server, TTS), or score directly:")
    print("  python scripts/03_score.py --sessions", args.out, "--pred naive_dyadic")


if __name__ == "__main__":
    main()
