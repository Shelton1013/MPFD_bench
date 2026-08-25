# -*- coding: utf-8 -*-
"""Generate scripted multi-party sessions (all cells) as JSON. CPU-only (no TTS)."""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.cells import CELLS
from mpfd.synth.generate_dialogues import (DEFAULT_PARAPHRASES, GRADED_TIERS, load_paraphrases,
                                           make_graded_session, make_session)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n_per_cell", type=int, default=50)
    ap.add_argument("--n_graded_per_tier", type=int, default=80,
                    help="graded addressee-curve sessions per implicitness tier (0 to skip)")
    ap.add_argument("--paraphrases", default=None,
                    help=f"LLM-expandable phrase bank JSON (default: auto-load {DEFAULT_PARAPHRASES} "
                         f"if present; pass '' to force the tiny built-in banks)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loaded = load_paraphrases(args.paraphrases)
    print(f"phrase bank: {'paraphrases.json' if loaded else 'built-in (tiny)'}")
    rng = random.Random(args.seed)
    n = 0
    for cell in CELLS:
        d = os.path.join(args.out, cell)
        os.makedirs(d, exist_ok=True)
        for i in range(args.n_per_cell):
            s = make_session(cell, i, rng)
            open(os.path.join(d, f"{s.session_id}.json"), "w", encoding="utf-8").write(s.to_json())
            n += 1
    # Track A graded addressee-difficulty sessions (I0->I2) for the addressee-F1 curve
    ng = 0
    if args.n_graded_per_tier > 0:
        d = os.path.join(args.out, "addressee_graded")
        os.makedirs(d, exist_ok=True)
        for tier in GRADED_TIERS:
            for i in range(args.n_graded_per_tier):
                s = make_graded_session(i, rng, tier=tier)
                open(os.path.join(d, f"{s.session_id}.json"), "w", encoding="utf-8").write(s.to_json())
                ng += 1
    print(f"wrote {n} sessions across {len(CELLS)} cells + {ng} graded sessions "
          f"({len(GRADED_TIERS)} tiers) to {args.out}")
    print("next: addressee-F1 curve (CPU):")
    print(f"  python scripts/10_train_addressee.py --sessions {args.out} --out addressee.json")
    print(f"  python scripts/12_addressee_curve.py --sessions {args.out} --addressee addressee.json")


if __name__ == "__main__":
    main()
