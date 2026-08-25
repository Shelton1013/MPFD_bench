# -*- coding: utf-8 -*-
"""Addressee-F1 as a function of implicitness tier (I0 explicit -> I2 contextual).

This is the ② TO-WHOM headline diagnostic (see docs/BENCHMARK_SPEC.md §5.2). A model that only
matches wake-words scores ~1.0 at I0 and collapses at I2; a model that actually infers the
addressee from context degrades gracefully. Requires sessions whose addressing utterances carry
an `implicitness` tag (produced by scripts/01_generate_synth.py graded sessions).

Usage:
  python scripts/12_addressee_curve.py --sessions data/synthetic --addressee addressee.json
  python scripts/12_addressee_curve.py --sessions data/synthetic --addressee oracle          # upper bound
  python scripts/12_addressee_curve.py --sessions data/synthetic --addressee addressee_nw.json  # no-wakeword ablation
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.controller.addressee import AddresseeClassifier
from mpfd.controller.features import session_examples
from mpfd.eval import load_sessions


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0, p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--addressee", default="oracle", help="'oracle' or a trained addressee.json")
    ap.add_argument("--no_wakeword", action="store_true", help="feature ablation (must match training)")
    args = ap.parse_args()

    sessions = load_sessions(args.sessions)
    clf = None if args.addressee == "oracle" else AddresseeClassifier.load(args.addressee)
    uw = not args.no_wakeword

    # bucket per implicitness tier: confusion counts over TAGGED addressing utterances only
    conf = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for s in sessions:
        tag = {u.utt_id: u.implicitness for u in s.utterances if u.implicitness}  # per-session, no collision
        for x, y, uid in session_examples(s, use_wakeword=uw):
            tier = tag.get(uid)
            if tier is None:
                continue
            pred = int(y) if clf is None else int(clf.predict([x])[0])
            c = conf[tier]
            if y and pred:      c["tp"] += 1
            elif y and not pred: c["fn"] += 1
            elif (not y) and pred: c["fp"] += 1
            else:               c["tn"] += 1

    if not conf:
        raise SystemExit(
            f"\n[ERROR] 0 tagged addressing utterances found in {len(sessions)} sessions under "
            f"'{args.sessions}'.\nThis dir has no graded (I0-I2) sessions — it is the old 9-cell "
            f"data. Regenerate WITH graded, then rerun:\n"
            f"  python scripts/01_generate_synth.py --out {args.sessions} --n_per_cell 50 --n_graded_per_tier 120\n"
            f"  (session count should jump to ~810; addressee_graded/ subdir must exist)")

    print(f"\n=== addressee-F1 by implicitness tier  (addressee={args.addressee}, use_wakeword={uw}) ===")
    print(f"  {'tier':6s} {'F1':>6s} {'prec':>6s} {'rec':>6s} {'acc':>6s}  {'pos':>4s} {'neg':>4s}")
    for tier in sorted(conf):
        c = conf[tier]
        F, P, R = _f1(c["tp"], c["fp"], c["fn"])
        n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        acc = (c["tp"] + c["tn"]) / n if n else 0.0
        pos, neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
        print(f"  {tier:6s} {F:6.3f} {P:6.3f} {R:6.3f} {acc:6.3f}  {pos:4d} {neg:4d}")
    print("  (expect F1 high at I0 via lexical markers, dropping toward I2 where only context reveals the addressee)")


if __name__ == "__main__":
    main()
