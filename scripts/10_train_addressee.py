# -*- coding: utf-8 -*-
"""Train the addressee classifier on labeled sessions (utterance text+context -> is_for_agent).
CPU, seconds. Splits by session (no leakage). Reports accuracy + F1 (+ a no-wakeword ablation)."""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.controller.addressee import AddresseeClassifier
from mpfd.controller.features import session_examples
from mpfd.eval import load_sessions


def build(sessions, use_wakeword):
    X, y = [], []
    for s in sessions:
        for x, lab, _ in session_examples(s, use_wakeword=use_wakeword):
            X.append(x); y.append(lab)
    return X, y


def f1(pred, y):
    tp = sum(1 for p, t in zip(pred, y) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, y) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, y) if p == 0 and t == 1)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * pr * rc / (pr + rc)) if pr + rc else 0.0, pr, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--out", default="addressee.json")
    ap.add_argument("--no_wakeword", action="store_true", help="ablation: drop the wake-word feature")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sessions = load_sessions(args.sessions)
    rng = random.Random(args.seed); rng.shuffle(sessions)
    k = int(len(sessions) * (1 - args.val_frac))
    tr, va = sessions[:k], sessions[k:]
    uw = not args.no_wakeword

    Xtr, ytr = build(tr, uw); Xva, yva = build(va, uw)
    clf = AddresseeClassifier().fit(Xtr, ytr)
    for name, X, y in [("train", Xtr, ytr), ("val", Xva, yva)]:
        pred = clf.predict(X)
        acc = sum(int(p == t) for p, t in zip(pred, y)) / max(1, len(y))
        F, P, R = f1(pred, y)
        print(f"{name}: acc={acc:.3f} F1={F:.3f} P={P:.3f} R={R:.3f} n={len(y)} pos={sum(y)}")
    clf.save(args.out)
    print(f"saved -> {args.out}  (use_wakeword={uw})")


if __name__ == "__main__":
    main()
