# -*- coding: utf-8 -*-
"""Trace the addressee gate's FBR<->miss tradeoff FRONTIER by sweeping the classifier threshold.

The single (FBR, miss) points from 11_run_controller are one operating point each. Sweeping the
addressee decision threshold traces the whole frontier a text-only gate can reach on real audio
(Track D). The oracle point sits at the ideal corner; the trained frontier stays far from it — the
definitive picture of the text-only addressee ceiling (SPEC §4.2).

  python scripts/15_addressee_frontier.py --sessions <injected> --fo_cache fo_inj \
      --addressee addressee_real.json --out results/frontier.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.baselines.controller import fo_cache_onsets
from mpfd.controller.addressee import AddresseeClassifier
from mpfd.controller.features import session_examples
from mpfd.controller.gate import gate_onsets
from mpfd.eval import load_sessions
from mpfd.metrics import aggregate
from mpfd.schema import SystemOutput


def score_at(sessions, provider, clf, thresh):
    outs = {}
    for s in sessions:
        proba = {uid: float(clf.predict_proba([x])[0]) for x, _y, uid in session_examples(s)}
        addr = {uid: (p >= thresh) for uid, p in proba.items()}
        seg = gate_onsets(s, provider(s), addr)
        outs[s.session_id] = SystemOutput(s.session_id, agent_segments=seg, addressee_pred=addr)
    m = aggregate(sessions, outs)
    return m["false_bargein_rate"], m["missed_response_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--fo_cache", required=True)
    ap.add_argument("--addressee", required=True, help="a trained addressee classifier (real or synthetic)")
    ap.add_argument("--out", default="results/frontier.md")
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    sessions = load_sessions(args.sessions)
    if not sessions:
        raise SystemExit(f"0 sessions under {os.path.abspath(args.sessions)}")
    provider = fo_cache_onsets(args.fo_cache)
    clf = AddresseeClassifier.load(args.addressee)
    ths = [float(t) for t in args.thresholds.split(",")]

    rows = [("thresh", "FBR", "miss")]
    for t in ths:
        fbr, miss = score_at(sessions, provider, clf, t)
        rows.append((f"{t:.2f}", f"{fbr:.3f}", f"{miss:.3f}"))
        print(f"thresh={t:.2f}  FBR={fbr:.3f}  miss={miss:.3f}")

    lines = ["# Addressee gate FBR<->miss frontier (Track D, real audio)", "",
             f"Classifier: `{os.path.basename(args.addressee)}`. Each row is one decision threshold; "
             "together they trace the frontier a text-only gate can reach. Compare to oracle "
             "(FBR 0.008, miss 0.022) — the frontier stays far from that ideal corner (SPEC §4.2).", "",
             "| " + " | ".join(rows[0]) + " |", "|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
