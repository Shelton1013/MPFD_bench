# -*- coding: utf-8 -*-
"""95% bootstrap confidence intervals for the headline metrics, resampling over sessions.

Adds the statistical rigor reviewers expect: every reported FBR / miss / wrong-addressee / resp gets
a CI. Consumes the same Freeze-Omni onset cache as 30_run_all (never re-runs the model).

  python scripts/31_bootstrap_ci.py --sessions <injected> --fo_cache fo_inj \
      --addressee addressee_real.json --n_boot 1000 --out results/ci.md
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.baselines import naive_dyadic_system, oracle_system
from mpfd.baselines.controller import MPFDController, fo_cache_onsets
from mpfd.baselines.real import vad_dyadic_baseline
from mpfd.controller.addressee import AddresseeClassifier
from mpfd.controller.gate import gate_onsets
from mpfd.eval import load_sessions
from mpfd.metrics import aggregate
from mpfd.schema import SystemOutput

METRICS = ["false_bargein_rate", "missed_response_rate", "wrong_addressee_rate", "correct_response_rate"]
SHORT = {"false_bargein_rate": "FBR", "missed_response_rate": "miss",
         "wrong_addressee_rate": "wrongAddr", "correct_response_rate": "resp"}


def make_fo_raw(provider):
    def sysf(session):
        onsets = provider(session)
        allpos = {u.utt_id: True for u in session.utterances if u.speaker != session.agent_id}
        return SystemOutput(session.session_id, agent_segments=gate_onsets(session, onsets, allpos),
                            addressee_pred=allpos)
    return sysf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--fo_cache", default=None)
    ap.add_argument("--addressee", default=None, help="trained classifier for controller(trained_addr)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ci.md")
    args = ap.parse_args()

    sessions = load_sessions(args.sessions)
    if not sessions:
        raise SystemExit(f"0 sessions under {os.path.abspath(args.sessions)}")

    systems = {"oracle": oracle_system, "naive_dyadic": naive_dyadic_system}
    if sessions[0].audio_path and os.path.exists(sessions[0].audio_path):
        systems["vad_dyadic"] = vad_dyadic_baseline
    if args.fo_cache and os.path.isdir(args.fo_cache):
        prov = fo_cache_onsets(args.fo_cache)
        systems["freeze_omni_raw"] = make_fo_raw(prov)
        systems["controller(oracle_addr)"] = MPFDController(prov, "oracle")
        if args.addressee and os.path.exists(args.addressee):
            systems["controller(trained_addr)"] = MPFDController(prov, AddresseeClassifier.load(args.addressee))

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(sessions))
    lines = [f"# 95% bootstrap CIs (resampling sessions, n_boot={args.n_boot})", "",
             "| system | " + " | ".join(SHORT[m] for m in METRICS) + " |",
             "|" + "---|" * (len(METRICS) + 1)]
    for name, sysf in systems.items():
        outs = {s.session_id: sysf(s) for s in sessions}          # run once
        boot = {m: [] for m in METRICS}
        for _ in range(args.n_boot):
            samp = [sessions[i] for i in rng.choice(idx, size=len(idx), replace=True)]
            mm = aggregate(samp, outs)
            for m in METRICS:
                v = mm.get(m)
                if v is not None:
                    boot[m].append(v)
        cells = []
        for m in METRICS:
            if boot[m]:
                a = np.array(boot[m]); lo, hi = np.percentile(a, [2.5, 97.5])
                cells.append(f"{a.mean():.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                cells.append("—")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
        print(f"{name:28s} " + "  ".join(f"{SHORT[m]}={c}" for m, c in zip(METRICS, cells)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
