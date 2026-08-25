# -*- coding: utf-8 -*-
"""Consolidate the whole benchmark into one table: every system x every track, the 2D headline
(over-speak FBR / wrong-addressee  vs  under-respond missed-response), and the addressee ablation.

Freeze-Omni is EXPENSIVE, so this script never re-runs it — it consumes the onset caches produced by
04_run_real.py (--onset_cache). From a cache it reconstructs, on CPU:
  • freeze_omni_raw          — the ungated dyadic model (keep every speak onset)
  • controller(oracle_addr)  — gate onsets by GOLD addressee     (upper bound of our method)
  • controller(trained_addr) — gate onsets by the trained classifier (the real headline system)
Plus the always-available symbolic references (oracle, naive_dyadic) and, where audio exists, vad_dyadic.

Tracks are passed as name,dir[,fo_cache] (comma-separated so paths with ':' are safe), e.g.:
  python scripts/30_run_all.py \
    --track synthetic,data/synthetic \
    --track ami,data/ami,fo_ami \
    --track injected,/abs/injected/injected,fo_inj \
    --addressee addressee.json --out results/table.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.baselines import naive_dyadic_system, oracle_system
from mpfd.baselines.controller import MPFDController, fo_cache_onsets
from mpfd.baselines.real import vad_dyadic_baseline
from mpfd.controller.addressee import AddresseeClassifier
from mpfd.controller.gate import gate_onsets
from mpfd.eval import load_sessions
from mpfd.metrics import aggregate
from mpfd.schema import SystemOutput

HEADLINE = ["false_bargein_rate", "missed_response_rate", "wrong_addressee_rate",
            "correct_response_rate", "addressee_f1", "continue_when_should_rate", "stop_success_rate"]
SHORT = {"false_bargein_rate": "FBR↓", "missed_response_rate": "miss↓", "wrong_addressee_rate": "wrongAddr↓",
         "correct_response_rate": "resp↑", "addressee_f1": "addrF1↑", "continue_when_should_rate": "cont↑",
         "stop_success_rate": "stop↑"}


def make_fo_raw(provider):
    def sysf(session):
        onsets = provider(session)
        allpos = {u.utt_id: True for u in session.utterances if u.speaker != session.agent_id}
        seg = gate_onsets(session, onsets, allpos)          # keep every onset = ungated dyadic behavior
        return SystemOutput(session.session_id, agent_segments=seg, addressee_pred=allpos)
    return sysf


def systems_for(track_sessions, fo_cache, addressee_clf):
    sysd = {"oracle": oracle_system, "naive_dyadic": naive_dyadic_system}
    if track_sessions and track_sessions[0].audio_path and os.path.exists(track_sessions[0].audio_path):
        sysd["vad_dyadic"] = vad_dyadic_baseline
    if fo_cache and os.path.isdir(fo_cache):
        provider = fo_cache_onsets(fo_cache)
        sysd["freeze_omni_raw"] = make_fo_raw(provider)
        sysd["controller(oracle_addr)"] = MPFDController(provider, "oracle")
        if addressee_clf is not None:
            sysd["controller(trained_addr)"] = MPFDController(provider, addressee_clf)
    return sysd


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else "  -  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", action="append", required=True,
                    help="name,dir[,fo_cache] (repeatable; comma-separated)")
    ap.add_argument("--addressee", default="addressee.json", help="trained addressee.json (optional)")
    ap.add_argument("--out", default="results/table.md")
    args = ap.parse_args()

    addressee_clf = AddresseeClassifier.load(args.addressee) if os.path.exists(args.addressee) else None
    if addressee_clf is None:
        print(f"[warn] {args.addressee} not found — skipping controller(trained_addr)")

    lines = ["# MPFD-Bench — consolidated results", "",
             "2D headline = (over-speak **FBR** / wrong-addressee) vs (under-respond **miss**). "
             "A true multi-party system is low on BOTH; a dyadic model is high-FBR; a mute model is high-miss.", ""]
    twod = []  # (track, system, fbr, miss)

    for spec in args.track:
        parts = spec.split(",")
        name, d = parts[0], parts[1]
        fo_cache = parts[2] if len(parts) > 2 else None
        sessions = load_sessions(d)
        if not sessions:
            lines += [f"## {name}", f"_0 sessions under {os.path.abspath(d)} — skipped_", ""]
            continue
        sysd = systems_for(sessions, fo_cache, addressee_clf)
        lines += [f"## {name}  ({len(sessions)} sessions{', fo_cache='+fo_cache if fo_cache else ''})", "",
                  "| system | " + " | ".join(SHORT[k] for k in HEADLINE) + " |",
                  "|" + "---|" * (len(HEADLINE) + 1)]
        for sname, sysf in sysd.items():
            try:
                m = aggregate(sessions, {s.session_id: sysf(s) for s in sessions})
            except Exception as e:
                lines.append(f"| {sname} | ERROR: {str(e)[:40]} |" + " |" * (len(HEADLINE) - 1))
                continue
            lines.append("| " + sname + " | " + " | ".join(fmt(m.get(k)) for k in HEADLINE) + " |")
            twod.append((name, sname, m.get("false_bargein_rate"), m.get("missed_response_rate")))
        lines.append("")

    # addressee ablation (oracle vs trained) + 2D coords
    lines += ["## Ablation — addressee: oracle vs trained (per track, from FBR/miss)", "",
              "| track | system | FBR↓ | miss↓ |", "|---|---|---|---|"]
    for name, sname, fbr, miss in twod:
        if "controller" in sname or sname in ("freeze_omni_raw", "oracle"):
            lines.append(f"| {name} | {sname} | {fmt(fbr)} | {fmt(miss)} |")
    lines += ["", "_WHO ablation (oracle-diar vs real streaming diarizer) pending: no diarization "
              "frontend wired yet._", ""]

    out = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(out)
    print(out)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
