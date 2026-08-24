# -*- coding: utf-8 -*-
"""Evaluation harness: load sessions, run a system, print metrics."""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, List

from .metrics import aggregate
from .schema import Session, SystemOutput


def load_sessions(path: str) -> List[Session]:
    files = [path] if path.endswith(".json") else sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    return [Session.from_dict(json.load(open(f, encoding="utf-8"))) for f in files]


def run(sessions: List[Session], system: Callable[[Session], SystemOutput]) -> Dict:
    outs = {s.session_id: system(s) for s in sessions}
    return aggregate(sessions, outs)


def print_report(m: Dict):
    keys = ["false_bargein_rate", "correct_silence_rate", "correct_response_rate",
            "wrong_addressee_rate", "stop_success_rate", "continue_when_should_rate",
            "mean_response_latency_s", "mean_stop_latency_s", "addressee_f1"]
    print(f"\n=== MPFD-Bench ({m['n_sessions']} sessions) ===")
    for k in keys:
        v = m.get(k)
        print(f"  {k:32s}: {v:.3f}" if isinstance(v, float) else f"  {k:32s}: {v}")
    print("  --- per cell (fbr / resp / continue / n) ---")
    for cell, d in sorted(m.get("per_cell", {}).items()):
        fbr = f"{d['fbr']:.2f}" if d["fbr"] is not None else " -  "
        rr = f"{d['resp_rate']:.2f}" if d["resp_rate"] is not None else " -  "
        cr = f"{d['continue_rate']:.2f}" if d.get("continue_rate") is not None else " -  "
        print(f"    {cell:24s} fbr={fbr} resp={rr} cont={cr} n={d['n']}")
