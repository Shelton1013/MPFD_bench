# -*- coding: utf-8 -*-
"""Run a REAL system on rendered audio and score. Systems read session.audio_path.

  --pred vad_dyadic   : runs now on rendered wavs (energy VAD + dyadic policy), no model
  --pred freeze_omni  : the real Freeze-Omni model (requires wiring FreezeOmniBaseline + --model_path)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.eval import load_sessions, run, print_report
from mpfd.baselines.real import REAL_SYSTEMS, FreezeOmniBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/rendered", help="dir of rendered sessions (with audio_path)")
    ap.add_argument("--pred", default="vad_dyadic")
    ap.add_argument("--fo_repo", default=None, help="freeze_omni: path to cloned Freeze-Omni repo")
    ap.add_argument("--model_path", default=None, help="freeze_omni: --model_path (audiollm checkpoints)")
    ap.add_argument("--llm_path", default=None, help="freeze_omni: frozen Qwen2-7B-Instruct path")
    ap.add_argument("--onset_cache", default=None, help="freeze_omni/minicpmo: dir to save raw speak "
                    "onsets per session (so the P2 controller can gate them without re-running)")
    ap.add_argument("--mc_model_path", default="openbmb/MiniCPM-o-4_5", help="minicpmo: model path/id")
    ap.add_argument("--ref_audio", default=None, help="minicpmo: short reference voice wav (TTS init)")
    ap.add_argument("--shard", default=None, help="k/N: process only session subset k-of-N (for "
                    "multi-GPU parallelism). Per-session onset caches from all shards merge in the "
                    "same --onset_cache dir; run 30_run_all on the full set afterwards.")
    args = ap.parse_args()

    if args.pred == "freeze_omni":
        if not (args.fo_repo and args.model_path and args.llm_path):
            raise SystemExit("freeze_omni needs --fo_repo --model_path --llm_path")
        system = FreezeOmniBaseline(args.fo_repo, args.model_path, args.llm_path,
                                    onset_cache=args.onset_cache)
    elif args.pred == "minicpmo":
        if not args.ref_audio:
            raise SystemExit("minicpmo needs --ref_audio <short reference voice wav>")
        from mpfd.baselines.minicpmo import MiniCPMoBaseline
        system = MiniCPMoBaseline(args.mc_model_path, args.ref_audio, onset_cache=args.onset_cache)
    elif args.pred in REAL_SYSTEMS:
        system = REAL_SYSTEMS[args.pred]
    else:
        raise SystemExit(f"unknown real system '{args.pred}'; available: {list(REAL_SYSTEMS)} + freeze_omni")

    sessions = load_sessions(args.sessions)
    if not sessions:
        raise SystemExit(f"0 sessions found under {os.path.abspath(args.sessions)} — check the path "
                         f"(injected sessions are written to <out>/injected/; use that exact dir)")
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        sessions = sessions[k::n]                     # deterministic partition (sorted glob order)
        print(f"shard {k}/{n}: processing {len(sessions)} of the sessions on this GPU")
    missing = [s.session_id for s in sessions if not s.audio_path]
    if missing:
        raise SystemExit(f"{len(missing)} sessions have no audio_path — run 02_render_audio.py first")
    print(f"loaded {len(sessions)} rendered sessions; running '{args.pred}'")
    print_report(run(sessions, system))


if __name__ == "__main__":
    main()
