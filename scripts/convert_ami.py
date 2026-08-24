# -*- coding: utf-8 -*-
"""Convert AMI (edinburghcstr/ami HF dataset) into MPFD Sessions + real mixed multi-party audio.

Each row is one utterance segment: {meeting_id, speaker_id, begin_time, end_time, text, audio}.
We group by meeting, split each meeting into fixed windows, and for each window build a Session
(utterances rebased to the window) + a mixed far-field wav (each segment's real audio placed at
its begin_time; overlaps sum). AMI is human-only -> every utterance is_for_agent=False, so the
whole thing is the "agent must stay silent" test -> a real-audio FALSE-BARGE-IN benchmark for a
dyadic full-duplex model (no addressee labels needed).

Audio is decoded from bytes with soundfile (no torchcodec dependency).
"""
import argparse
import io
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.schema import Session, Utterance

_Q = re.compile(r"\b(what|when|where|who|why|how|which|can|could|would|is|are|do|does|did)\b", re.I)


def _decode(b) -> "tuple[np.ndarray, int]":
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(b))
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(1)
    return wav.astype(np.float32), sr


def _resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    from math import gcd
    try:
        from scipy.signal import resample_poly
        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n = int(round(len(x) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ihm", help="ihm (per-speaker headset) or sdm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/ami")
    ap.add_argument("--max_meetings", type=int, default=3)
    ap.add_argument("--window", type=float, default=60.0, help="seconds per Session window")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--min_utts", type=int, default=3)
    args = ap.parse_args()

    from datasets import load_dataset, Audio
    ds = load_dataset("edinburghcstr/ami", args.config, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))

    # collect rows until we have max_meetings distinct meetings
    meetings = defaultdict(list)
    for r in ds:
        m = r["meeting_id"]
        if m not in meetings and len(meetings) >= args.max_meetings:
            continue
        meetings[m].append(r)
    print(f"collected {len(meetings)} meetings: {list(meetings)}")

    n_sess = 0
    for m, rows in meetings.items():
        rows.sort(key=lambda r: r["begin_time"])
        total = max(r["end_time"] for r in rows)
        nwin = int(total // args.window) + 1
        for wi in range(nwin):
            w0, w1 = wi * args.window, (wi + 1) * args.window
            win_rows = [r for r in rows if r["begin_time"] < w1 and r["end_time"] > w0]
            spk = {r["speaker_id"] for r in win_rows}
            if len(win_rows) < args.min_utts or len(spk) < 2:
                continue
            sid = f"{m}_w{wi:03d}"
            # build mixed audio + utterances
            buf = np.zeros(int(args.window * args.sr) + 1, np.float32)
            utts = []
            for j, r in enumerate(win_rows):
                wav, sr = _decode(r["audio"]["bytes"])
                if sr != args.sr:
                    wav = _resample(wav, sr, args.sr)
                st = max(0.0, r["begin_time"] - w0)
                i0 = int(st * args.sr)
                seg = wav[: max(0, len(buf) - i0)]
                buf[i0: i0 + len(seg)] += seg
                txt = (r["text"] or "").strip()
                utts.append(Utterance(utt_id=f"{sid}_u{j}", speaker=r["speaker_id"],
                                      start=round(st, 2), end=round(min(args.window, r["end_time"] - w0), 2),
                                      text=txt, addressee=None, is_for_agent=False,
                                      dialogue_act="question" if (txt.endswith("?") or _Q.match(txt)) else "statement"))
            peak = float(np.max(np.abs(buf)) + 1e-8)
            buf = 0.97 * buf / peak
            import soundfile as sf
            wav_path = os.path.join(args.out, "ami", f"{sid}.wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            sf.write(wav_path, buf, args.sr)
            s = Session(session_id=sid, cell="ami", speakers=sorted(spk), agent_id="Agent",
                        utterances=utts, sample_rate=args.sr, audio_path=wav_path)
            json.dump(json.loads(s.to_json()), open(os.path.join(args.out, "ami", f"{sid}.json"), "w",
                                                    encoding="utf-8"), ensure_ascii=False, indent=2)
            n_sess += 1
    print(f"wrote {n_sess} AMI sessions -> {args.out}/ami  (window={args.window}s, human-only, all is_for_agent=False)")
    print("Next: feed to Freeze-Omni for the REAL-audio false-barge-in number:")
    print(f"  python scripts/04_run_real.py --sessions {args.out}/ami --pred freeze_omni --fo_repo ... --model_path ... --llm_path ...")


if __name__ == "__main__":
    main()
