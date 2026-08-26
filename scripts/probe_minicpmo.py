# -*- coding: utf-8 -*-
"""Minimal probe for the MiniCPM-o 4.5 full-duplex streaming API, to VERIFY the interface on your
box before we wire the full adapter. Loads the model in duplex mode, streams a wav in ~1s chunks,
and prints the speak/listen decision per chunk.

The gate-able signal we need is result["is_listen"]: False => the model decided to SPEAK at this
chunk (our onset), True => it stays listening. If the method names / return keys below differ on
your version, paste the error or the printed dict and I'll fix the adapter accordingly.

  python scripts/probe_minicpmo.py --model_path openbmb/MiniCPM-o-4_5 \
      --wav <any 16k wav> --ref_audio <a short ref voice wav> --n_chunks 20
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="openbmb/MiniCPM-o-4_5")
    ap.add_argument("--wav", required=True, help="input mixed wav to stream")
    ap.add_argument("--ref_audio", required=True, help="short reference voice wav (for TTS init)")
    ap.add_argument("--chunk_s", type=float, default=1.0)
    ap.add_argument("--n_chunks", type=int, default=20)
    args = ap.parse_args()

    import torch, soundfile as sf
    from transformers import AutoModel

    print(f"loading {args.model_path} ...")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    model = model.as_duplex()                          # [VERIFY] full-duplex mode
    print("duplex object type:", type(model).__name__)
    meth = sorted(m for m in dir(model) if not m.startswith("__")
                  and any(k in m.lower() for k in ("tts", "stream", "prepare", "duplex",
                          "generate", "prefill", "reset", "init", "omni", "listen", "speak")))
    print("relevant methods:", meth)
    for name in ("init_tts", "init_tts_module", "prepare_tts"):    # init TTS if such a method exists
        if hasattr(model, name):
            print(f"  calling model.{name}() ..."); getattr(model, name)(); break
    for name in ("prepare", "prepare_session", "reset_session"):    # session init if it exists
        if hasattr(model, name):
            try:
                getattr(model, name)(prompt_wav_path=args.ref_audio)
                print(f"  called model.{name}(prompt_wav_path=...)")
            except Exception as e:
                print(f"  model.{name} exists but call failed: {str(e)[:80]}")
            break

    wav, sr = sf.read(args.wav)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(wav).float()).numpy()
        sr = 16000
    step = int(args.chunk_s * sr)

    print(f"streaming {args.n_chunks} chunks of {args.chunk_s}s ...")
    for i in range(min(args.n_chunks, len(wav) // step)):
        chunk = wav[i * step:(i + 1) * step].astype(np.float32)
        model.streaming_prefill(audio_waveform=chunk, frame_list=[])   # [VERIFY] arg names/format
        result = model.streaming_generate(prompt_wav_path=args.ref_audio,
                                          max_new_speak_tokens_per_chunk=20, decode_mode="sampling")
        if i == 0:
            print("  [result keys]:", list(result.keys()))            # so we see the real schema
        is_listen = result.get("is_listen", None)
        txt = (result.get("text") or "").strip()[:40]
        print(f"  t={i*args.chunk_s:5.1f}s  is_listen={is_listen}  {'' if is_listen else 'SPEAK> '+txt}")

    print("\nOK if you see per-chunk is_listen (True/False). Paste this output and I'll finalize "
          "mpfd/baselines/minicpmo.py to match the exact API.")


if __name__ == "__main__":
    main()
