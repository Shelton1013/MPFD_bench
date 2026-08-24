# -*- coding: utf-8 -*-
"""CosyVoice2 TTS backend for rendering scripted sessions to real audio (distinct voice/speaker).

Server-side (needs the CosyVoice2 repo + a pretrained checkpoint). Implements the `TTSBackend`
protocol from compose.py: `synthesize(text, voice, sr) -> mono float32 np.ndarray`.

Install (server):
  git clone https://github.com/FunAudioLLM/CosyVoice && cd CosyVoice
  pip install -r requirements.txt
  # download pretrained_models/CosyVoice2-0.5B  (see repo README / modelscope)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class CosyVoice2Backend:
    def __init__(self, model_dir: str = "pretrained_models/CosyVoice2-0.5B",
                 fp16: bool = False):
        from cosyvoice.cli.cosyvoice import CosyVoice2   # [VERIFY import path with your repo version]
        self.cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=fp16)
        self.model_sr = 24000  # CosyVoice2 outputs 24 kHz
        # available built-in SFT speakers (distinct voices); fall back to zero-shot if empty
        try:
            self.spks: List[str] = list(self.cosy.list_available_spks())
        except Exception:
            self.spks = []
        self._voice_map: Dict[str, str] = {}

    def _spk_for(self, voice: str) -> str:
        """Map a benchmark voice id (voice_0, voice_1, ...) to a distinct built-in speaker."""
        if voice not in self._voice_map:
            if not self.spks:
                raise RuntimeError("no SFT speakers available; use zero-shot with reference wavs")
            self._voice_map[voice] = self.spks[len(self._voice_map) % len(self.spks)]
        return self._voice_map[voice]

    def synthesize(self, text: str, voice: str, sr: int) -> np.ndarray:
        spk = self._spk_for(voice)
        chunks = []
        for out in self.cosy.inference_sft(text, spk, stream=False):   # [VERIFY signature]
            wav = out["tts_speech"].squeeze().cpu().numpy().astype(np.float32)
            chunks.append(wav)
        wav = np.concatenate(chunks) if chunks else np.zeros(int(0.3 * self.model_sr), np.float32)
        if sr != self.model_sr:
            wav = _resample(wav, self.model_sr, sr)
        return wav


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n = int(round(len(x) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x).astype(np.float32)
