# -*- coding: utf-8 -*-
"""Piper TTS backend — lightweight, offline, no torch/transformers (avoids CosyVoice dep hell).

Distinct voices = distinct Piper voice models (.onnx). Uses the piper CLI (version-robust).

Install (server):
  pip install piper-tts
  # download a few English voices (.onnx + .onnx.json) from
  #   https://huggingface.co/rhasspy/piper-voices
  # e.g. into ~/piper_voices/ :
  #   en_US-lessac-medium.onnx(.json), en_US-ryan-high.onnx(.json), en_GB-alba-medium.onnx(.json)
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from typing import Dict, List

import numpy as np


class PiperBackend:
    def __init__(self, voice_dir: str, piper_bin: str = "piper"):
        self.piper = piper_bin
        self.models: List[str] = sorted(glob.glob(os.path.join(voice_dir, "*.onnx")))
        if not self.models:
            raise RuntimeError(f"no *.onnx piper voices in {voice_dir} (download from rhasspy/piper-voices)")
        self._voice_map: Dict[str, str] = {}

    def _model_for(self, voice: str) -> str:
        if voice not in self._voice_map:
            self._voice_map[voice] = self.models[len(self._voice_map) % len(self.models)]
        return self._voice_map[voice]

    def synthesize(self, text: str, voice: str, sr: int) -> np.ndarray:
        import soundfile as sf
        model = self._model_for(voice)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out = tf.name
        try:
            subprocess.run([self.piper, "-m", model, "-f", out],
                           input=text.encode("utf-8"), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wav, msr = sf.read(out)
        finally:
            if os.path.exists(out):
                os.remove(out)
        if wav.ndim > 1:
            wav = wav.mean(1)
        wav = wav.astype(np.float32)
        if msr != sr:
            from math import gcd
            try:
                from scipy.signal import resample_poly
                g = gcd(int(msr), int(sr))
                wav = resample_poly(wav, sr // g, msr // g).astype(np.float32)
            except Exception:
                n = int(round(len(wav) * sr / msr))
                wav = np.interp(np.linspace(0, len(wav), n, endpoint=False),
                                np.arange(len(wav)), wav).astype(np.float32)
        return wav
