# -*- coding: utf-8 -*-
"""Render a scripted Session to a multi-channel timeline and a mixed far-field wav.

Model-dependent (needs a TTS backend + optional room IR/noise) — kept behind an interface so the
benchmark logic stays CPU-only. On the server, implement `TTSBackend.synthesize` with CosyVoice2
(distinct voice per speaker) and call `render_session`.
"""
from __future__ import annotations

import os
from typing import Dict, Protocol

import numpy as np

from ..schema import Session


class TTSBackend(Protocol):
    def synthesize(self, text: str, voice: str, sr: int) -> "np.ndarray":
        """Return mono float32 waveform at sample rate sr for `text` in voice `voice`."""
        ...


def assign_voices(session: Session) -> Dict[str, str]:
    """Map each speaker (incl. agent + any non-participant) to a distinct voice id."""
    spk = list(dict.fromkeys([u.speaker for u in session.utterances]))
    return {s: f"voice_{i}" for i, s in enumerate(spk)}


def render_session(session: Session, tts: TTSBackend, out_wav: str,
                   noise: "np.ndarray | None" = None, snr_db: float = 20.0) -> str:
    """Place each utterance on a shared timeline at its (start,end), sum to one far-field mix.
    Overlaps sum naturally. Optionally add background noise at `snr_db`. Writes `out_wav`."""
    import soundfile as sf
    sr = session.sample_rate
    voices = assign_voices(session)
    total = max((u.end for u in session.utterances), default=1.0) + 0.5
    mix = np.zeros(int(total * sr) + 1, dtype=np.float32)
    for u in session.utterances:
        wav = tts.synthesize(u.text, voices[u.speaker], sr).astype(np.float32)
        i0 = int(u.start * sr)
        seg = wav[: max(0, len(mix) - i0)]
        mix[i0: i0 + len(seg)] += seg  # overlaps add
    if noise is not None and len(noise):
        n = np.resize(noise.astype(np.float32), len(mix))
        sp = float(np.mean(mix ** 2) + 1e-8); npow = float(np.mean(n ** 2) + 1e-8)
        scale = (sp / (npow * (10 ** (snr_db / 10)))) ** 0.5
        mix = mix + scale * n
    peak = float(np.max(np.abs(mix)) + 1e-8)
    mix = 0.97 * mix / peak
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    sf.write(out_wav, mix, sr)
    session.audio_path = out_wav
    return out_wav
