# -*- coding: utf-8 -*-
"""Second base model: MiniCPM-o 4.5 (openbmb/MiniCPM-o-4_5), a full-duplex omni model with a 1 Hz
proactive speak/listen decision. Same adapter contract as FreezeOmniBaseline: stream the rendered
wav, read the model's per-chunk decision, record the onsets at which it would SPEAK, and turn them
into nominal speak spans. The gate-able signal is result["is_listen"] (False => speak).

API markers tagged [VERIFY] mirror scripts/probe_minicpmo.py; confirm with the probe, then this runs
unchanged. Loads via transformers trust_remote_code (no separate repo needed, unlike Freeze-Omni).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..schema import Session, SystemOutput


class MiniCPMoBaseline:
    def __init__(self, model_path: str, ref_audio: str, chunk_s: float = 1.0,
                 refractory_s: float = 1.5, resp_dur_s: float = 1.5, onset_cache: str = None):
        self.ref_audio, self.chunk_s = ref_audio, chunk_s
        self.refractory, self.resp_dur, self.onset_cache = refractory_s, resp_dur_s, onset_cache
        import torch
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                          attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        self.model = model.eval().cuda()
        self.model = self.model.as_duplex()            # [VERIFY]
        self.model.init_tts()                          # [VERIFY]
        try:
            self.model.prepare(prompt_wav_path=ref_audio)   # [VERIFY] session init
        except Exception:
            pass

    def _reset(self):
        # [VERIFY] per-session reset so sessions don't leak state; name may differ (reset/clear/prepare)
        for m in ("reset", "reset_session", "clear_history"):
            if hasattr(self.model, m):
                getattr(self.model, m)()
                return
        try:
            self.model.prepare(prompt_wav_path=self.ref_audio)
        except Exception:
            pass

    def __call__(self, session: Session) -> SystemOutput:
        import soundfile as sf, torch
        wav, sr = sf.read(session.audio_path)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(wav).float()).numpy()
            sr = 16000
        wav = wav.astype(np.float32)
        step = int(self.chunk_s * sr)
        n = math.ceil(len(wav) / step)

        self._reset()
        onsets = []
        for idx in range(n):
            chunk = wav[idx * step:(idx + 1) * step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            self.model.streaming_prefill(audio_waveform=chunk, frame_list=[])          # [VERIFY]
            result = self.model.streaming_generate(prompt_wav_path=self.ref_audio,
                                                   max_new_speak_tokens_per_chunk=20,
                                                   decode_mode="sampling")             # [VERIFY]
            if not result.get("is_listen", True):      # False => decided to speak
                onsets.append(round(idx * self.chunk_s, 2))

        if self.onset_cache:
            os.makedirs(self.onset_cache, exist_ok=True)
            json.dump({"onsets": onsets}, open(os.path.join(self.onset_cache, f"{session.session_id}.json"), "w"))
        seg, last = [], -1e9
        for t in onsets:
            if t - last >= self.refractory:
                seg.append((round(t, 2), round(t + self.resp_dur, 2)))
                last = t
        addr = {u.utt_id: True for u in session.utterances if u.speaker != session.agent_id}
        return SystemOutput(session.session_id, agent_segments=seg, addressee_pred=addr)
