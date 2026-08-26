# -*- coding: utf-8 -*-
"""Second base model: MiniCPM-o 4.5 (openbmb/MiniCPM-o-4_5), a full-duplex omni model with a 1 Hz
proactive speak/listen decision. Same adapter contract as FreezeOmniBaseline: stream the rendered
wav, read the model's per-chunk decision, record the onsets at which it starts to SPEAK, and turn
them into nominal speak spans (refractory + resp_dur, identical to FreezeOmniBaseline so the two
base models are directly comparable in 30_run_all / the frontier).

Verified API (openbmb/MiniCPM-o-4_5, transformers dynamic module):
  model = AutoModel.from_pretrained(..., trust_remote_code=True).eval().cuda().as_duplex()
  model.prepare(prefix_system_prompt=..., ref_audio=<np.ndarray 16k mono>)   # per session; no torchcodec
  model.streaming_prefill(audio_waveform=<1s chunk>, frame_list=[])
  result = model.streaming_generate(max_new_speak_tokens_per_chunk=20, decode_mode="sampling")
  result["is_listen"]  -> True = listening, False = speaking (our speak signal)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..schema import Session, SystemOutput


def _load_16k_mono(path):
    import soundfile as sf
    wav, sr = sf.read(path)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(1)
    wav = wav.astype(np.float32)
    if sr != 16000:
        import torch, torchaudio
        wav = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(wav).float()).numpy()
    return wav


class MiniCPMoBaseline:
    def __init__(self, model_path: str, ref_audio: str, chunk_s: float = 1.0,
                 refractory_s: float = 1.5, resp_dur_s: float = 1.5, onset_cache: str = None,
                 system_prompt: str = "You are a helpful assistant."):
        self.chunk_s, self.refractory, self.resp_dur = chunk_s, refractory_s, resp_dur_s
        self.onset_cache, self.system_prompt = onset_cache, system_prompt
        self.ref = _load_16k_mono(ref_audio)           # pass as ndarray -> bypasses torchcodec
        import torch
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                          attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        self.model = model.eval().cuda().as_duplex()

    def __call__(self, session: Session) -> SystemOutput:
        wav = _load_16k_mono(session.audio_path)
        sr = 16000
        step = int(self.chunk_s * sr)
        n = math.ceil(len(wav) / step)

        self.model.prepare(prefix_system_prompt=self.system_prompt, ref_audio=self.ref)  # reset+init per session
        onsets, prev_listen = [], True
        for idx in range(n):
            chunk = wav[idx * step:(idx + 1) * step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            self.model.streaming_prefill(audio_waveform=chunk, frame_list=[])
            result = self.model.streaming_generate(max_new_speak_tokens_per_chunk=20,
                                                   decode_mode="sampling")
            is_listen = bool(result.get("is_listen", True))
            if (not is_listen) and prev_listen:        # True->False = it STARTS speaking here
                onsets.append(round(idx * self.chunk_s, 2))
            prev_listen = is_listen

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
