# -*- coding: utf-8 -*-
"""Real-audio baselines. A *real system* reads `session.audio_path` (rendered wav) and returns a
SystemOutput. Two provided:

  • `vad_dyadic_baseline`  — RUNS NOW on rendered audio, no model: energy VAD detects speech turns
    (cannot attribute speakers), then applies the dyadic policy (respond after each turn; stop own
    turn on any new onset). Real perception + real timing -> a genuine dyadic-failure number.
  • `FreezeOmniBaseline`   — the real dyadic full-duplex model: streams the rendered audio through
    Freeze-Omni and reads its chunk-level state classifier ({keep-listen / interrupt / respond}) to
    recover when the agent would speak. Model-specific calls are marked [VERIFY] — wire to the repo.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..schema import Session, SystemOutput


# ---------------- energy VAD (numpy, no deps) ----------------
def energy_vad(wav: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0,
               thresh_pct: float = 60.0, min_speech: float = 0.2, min_gap: float = 0.2
               ) -> List[Tuple[float, float]]:
    """Return speech regions (start,end) in seconds via framewise RMS energy + hysteresis merge."""
    hop = int(sr * hop_ms / 1000); win = int(sr * frame_ms / 1000)
    if len(wav) < win:
        return []
    n = 1 + (len(wav) - win) // hop
    e = np.array([np.sqrt(np.mean(wav[i * hop: i * hop + win] ** 2) + 1e-9) for i in range(n)])
    thr = np.percentile(e, thresh_pct) * 0.6 + e.mean() * 0.4
    active = e > thr
    regions, i = [], 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            regions.append((i * hop / sr, (j * hop + win) / sr))
            i = j
        else:
            i += 1
    # merge close regions, drop short ones
    merged = []
    for s, en in regions:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], en)
        else:
            merged.append((s, en))
    return [(s, en) for s, en in merged if en - s >= min_speech]


def _dyadic_policy(turns: List[Tuple[float, float]], resp_dur: float = 1.4,
                   resp_gap: float = 0.3) -> List[Tuple[float, float]]:
    """Dyadic model behavior over detected speech turns (no speaker attribution):
    respond after each turn offset; if a new turn onset arrives while the agent is speaking, stop."""
    seg: List[Tuple[float, float]] = []
    for i, (_, off) in enumerate(turns):
        start = off + resp_gap
        end = start + resp_dur
        # stop if the next detected turn starts during the agent's response
        if i + 1 < len(turns) and turns[i + 1][0] < end:
            end = max(start, turns[i + 1][0] + 0.2)
        seg.append((round(start, 2), round(end, 2)))
    return seg


def vad_dyadic_baseline(session: Session) -> SystemOutput:
    import soundfile as sf
    if not session.audio_path:
        raise RuntimeError(f"{session.session_id}: no audio_path; render first (02_render_audio.py)")
    wav, sr = sf.read(session.audio_path)
    if wav.ndim > 1:
        wav = wav.mean(1)
    turns = energy_vad(wav.astype(np.float32), sr)
    seg = _dyadic_policy(turns)
    # dyadic model has no addressee reasoning -> predicts everything is for it
    addr = {u.utt_id: True for u in session.utterances if u.speaker != session.agent_id}
    return SystemOutput(session.session_id, agent_segments=seg, addressee_pred=addr)


# ---------------- real model: Freeze-Omni ----------------
class FreezeOmniBaseline:
    """Adapter around Freeze-Omni (https://github.com/VITA-MLLM/Freeze-Omni), source-verified.

    Feeds the rendered mixed wav to Freeze-Omni's OFFLINE pipeline (bin/inference.py) in 160 ms /
    2560-sample @16 kHz chunks and reads its chunk-level dialogue-state head each step. The head
    returns stat=='ss' when it decides to START SPEAKING (state_1>0.5) — this is the model's
    barge-in decision. We keep the model listening across the whole session (force stat back to
    'cl') to record EVERY moment it would speak, then turn those onsets into nominal speak spans.

    Run this in Freeze-Omni's OWN env (torch==2.2.0, transformers==4.45.2). Our mpfd package is
    pure-numpy so it imports fine there.

    Args:
      fo_repo   : path to the cloned Freeze-Omni repo
      model_path: --model_path (contains audiollm/{train.yaml,global_cmvn,final.pt})
      llm_path  : --llm_path (frozen Qwen2-7B-Instruct)
    """
    def __init__(self, fo_repo: str, model_path: str, llm_path: str,
                 role: str = "You are a helpful assistant.",
                 refractory_s: float = 1.5, resp_dur_s: float = 1.5,
                 top_p: float = 0.8, top_k: int = 20, temperature: float = 0.8):
        import sys, os, importlib.util, torch
        from types import SimpleNamespace
        # workaround for cuDNN "unable to find an engine to execute this computation" on the audio
        # encoder conv2d (version/bf16 mismatch). The audio encoder conv is tiny; Qwen has no conv.
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        sys.path.insert(0, fo_repo)  # so `from models.pipeline import ...` resolves
        spec = importlib.util.spec_from_file_location("fo_inference", os.path.join(fo_repo, "bin", "inference.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # defs only; run is under __main__
        self._proc_cls = mod.audioEncoderProcessor
        cfg = SimpleNamespace(model_path=model_path, llm_path=llm_path,
                              top_p=top_p, top_k=top_k, temperature=temperature)
        self.pipeline = mod.inferencePipeline(cfg)
        self.role, self.refractory, self.resp_dur = role, refractory_s, resp_dur_s

    def __call__(self, session: Session) -> SystemOutput:
        import math, torch, soundfile as sf
        wav, sr = sf.read(session.audio_path)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(1)
        wav = torch.tensor(wav).float()
        if sr != 16000:
            import torchaudio
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        proc = self._proc_cls()
        chunk = proc.get_chunk_size()                       # 2560 samples = 160 ms
        n = math.ceil(wav.shape[0] / chunk) * chunk
        buf = torch.zeros(n); buf[:wav.shape[0]] = wav

        outputs = self.pipeline.speech_dialogue(None, stat='pre', role=self.role)
        onsets = []
        for idx, i in enumerate(range(0, n, chunk)):
            fbank = proc.process(buf[i:i + chunk])
            outputs = self.pipeline.speech_dialogue(fbank, **outputs)
            if outputs.get('stat') == 'ss':                 # decided to START SPEAKING at this chunk
                onsets.append(idx * 0.16)
            outputs['stat'] = 'cl'                           # keep listening to probe the whole session

        # merge onsets within the refractory window; each -> a nominal speaking span
        seg, last = [], -1e9
        for t in onsets:
            if t - last >= self.refractory:
                seg.append((round(t, 2), round(t + self.resp_dur, 2)))
                last = t
        addr = {u.utt_id: True for u in session.utterances if u.speaker != session.agent_id}  # dyadic: all "for me"
        return SystemOutput(session.session_id, agent_segments=seg, addressee_pred=addr)


REAL_SYSTEMS = {"vad_dyadic": vad_dyadic_baseline}   # "freeze_omni" is constructed in 04_run_real
