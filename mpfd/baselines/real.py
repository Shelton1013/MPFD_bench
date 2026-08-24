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
    """Adapter around Freeze-Omni (https://github.com/VITA-MLLM/Freeze-Omni).

    Usage:  fo = FreezeOmniBaseline(model_path=...);  out = fo(session)
    Streams the rendered wav through the model and records chunks where its state classifier
    decides to speak. Fill the [VERIFY] calls against the repo's inference API.
    """
    def __init__(self, model_path: str, chunk_ms: float = 80.0, device: str = "cuda"):
        self.chunk_ms = chunk_ms
        self.device = device
        # [VERIFY] load Freeze-Omni exactly as in its inference script (encoder + frozen LLM + state head)
        # from freeze_omni.inference import FreezeOmniPipeline
        # self.pipe = FreezeOmniPipeline(model_path, device=device)
        self.model_path = model_path
        raise NotImplementedError(
            "Wire FreezeOmniBaseline to the repo: load the model, then in __call__ feed audio chunks "
            "and read the state classifier. See the docstring / docs/REAL_BASELINE.md.")

    def __call__(self, session: Session) -> SystemOutput:
        import soundfile as sf
        wav, sr = sf.read(session.audio_path)
        if wav.ndim > 1:
            wav = wav.mean(1)
        hop = int(sr * self.chunk_ms / 1000)
        speaking, seg, t0 = False, [], 0.0
        for i in range(0, len(wav), hop):
            chunk = wav[i:i + hop]
            t = i / sr
            # [VERIFY] state = self.pipe.step(chunk)  # -> one of {"listen","interrupt","respond"}
            # For the benchmark we only need WHEN the agent produces/decides speech:
            #   state in {"respond"} (or the model is currently generating audio) -> agent speaking
            state = "listen"  # placeholder
            is_speaking = state in ("respond",)
            if is_speaking and not speaking:
                speaking, t0 = True, t
            elif not is_speaking and speaking:
                speaking = False; seg.append((round(t0, 2), round(t, 2)))
        if speaking:
            seg.append((round(t0, 2), round(len(wav) / sr, 2)))
        addr = {u.utt_id: True for u in session.utterances if u.speaker != session.agent_id}
        return SystemOutput(session.session_id, agent_segments=seg, addressee_pred=addr)


REAL_SYSTEMS = {"vad_dyadic": vad_dyadic_baseline}   # FreezeOmniBaseline added after wiring
