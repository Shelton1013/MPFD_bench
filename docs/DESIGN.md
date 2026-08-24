# MPFD System Design (summary)

The MPFD program moves full-duplex spoken dialogue from the **dyadic** setting (1 user ↔ 1 agent,
two audio streams; assumes one speaker and that everything is addressed to the agent) to genuine
**multi-party** conversation. We do **not** train a full-duplex model from scratch; we wrap an
open 1v1 model in a trained controller.

## Architecture (speaker-routing + addressee-gating wrapper)
```
multi-speaker mixed audio
 → (A) perception front-end [reused]  : streaming diarization + TS-VAD + SA-ASR → speaker-tagged frames
 → (B) addressee/floor controller [trained, the core]:
        - floor/turn head   (REUSE SoulX-Duplug 2603.14877 / X2-Turn — dyadic, shipped)
        - addressee head     (OUR novelty: "is this for me?")  + N-party speaker arbitration
 → (C) 1v1 full-duplex base [frozen + LoRA]: route addressed speaker into the user slot;
        speak/stop only when addressed ∧ floor allows; else hold in Listen/Wait
 → assistant speech out
```

## Base model
- **Primary: Freeze-Omni** (2411.00774) — Apache-2.0, weights released, **frozen Qwen2-7B + an
  explicit chunk-level state classifier** = the ideal accessible hook to gate/extend; ZH+EN;
  8×A6000 LoRA-friendly.
- **Bench against: MiniCPM-o 4.5** (Apache-2.0, explicit 1 Hz proactive speak/stop controller).
- **Do NOT build on:** MinMo/SyncLLM/SALMONN-omni/OmniFlatten (weights not released),
  LLaMA-Omni2/VITA-1.5 (non-commercial).
- Principle: the **controller is base-agnostic** and is our durable asset; the base is a
  swappable component — do not retrain a base yourself.

## Why not Moshi (native L2 full-duplex)?
Moshi is open and SOTA on latency/genuine-duplex dynamics, but its speak/stop decision is
**emergent in the token stream (not externally gate-able)** and it is English-only — the wrong
shape for a wrapper method. We keep it as a native-duplex reference.

## Papers
- **P1 (this repo):** MPFD-Bench + baselines; show dyadic models fail.
- **P2:** train the addressee/floor controller (AMI addressee labels + synthetic).
- **P3:** controller + LoRA'd base = first participating multi-party full-duplex agent.
- **P4:** full system + meeting-copilot / interpretation application.

See `docs/METRICS.md` for the metric spec and `scripts/00_prepare_ami.md` for real-data notes.
