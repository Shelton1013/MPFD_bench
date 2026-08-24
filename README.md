# MPFD-Bench

**Multi-Party Full-Duplex Benchmark** — the first benchmark for whether a full-duplex spoken
dialogue agent behaves correctly in a **multi-party** conversation: attributing *who* is
speaking, inferring *whether it is being addressed*, and pausing / responding / staying silent
accordingly.

This is **Paper P1** of the MPFD research program. It (a) provides a reusable evaluation harness
and (b) quantifies how badly current **dyadic** (1-user) full-duplex models fail when a second
human is present (false barge-in, wrong-addressee responses).

> Current full-duplex models are dyadic: they assume one user and that everything is addressed
> to them. MPFD-Bench measures the two things that break when a second human enters —
> **speaker attribution** ("who is speaking") and **addressee** ("is this for me?").

---

## What's here (P1 scope)

- `mpfd/schema.py` — data structures (utterance, session, gold event, system prediction).
- `mpfd/cells.py` — the **9 test cells** (the interaction types that expose multi-party failure).
- `mpfd/metrics.py` — the metrics (fully runnable; no GPU): **false-barge-in rate**, addressee
  F1, wrong-addressee rate, correct-silence/response, attribution DER, latencies.
- `mpfd/synth/generate_dialogues.py` — template-based **N-party dialogue generator** producing
  scripted sessions for every cell (with gold addressee tags). Optional LLM hook for variety.
- `mpfd/synth/compose.py` — spec + interface to render scripts to a multi-channel timeline via
  a TTS backend (CosyVoice2 / any) and mix to a far-field stream.
- `mpfd/eval.py` — harness: run a **system** on the sessions → compute metrics.
- `mpfd/baselines/` — baseline interfaces: (i) dyadic frozen FD model on mixed input,
  (ii) modular L0 (diarization + heuristic addressee + gated FD model), (iii) oracle.
- `scripts/` — entry points + AMI/ICSI prep notes.

Model-dependent pieces (TTS, diarization, the full-duplex base model) are behind clean
interfaces so the benchmark logic runs CPU-only; plug the heavy components in on the server.

## Quick start (CPU, no models)

```bash
pip install -r requirements.txt
# 1) generate scripted multi-party sessions (all cells) as JSON
python scripts/01_generate_synth.py --out data/synthetic --n_per_cell 50
# 2) score a system's predictions against the gold (demo uses a dummy system)
python scripts/03_score.py --sessions data/synthetic --pred demo
```

## Roadmap
- **P1 (this repo):** benchmark + baselines; show dyadic models fail.
- **P2:** addressee/floor controller (trained on AMI addressee labels + synthetic).
- **P3:** controller + LoRA'd full-duplex base (Freeze-Omni / MiniCPM-o 4.5) = first
  participating multi-party full-duplex agent.
- **P4:** full system + meeting-copilot / interpretation application.

See `docs/DESIGN.md` for the full system design and `docs/METRICS.md` for the metric spec.
