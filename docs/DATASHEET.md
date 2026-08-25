# MPFD-Bench — Datasheet

Following *Datasheets for Datasets* (Gebru et al.). This documents what MPFD-Bench is, how it was
built, what it measures, and — explicitly — where it is limited. Companion to `docs/BENCHMARK_SPEC.md`
(the definitional constitution). Status: v1 (2026-08-26).

---

## 1. Motivation

**What gap does it fill?** Every open full-duplex spoken-dialogue model is *dyadic*: it assumes one
interlocutor and that everything it hears is addressed to it. No benchmark measures whether such a
model behaves correctly in a **multi-party** conversation — telling who speaks, whether it is being
addressed, and letting that govern when it speaks. MPFD-Bench is the first benchmark for that regime.

**Headline finding (motivation figure).** A real SOTA full-duplex model (Freeze-Omni) on real
multi-party acoustics barges in on **~89–92%** of the moments it should stay silent, and answers
**~93–96%** of questions directed at other people. It cannot tell it is not being addressed.

---

## 2. What it measures (capabilities → metrics)

Derived from dropping the two dyadic assumptions (see SPEC §1): **WHO** (speaker attribution),
**TO-WHOM** (addressee inference — the center of gravity), **WHEN** (N-party floor control), plus
**ACT** (integration) under a **REAL-TIME/causal** gate.

- **Primary, model-agnostic (behavioral): the 2D headline** — over-speak axis
  (`false_bargein_rate`, `wrong_addressee_rate`) vs under-respond axis (`missed_response_rate`).
  A dyadic model sits at high-FBR; a mute model at high-miss; only a true multi-party system is low
  on both. Plus `stop_success`, `continue_when_should`, latencies.
- **Diagnostic (white-box only): `addressee_f1`** (per addressed/other), `simplified_der` (WHO).
- Metrics are timing-robust: a correct response that (on real audio) overlaps ongoing human speech is
  **not** counted as a barge-in; a response to an addressed turn is never misattributed to a nearby
  human-directed question. Regression-tested so a correct system reads 0/0/0 on the headline axes.

---

## 3. Composition (tracks)

Real acoustics and clean labels cannot be maximized in one dataset, so MPFD-Bench is **tiered**; each
track states its scope.

| Track | Audio | Labels | Sessions | Role / scope |
|---|---|---|---|---|
| **A — synthetic** | TTS (or symbolic) | clean, by construction | 9 behavioral cells × N + graded I0–I2 × N | Diagnostic; the addressee **difficulty curve**. Scope: templated language, TTS acoustics. |
| **B — AMI real** | real meeting audio | human-only ⇒ agent must stay silent | AMI windows | Real-acoustics **FBR anchor**. Scope: over-speak half only; no response events; no real addressee labels. |
| **D — injected real** | real AMI + spliced TTS | clean for injected events | 115 (this release) | Real acoustics + clean labels for **both** halves via the injection protocol (§4.3). |

**Cells (Track A):** inter_human, overlapping_humans, addressed_turn, addressee_switch,
wrong_addressee, addressed_bargein, nonaddressing_overlap, backchannel, third_party — each exposes a
specific multi-party failure mode. **Graded addressing (Track A):** I0 explicit (wake-word / vocative
name), I1 vocative-dropped (position-cued), I2 contextual (content-only, position-misleading).
**Label schema (per utterance):** speaker, start, end, text, addressee, `is_for_agent`, dialogue_act,
`implicitness` tier. **Gold agent behavior** (RESPOND/SILENT/STOP/BACKCHANNEL/CONTINUE) is derived
from these (`mpfd/cells.py`).

---

## 4. Construction

### 4.1 Synthetic (Track A)
Deterministic templates (`mpfd/synth/generate_dialogues.py`) instantiate each cell with random timing
jitter; addressee labels are set at generation time (clean by construction). **Lexical variety** comes
from `data/paraphrases.json`, a per-speech-act phrase bank. The label is a property of the *category*,
so variants never change labels. The bank is expanded offline with a **local Qwen3 model**
(`scripts/13_gen_paraphrases.py`, vLLM or transformers) under a **per-category lexical validator** that
rejects any generation violating the category invariant (wake-word / vocative present-or-absent) —
labels are enforced, not trusted. Template-enumeration clusters are pruned by a prefix-diversity filter.

### 4.2 AMI conversion (Track B)
`scripts/convert_ami.py` turns `edinburghcstr/ami` (ihm) into Sessions + a timeline-mixed far-field wav
(each real segment placed at its begin_time; overlaps sum; decoded via soundfile, no torchcodec). Every
utterance is `is_for_agent=False` (human-only), so the whole track is a real-audio false-barge-in test.

### 4.3 Injection protocol (Track D) — the key "real" contribution
`scripts/20_inject_agent.py` splices TTS utterances (from the same phrase banks) into real AMI audio at
timeline gaps, with clean labels: **agent-addressed** injections (is_for_agent=True) → gold RESPOND;
**human-addressed** injections (is_for_agent=False) → gold SILENT. This yields real acoustics + clean
labels for both the over-speak and under-respond halves, from any human meeting corpus.

---

## 5. Baselines and headline results (this release)

**Track D (115 injected sessions, real Freeze-Omni onsets; 2D = FBR / miss):**

| system | FBR↓ | miss↓ | wrongAddr↓ | resp↑ | addrF1↑ |
|---|---|---|---|---|---|
| oracle (sanity) | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| freeze_omni_raw (real SOTA) | 0.915 | 0.000 | 0.960 | 1.000 | 0.022 |
| naive_dyadic | 0.841 | 0.000 | 0.994 | 1.000 | 0.022 |
| vad_dyadic | 0.675 | 0.022 | 0.876 | 0.978 | 0.022 |
| **controller (oracle addressee)** | **0.008** | **0.022** | 0.034 | 0.978 | 1.000 |
| controller (synthetic-trained addressee) | 0.007 | 0.667 | 0.023 | 0.333 | — |
| controller (real-trained addressee) | 0.427 | 0.111 | 0.475 | 0.889 | — |

**Track A (addressee-F1 vs implicitness, trained lexical+context LR, balanced):** I0 = 1.00, I1 = 1.00,
**I2 = 0.51 (≈ chance)**; oracle = 1.00 at every tier. **Track B (real AMI, human-only):** Freeze-Omni
FBR ≈ 0.896. **Track C (real AMI addressee, single-vs-group, meeting-split):** a synthetic-trained
classifier scores **F1 = 0.108** on real addressing (near-useless — the sim2real gap); retrained on
real AMI it reaches **F1 = 0.681** — better than chance but far from solved (the text-only ceiling,
consistent with GPT-4o being near-chance on AMI addressee, arXiv:2606.17542).

**The story the numbers tell.** (1) Real SOTA full-duplex collapses in real multi-party (FBR 0.9). (2)
The addressee-gating method is sound *in principle*: with a **perfect** addressee it reaches the ideal
corner (FBR 0.008, miss 0.022). (3) But **text-only addressee has a hard ceiling.** A synthetic-trained
classifier is near-useless on real speech (Track C F1 0.108) → the controller is over-conservative
(FBR 0.007 / miss 0.667). Retraining on real AMI addressee (F1 0.681) does not reach the oracle corner;
it slides *along a tradeoff frontier* (FBR 0.427 / miss 0.111) — trading over-speaking for
under-responding — because "directed at the agent" and "directed at another person" are linguistically
near-identical. (4) The addressee task is not a keyword regex: on the graded synthetic curve a strong
lexical+context baseline drops to chance at I2.

**Core finding / open problem defined by MPFD-Bench:** multi-party addressee-gating works given the
addressee, but resolving the addressee from audio/text alone is bounded well away from the ideal
corner. Closing it needs information beyond text — role grounding (what the agent can do), explicit
address, or multimodality. The 2D frontier (synthetic-gate → real-gate → oracle-gate) quantifies
exactly how far current addressee inference is from sufficient.

---

## 6. Limitations (read before using)

- **Track A language is synthetic/templated** and, when rendered, uses TTS acoustics — not natural
  speech. It is the *diagnostic*, not the ecological-validity, track.
- **Track D injections are TTS**, with an acoustic seam vs. real speakers. We built in a **seam check**:
  on this release the raw model answers injected *human*-addressed TTS at 0.93 (as high as agent-addressed)
  — i.e. it is **not** keying on "clean voice = respond", so the clean labels are valid. Still, the seam
  is a known caveat; voice-conversion is future work.
- **The addressee classifier does not transfer** from synthetic to real text (F1 0.5 on AMI). Closing
  this needs real addressee training data (Track C / Jovanović AMI annotations) — this is the headline
  next step, not a hidden failure.
- **No WHO ablation yet**: oracle-diarization vs a real streaming diarizer is unmeasured (no diarization
  frontend wired). Reported as pending.
- **AMI has no real addressee labels** in the standard release; Track B tests only the over-speak half.
- **`correct_response=1.0` for the raw model is not a virtue** — it is a byproduct of over-speaking; read
  it only inside the 2D (it sits at the high-FBR corner).
- Small cross-method differences (e.g. freeze_omni_raw FBR 0.915 via cached-onset reconstruction vs 0.893
  via the live streaming merge) reflect span-reconstruction, not different behavior.

---

## 7. Intended uses / out of scope

**In scope:** evaluating whether a full-duplex speech model behaves correctly under multi-party
addressing (audio-only); comparing controllers/gates; measuring addressee inference difficulty. **Out of
scope:** ASR/WER quality; vision/gaze-based addressing; a standalone diarization benchmark (WHO is a
prerequisite, not the contribution).

---

## 8. Reproduction

```
# Track A (CPU): generate (with the Qwen-expanded bank), score, addressee curve
python scripts/01_generate_synth.py --out data/synthetic --n_per_cell 50 --n_graded_per_tier 150
python scripts/03_score.py  --sessions data/synthetic --pred oracle        # sanity: 0/0/0 headline
python scripts/10_train_addressee.py --sessions data/synthetic --out addressee.json
python scripts/12_addressee_curve.py --sessions data/synthetic --addressee addressee.json
# Track B/D (GPU + piper): real AMI + injection + real model
python scripts/convert_ami.py --config ihm --split test --out data/ami --max_meetings 3
python scripts/20_inject_agent.py --ami data/ami --out data/injected --model_dir ~/piper_voices
python scripts/04_run_real.py --sessions <injected> --pred freeze_omni --fo_repo ... --model_path ... --llm_path ... --onset_cache fo_inj
python scripts/11_run_controller.py --sessions <injected> --onsets fo_cache --fo_cache fo_inj --addressee oracle
# Consolidated table
python scripts/30_run_all.py --track injected,<injected>,fo_inj --addressee addressee.json --out results/table.md
```

Phrase-bank expansion (local Qwen3 via vLLM): serve the model, then
`python scripts/13_gen_paraphrases.py --backend vllm --model qwen --base_url http://localhost:8002/v1 --fresh`.
