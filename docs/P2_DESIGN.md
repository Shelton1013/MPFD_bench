# P2 Design: Addressee / Floor Controller (the fix for P1's failures)

**Goal.** P1 showed a real dyadic full-duplex model (Freeze-Omni) collapses in multi-party:
FBR 0.725, wrong-addressee 0.555, addressee-F1 0.375. P2 builds a lightweight **controller** that
gates the dyadic model so it speaks only when it is actually the addressee — driving FBR and
wrong-addressee down while keeping (or improving) correct responses.

**Principle (base-agnostic wrapper, no base retraining).** The dyadic base already produces a
"should I speak now" signal (Freeze-Omni's chunk-level `stat=='ss'`). We do **not** retrain it.
We add a trained **addressee head** + **N-party arbitration** that *filters* those speak decisions:
suppress them when the current speech is inter-human / not addressed to the agent; keep them when
addressed. The turn/floor timing is reused (Freeze-Omni's own `ss`, or SoulX-Duplug/X2-Turn).

```
per utterance: {who (diarization), text (ASR), prosody, context}
        │
        ▼
   ADDRESSEE CLASSIFIER  →  P(is_for_agent)          ← the trained novelty (P2)
        │
        ▼  gate
  Freeze-Omni speak onsets ── keep onset only if the covering utterance is_for_agent
        │
        ▼
   gated agent speech  →  SystemOutput  →  MPFD-Bench metrics
```

## Components
1. **Perception** (reuse, not trained): diarization (who/when) + ASR (utterance text). Synthetic
   sessions have gold segments+text; real (AMI) uses pyannote + whisper.
2. **Addressee classifier** (`mpfd/controller/addressee.py`) — the trained module. Input per
   utterance: text features + prosody + **context** (previous speaker/addressee, speaker-change);
   output P(is_for_agent). v1 = logistic-regression/MLP over features (CPU, minutes); v2 = a
   fine-tuned small LM on text+context.
3. **Gate** (`mpfd/controller/gate.py`): keep a Freeze-Omni speak onset only if the utterance
   active at that time is predicted `is_for_agent`; also do not stop for non-addressing
   overlap / backchannel.
4. **Controller system** (`mpfd/baselines/controller.py`): wraps a source of base speak-onsets
   (cached Freeze-Omni decisions) + the addressee classifier → gated `SystemOutput` for the bench.

## Features for the addressee classifier (`controller/features.py`)
- **Lexical/content:** wake-word / assistant-name mention; is-question; imperative/2nd-person;
  utterance length; sentence embedding (optional, pluggable).
- **Context:** previous utterance's speaker and predicted addressee; did the speaker change;
  is the agent currently speaking; position in the exchange.
- **Prosody (optional, from audio):** pitch/energy trend, final-rise (question intonation).

## Training data
- **Synthetic** (`is_for_agent` gold from the scripts). ⚠️ Our current templates use an explicit
  wake-word ("assistant, …") → a wake-word detector would trivially ace it. So P2 must be
  evaluated on: (a) an **implicit-addressing** synthetic subset (no wake-word; addressing via
  content/context) — add to the generator; and (b) **AMI** with the Jovanović addressee layer
  (real, no wake-word). Report explicit vs implicit separately.
- **AMI** (real): Jovanović/op-den-Akker addressee annotations on an AMI/M4 subset → the real
  supervised signal; report addressee-F1 here as the honest number.

## Evaluation (on MPFD-Bench, vs P1 baselines)
Run the gated **controller system** and compare:
| System | FBR | wrong-addr | addressee-F1 | correct-resp |
|---|---|---|---|---|
| Freeze-Omni (ungated, P1) | 0.725 | 0.555 | 0.375 | 0.18 |
| **+ oracle-addressee gate** (upper bound) | ~0 (target) | ~0 | 1.0 | ↑ |
| **+ trained-addressee gate** (ours) | ↓ (target) | ↓ | ↑ | ↑ |

- **Oracle-gate** = gate by gold `is_for_agent`: shows the *ceiling* of the addressee-gating idea
  (how much of P1's failure is pure addressee error).
- **Trained-gate** = our classifier: the real contribution; how close to the ceiling.
- Ablations: wake-word-only vs learned; text-only vs +context vs +prosody; explicit vs implicit
  subset; synthetic-only vs +AMI.

## Metrics refinement (carry over from P1 caveat)
Refine onset→segment mapping and the response-window tolerance so `correct_response` is fair to
real models (count a response if the agent speaks within [utt_end − 0.3, utt_end + horizon]).
The over-speaking metrics (FBR / wrong-addressee) are unaffected and remain the headline.

## Deliverables
- `mpfd/controller/{features,addressee,gate}.py`, `mpfd/baselines/controller.py`
- `scripts/10_train_addressee.py` (build examples from sessions → train), `scripts/11_run_controller.py`
- FreezeOmniBaseline caches per-session speak onsets (so the controller reuses them without
  re-running the model).

## Honest risks
- **Wake-word triviality** on synthetic → must use implicit subset + AMI for the real claim.
- **Addressee from audio has a ceiling** (humans use gaze); expect imperfect F1, especially
  implicit/AMI — report it honestly, and the oracle-gate as the ceiling.
- **Diarization errors** propagate → report oracle-channel vs real-perception gap.
