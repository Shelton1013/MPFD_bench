# MPFD-Bench Specification (the constitution)

> This document defines **what MPFD-Bench measures and why**. Every data-generation, metric, and
> code change must conform to it. It is the reference all other docs (DESIGN, METRICS, REAL_BASELINE,
> P2_DESIGN) defer to. Status: v1 (2026-08-25).

---

## 0. One-sentence purpose

MPFD-Bench is the first benchmark that measures whether a **real-time speech dialogue model** behaves
correctly in a **multi-party** conversation — i.e. whether it can tell who is speaking, tell whether it
is being addressed, and let that judgment govern when it speaks — rather than assuming (as every
dyadic full-duplex model does) that there is one interlocutor and everything is addressed to it.

---

## 1. What defines "multi-party full-duplex" (the capabilities under test)

We do **not** invent a capability list; we derive it. Every dyadic full-duplex model bakes in two
assumptions. Multi-party is exactly the regime where both break:

- **A1 — there is exactly one other party** ⇒ "who spoke" is trivial (always the one user).
- **A2 — everything the other party says is addressed to me** ⇒ "is it for me" is trivial (always yes).

Dropping A1 and A2 yields, with nothing added or missing, **three core capabilities + one integration
+ one cross-cutting constraint**:

| # | Capability | Breaks | What it means |
|---|------------|--------|---------------|
| ① | **WHO** — online speaker attribution | A1 | Keep the N human voices apart, online, and track stable identities across the conversation. |
| ② | **TO-WHOM** — addressee inference | A2 | Given a human utterance, infer its **target**: me / a specific other human / the group. *The defining capability.* Distinct from WHO (source ≠ target). |
| ③ | **WHEN** — N-party floor / turn control | A1∧A2 | Stay silent while humans talk **to each other**; don't treat every gap as an invitation; hold the floor when a non-addressing human overlaps; yield when the addressed human interrupts. |
| ④ | **ACT** — behavior conditions on ①②③ | — | The right action (respond / silent / continue / stop / backchannel) actually fires as a function of who + to-whom + floor state. Not a new metric — the *joint* of ①②③. |
| ⑤ | **REAL-TIME / causal** | — | All of the above hold **online, low-latency, no lookahead**. This is the gate that separates a full-duplex model from an offline meeting analyzer. |

### 1.1 The center of gravity

The capabilities form a dependency chain:

```
WHO (perception; already ~solved by streaming diarization; a PREREQUISITE, not our novelty)
  └─▶ TO-WHOM (inference; the real gap; the SUPPORT of the whole benchmark)
        └─▶ WHEN / ACT (decision; the OBSERVABLE behavior)      ⟵ REAL-TIME gate over everything
```

**The benchmark's headline capability is _addressee-conditioned floor control_**: does the model do the
right thing (speak / stay silent) as a function of who is being addressed. We cannot read a model's
mind, so we measure TO-WHOM through its **behavior** (WHEN/ACT). WHO is a measured *prerequisite*;
REAL-TIME is a hard *gate*.

### 1.2 One-line litmus test

> A speech model counts as multi-party full-duplex **iff**, on a live N-party stream, it can
> (1) tell the speakers apart, (2) tell when it is vs. isn't being addressed, (3) let that distinction
> govern whether it speaks — staying silent through human-to-human exchange and entering only when
> addressed — (4) all online with no lookahead. **Failing any one ⇒ still dyadic (or an offline analyzer).**

### 1.3 The atomic scenario

Two humans converse in front of the agent; then one turns and addresses the agent. This single scenario
exercises WHO+TO-WHOM+WHEN simultaneously, and a dyadic model **provably fails it** (it barges in during
the human-human exchange). The whole benchmark is variations of this atom. This is why false-barge-in on
real human-human audio (AMI: 0.896) is our sharpest single-capability probe.

### 1.4 Scoping decision: audio-only addressee

Human addressee cues include gaze/gesture, which are unavailable to audio-only base models (Freeze-Omni,
Moshi, etc.). **We scope TO-WHOM to _audio-inferable_ addressing** (vocative/name, second-person,
question directedness, topic/QA adjacency, prosody). Cases where the audio genuinely underdetermines the
addressee are labeled a distinct **ambiguous** class, where the *correct* behavior may be to not confidently
respond (hedge / ask) rather than barge in. We do **not** introduce vision. Rationale: matches the base
models; keeps the benchmark model-agnostic; the ambiguity itself is a research-interesting sub-question.

---

## 2. Positioning against existing taxonomies

- **Survey L0–L3 axis** (arXiv:2606.19453, `[待核实 arxiv id]`) — *where in the architecture the duplex
  decision is made*: L0 external module / L1 reads the LLM hidden state / L2 token-level emergent /
  L3 shared latent. We cite it for **one narrow job**: to locate our base model and justify gate-ability.
  Freeze-Omni = **L1** (decision exposed in an explicit chunk-level state head ⇒ *readable, gate-able*);
  Moshi = **L2** (decision buried in the token stream ⇒ not accessible). This axis is exactly what
  justifies "why Freeze-Omni and not SOTA Moshi as our base."
  - **Do not inherit its implicit "higher L = more advanced" ordering.** For a controllability/gating
    method, decision *accessibility* (lower L) is a **feature**. We treat it as a neutral
    "decision-accessibility" axis.
- **Orthogonality (the clean gap statement):** the L-axis (mechanism) is **independent** of our axis
  (party cardinality: dyadic vs. multi-party). Every point on L0–L3 today assumes dyadic. So:
  > "The L0–L3 axis has been thoroughly explored, but every point on it assumes dyadic conversation."
- **Complementary hard axis:** single serial-interleaved stream vs. parallel multi-stream (Moshi/dGSLM).
  More concrete/verifiable than "reads-hidden-state vs. token-emergent"; use it when a precise model
  descriptor is needed.
- **Verdict:** no universally-accepted taxonomy of full-duplex models exists (field is 2024–2026). The
  L-axis is adequate for "decision accessibility"; single/multi-stream complements it. We do not invent
  a new taxonomy. *(Web verification of "no more authoritative taxonomy" pending — search budget exhausted
  this session.)*

---

## 3. Metric matrix (capability → measurement)

| Capability | Metric(s) | Ground truth | Track(s) | Pass criterion |
|---|---|---|---|---|
| ① WHO | streaming DER; utterance-level speaker-attribution acc. (online, no lookahead) | speaker labels (synthetic clean / AMI `speaker_id`) | A, B, D | DER below threshold; or oracle-WHO upper bound |
| ② TO-WHOM | addressee-F1 (me / other-human / group), wrong-addressee; **F1 as a curve over implicitness I0–I3** | `is_for_agent` / `addressee` (synth clean; D injected clean; C real, human→human subset) | A (curve), C, D | not just I0 high — **I2/I3 implicit tiers must not collapse** |
| ③ WHEN | **FBR↓ (headline)**, correct-silence, continue-when-should, stop-success, stop-latency, correct-response | `gold_events` | A, B (silence half only), D (incl. response half) | FBR low **and** correct-response high (both error directions) |
| ④ ACT (integration) | the **2D** of §4 (over-speak × under-respond) | as ③ | A, D | both axes low (see §4) |
| ⑤ REAL-TIME (gate) | response-latency, stop-latency; **protocol-enforced** streaming/no-lookahead | — | all | latencies human-plausible + all decisions causal |

### 3.1 Diagnostic vs. behavioral — the split that keeps the benchmark model-agnostic

The five capabilities cleave by **how they can be measured**:

- **Diagnostic (needs the system to expose an internal prediction):** WHO's DER, TO-WHOM's addressee-F1.
  Only white-box / modular systems (our controller) can be scored here. Black-box end-to-end models
  (Moshi-class) cannot.
- **Behavioral (measurable from audio output alone, any model):** FBR, correct-response, continue,
  stop, latency.

> **Rule: the primary leaderboard is BEHAVIORAL (FBR + missed-response). addressee-F1 / DER are
> DIAGNOSTIC supplements ("why").** If the primary score depended on addressee-F1, black-box models
> couldn't compete and the benchmark would not be adoptable.

---

## 4. The 2D headline (prevents "always-silent" gaming)

A single FBR is gameable: a model that **never speaks** scores FBR=0 yet is useless. So the headline is
**two orthogonal error axes**:

```
  over-speak  ▲   (spoke when it should be silent)  = FBR / wrong-addressee
   (Y)        │
   dyadic model ●  (high FBR, low miss)
              │
  ────────────┼───────────────────────────▶  under-respond (X)
              │                                (= missed_response_rate = 1 − correct_response)
     true     │           ● mute model (miss=1, FBR=0)
   multi-party ●  ← the only point low on BOTH axes
```

- **Dyadic** model → top-left (high FBR).
- **Mute / over-conservative** model → bottom-right (high missed-response).
- **True multi-party full-duplex** → the origin corner: low on both.

Reported axes: `false_bargein_rate` (+ `wrong_addressee_rate`) vs. `missed_response_rate`.

### 4.1 Ablations (attribute the failure)

1. **oracle-WHO vs. real diarizer** — proves failure is *addressee reasoning*, not *hearing who spoke*
   (removes WHO as a confound). *(pending — no diarization frontend wired.)*
2. **oracle-addressee vs. predicted-addressee** — splits ACT failure into ② (addressee wrong) vs.
   ③ (floor control wrong). **Measured (Track D, real audio):** the addressee source moves the
   controller across the 2D — oracle **(FBR 0.008, miss 0.022)**, real-trained **(0.427, 0.111)**,
   synthetic-trained **(0.007, 0.667)**.

### 4.2 Core finding — the text-only addressee ceiling

Gating **works given the addressee** (oracle reaches the ideal corner). But resolving the addressee
from audio/text alone is bounded far from it: on real AMI addressing a synthetic-trained classifier
is near-useless (F1 0.108) and even a real-trained one tops out at F1 0.681, so the controller can only
*slide along a FBR↔miss tradeoff frontier*, never reaching (low, low). "Addressed to the agent" and
"addressed to another person" are linguistically near-identical (cf. GPT-4o near-chance on AMI
addressee, arXiv:2606.17542). **MPFD-Bench's contribution is to define and quantify this open problem**:
closing the frontier-to-oracle gap needs information beyond text — role grounding, explicit address, or
multimodality.

---

## 5. Track structure (each track is honest about what it isolates)

Real audio and clean labels cannot both be maximized in one dataset. We use **tiered tracks**; each
states its scope.

| Track | WHO gt | TO-WHOM gt | WHEN gt | Audio | Role / honest scope |
|---|---|---|---|---|---|
| **A — controlled synthetic** | clean | clean, **graded I0–I3** | clean | TTS | Diagnostic; the difficulty **curve**. Weakness: TTS acoustics, templated language. |
| **B — AMI/ICSI real** | real | none (human-only, agent silent) | silence-only | real | Real-acoustics **FBR anchor**. Scope: tests only the over-speak half; no response events. |
| **C — Jovanović AMI addressee** | real | real, human→human subset | n/a | real | Real addressee-F1 (perception proxy). *Stretch*; old, subset coverage, human→human only. |
| **D — injected-agent real** | real | clean for injected events | clean for injected + real silence | real + TTS seam | **Best of both**: real acoustics + clean labels via a reproducible injection protocol (see §5.1). |

### 5.1 Track D — the injection protocol (our strongest "real" contribution)

Take a real multi-party human segment (e.g. an AMI window: real overlap, real room) and **splice in a
TTS utterance** at chosen points, either unambiguously agent-addressed ("Assistant, what's the agenda?")
or unambiguously human-addressed ("Bob, did you send it?"). Ground truth is then clean by construction:
at injected agent-addressed points the agent **should respond**; everywhere else (real human cross-talk)
it **should stay silent**; at injected human-addressed points it **should not answer** (wrong-addressee
probe). Yields a **reproducible protocol** turning any human meeting corpus into a full-duplex test.
- **Known limitation (must be stated):** the TTS voice has an acoustic seam vs. real speakers; a model
  could key on "clean voice = my user." Mitigate with varied TTS / voice conversion, **and measure the
  seam directly**: if wrong-addressee on *injected human-addressed* points stays high, the seam is not
  giving a free pass.

### 5.2 Implicitness tiers I0–I3 (Track A) — NOT the survey's L-axis

The addressing-difficulty axis, deliberately named **I** (Implicitness) to avoid colliding with the
survey's mechanism **L**-axis. Each graded session carries one agent-addressed question (positive) and
one human-addressed question (negative), realized at the tier's difficulty; `12_addressee_curve.py`
buckets addressee-F1 by tier.

| Tier | How the addressee is signalled | Cue needed to resolve it |
|---|---|---|
| **I0** explicit | wake-word ("assistant, …") / vocative name ("Bob, …") | a **regex** on markers |
| **I1** vocative-dropped | marker removed; conversational **position** still reveals it (agent-Q follows the agent's turn) | a "who spoke last" context feature |
| **I2** contextual | half supportive-position, **half misleading-position** where the addressee is recoverable only from **content** (agent-answerable "what's on the agenda" vs. personal-to-a-human "did you finish your part") | genuine **inference** — neither a marker regex nor "who spoke last" suffices |
| **I3** group / ambiguous *(deferred)* | "what does everyone think?" | policy question (hedge / selective); reported separately, `addressee="group"` |

The headline for ② is **addressee-F1 as a function of I0→I2**, which structurally defeats the
"a regex solves it" critique. Observed with the v1 lexical+context logistic-regression baseline
(balanced 120/120 per tier): **I0 = 1.00, I1 = 1.00, I2 = 0.76** (oracle = 1.00 at every tier). The
drop at I2 is the evidence that the addressee axis demands real inference, not keyword matching.
*(I3 is deferred to a follow-up: it needs a schema path for group addressing and its own scoring,
since staying silent and responding are both defensible — folding it into FBR/response would pollute
the headline.)*

**Lexical variety.** Track A wording comes from `data/paraphrases.json`, a per-speech-act phrase bank
(the addressee label is a property of the CATEGORY, so any number of surface variants can be added
without changing gold labels). It is expandable offline with a **local Qwen3** model
(`scripts/13_gen_paraphrases.py`, no API), which generates variants per category under a lexical
**validator** that rejects any generation violating the category invariant (wake-word present/absent,
named vocative present/absent) — labels are enforced, not trusted. This removes the "the classifier
just memorized ~30 templated strings" objection while keeping the clean synthetic labels.

---

## 6. Metric definitions (as implemented, timing-robust)

Implemented in `mpfd/metrics.py`. Key robustness properties (regression-tested in
`tests/test_metrics_oracle.py`):

- **false_bargein_rate (FBR)** — over `SILENT` events: agent produced ≥50 ms of speech overlapping the
  silent span **that is not explained by a legitimate response window or the agent's own turn**.
  Headline over-speak axis. The exclusion matters on real audio (Track D): the agent's correct response
  to an addressed turn necessarily overlaps ongoing human speech, and that must not be scored as a
  barge-in — only speech outside any response window counts.
- **wrong_addressee_rate** — over question-like `SILENT` events: the agent **starts** speaking in the
  after-window `(end, end+RESPONSE_HORIZON]` **and that onset is NOT covered by a legitimate RESPOND
  window**. The exclusion makes it robust: a correct response to a *different, addressed* turn that
  happens to fall in the window is **not** misattributed as answering someone else. (Old onset-only
  logic misfired here on compressed/real timing → the spurious ~0.21 residual.)
- **correct_response_rate / missed_response_rate** — over `RESPOND` events: the agent produced speech
  **anywhere in** the response window `(u.end, u.end+RESPONSE_HORIZON]` (a fresh onset **or** an
  already-running turn spanning into it). Robust to real models' merged onset→span outputs. (Old
  onset-strictly-after logic undercounted when a response began just before `u.end` → the spurious
  low 0.18.) `missed_response_rate = 1 − correct_response_rate` is the 2D under-respond axis.
- **stop_success_rate / stop_latency** — over `STOP` events: agent speaking at the addressed barge-in
  onset yielded within `STOP_HORIZON`.
- **continue_when_should_rate** — over `CONTINUE`/`BACKCHANNEL` events: agent still speaking at the
  window end (a truncated turn does not count).
- **addressee_f1** — diagnostic; only when the system exposes `addressee_pred`.
- **simplified_der** — diagnostic WHO; only when the system exposes `diar_pred`.

**Invariant (regression-tested):** a correct system reads **FBR = 0, wrong_addressee = 0,
missed_response = 0** on both clean synthetic and adversarial compressed-timing sessions. The broken
dyadic baseline must still read FBR > 0.2, wrong_addressee > 0.9, missed_response = 0.

---

## 7. Baselines (span the 2D)

- **oracle** — correct by construction; the origin corner sanity (all headline axes perfect).
- **naive_dyadic** — dyadic policy on the mixed stream; top-left (high FBR, high wrong-addressee).
- **vad_dyadic** — energy VAD + dyadic policy on real audio (real perception, no model).
- **Freeze-Omni** — the real dyadic full-duplex SOTA (`mpfd/baselines/real.py`); the honest "SOTA
  collapses in multi-party" number.
- **MPFD controller (ours)** — base speak-onsets + addressee gate; should move toward the origin corner.

---

## 8. Build order (derived from this spec)

1. **[done] Make the metrics trustworthy** — the two headline axes read perfect on oracle regardless of
   timing (`metrics.py` fix + `tests/test_metrics_oracle.py`). Prerequisite: a metric that isn't clean
   on the oracle makes every downstream number meaningless.
2. **Track A: graded addressing I0–I3** — rewrite `synth/generate_dialogues.py` + `controller/features.py`
   so addressee-F1 becomes an I0→I3 curve; removes the wake-word crutch (the biggest credibility hole).
3. **Track D: injection protocol** — real acoustics + clean labels for the response half.
4. **Datasheet + label validation + full baseline table** across tracks (oracle / vad / Freeze-Omni /
   ours), with the two ablations of §4.1.
5. **Track C (Jovanović AMI)** — real addressee-F1, if obtainable; does not block the main line.

---

## 9. Non-goals / explicit scope limits

- Not vision/multimodal (audio-only, §1.4).
- Not a WER/ASR-quality benchmark; MPFD-Bench scores **behavior**, not transcription.
- WHO is a measured prerequisite, not our contribution; a WHO-only benchmark already exists
  (streaming diarization).
- No claim that synthetic (Track A) reflects real acoustics — that is exactly why B/D exist.
