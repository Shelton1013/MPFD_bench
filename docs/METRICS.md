# MPFD-Bench Metrics

Full-duplex dialogue has **no single WER-style metric**; evaluation is multi-dimensional. We
inherit the standard full-duplex metrics and **add multi-party-specific ones** (no prior
benchmark measures these). All are computed in `mpfd/metrics.py` from a `Session` (gold) and a
`SystemOutput` (the agent's actual speech intervals + optional addressee/diarization predictions).

## New multi-party metrics (our contribution)
| Metric | Definition | Good |
|---|---|---|
| **false_bargein_rate (FBR)** | over SILENT events (inter-human / third-party), fraction where the agent wrongly spoke | **low** (headline) |
| correct_silence_rate | 1 − FBR | high |
| correct_response_rate | over addressed turns, fraction the agent actually answered (within the response horizon) | high |
| wrong_addressee_rate | over question-like utterances addressed to another human, fraction the agent answered | low |
| stop_success_rate | over addressed barge-ins, fraction the agent stopped within the stop horizon | high |
| continue_on_backchannel_rate | over listener backchannels during agent speech, fraction the agent kept speaking | high |
| addressee_f1 | F1 of predicted "addressed-to-agent" vs. gold (if the system predicts it) | high |
| attribution DER | streaming diarization error on the conditioning stream (`simplified_der`, or use dscore) | low |

## Inherited full-duplex metrics
- mean_response_latency_s (first agent onset after an addressed turn ends)
- mean_stop_latency_s (halt time after an addressed barge-in onset)
- (server) self-WER of the agent's output, MOS/UTMOS, LLM-as-judge response quality.

## Protocol
- Report **per-cell + aggregate**.
- Two settings: **mixed far-field stream** (primary, realistic) and **oracle-channel +
  oracle-addressee** (upper bound) — the gap isolates *perception* error from *reasoning* error.
- Baselines: dyadic FD on mixed input (expected high FBR / wrong-addressee), modular L0
  (diarization + heuristic addressee), our controller (P2/P3), oracle.

## Sanity check
`oracle` system ≈ perfect (FBR≈0, correct_response≈1, addressee_f1≈1); `naive_dyadic` shows the
failure (high FBR, high wrong_addressee, stops on backchannels). This validates the metrics
before any real model is plugged in.
