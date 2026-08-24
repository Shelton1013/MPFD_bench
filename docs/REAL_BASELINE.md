# Running real baselines (rendered audio → real numbers)

The simulated `naive_dyadic` reads the gold script. Real numbers come from **rendered audio +
real perception/model**. Two levels:

## Level 1 — VAD dyadic baseline (runs immediately, no model)
Real perception (energy VAD on the mixed wav) + the dyadic policy. This is the honest "perfect-ish
perception, zero addressee reasoning on real audio" floor.
```bash
python scripts/01_generate_synth.py --out data/synthetic --n_per_cell 50
python scripts/02_render_audio.py   --sessions data/synthetic --out data/rendered --tts cosyvoice2
python scripts/04_run_real.py       --sessions data/rendered --pred vad_dyadic
```
Expect: high false_bargein_rate + wrong_addressee (it answers inter-human turns), addressee_f1 low.

## Level 2 — Freeze-Omni (the real dyadic full-duplex model)
This is the headline baseline: feed the rendered mixed audio to Freeze-Omni and read WHEN it
decides to speak. Wire `mpfd/baselines/real.py::FreezeOmniBaseline` to the repo
(https://github.com/VITA-MLLM/Freeze-Omni):

1. In `__init__`: load the model as in Freeze-Omni's inference/`server` script (streaming speech
   encoder + frozen Qwen2-7B + the chunk-level **state classifier**).
2. In `__call__`: feed the wav in `chunk_ms` chunks; at each chunk read the model's dialogue-state
   decision (`{listen / interrupt / respond}`). Mark the chunk as agent-speaking when the state is
   `respond` (or when the model is actively generating output audio). The loop already turns those
   into `agent_segments` and feeds the metrics.
3. Freeze-Omni is dyadic → it has no addressee output; `addressee_pred` stays all-True (which is
   exactly the failure MPFD-Bench measures).
```bash
python scripts/04_run_real.py --sessions data/rendered --pred freeze_omni --model_path /path/to/Freeze-Omni
```

## Interpreting
- **false_bargein_rate** is the headline: a real dyadic model on multi-party audio should be high
  (it speaks during inter-human exchanges).
- Compare Level-1 (VAD) vs Level-2 (Freeze-Omni): similar failure pattern confirms it's the
  *dyadic assumption*, not a perception artifact.
- These are the numbers that motivate P2 (the addressee/floor controller).
