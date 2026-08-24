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

## Level 2 — Freeze-Omni (the real dyadic full-duplex model) — WIRED (source-verified)
The headline baseline. `FreezeOmniBaseline` is implemented against Freeze-Omni's offline pipeline
(`bin/inference.py`): it feeds the rendered mixed wav in **160 ms / 2560-sample @16 kHz** chunks
through `pipeline.speech_dialogue(...)` and reads the chunk-level state head each step —
`stat=='ss'` is the model's **decide-to-speak (barge-in)** signal. We keep it listening across the
whole session (force `stat='cl'`) to record every barge-in onset, then form nominal speak spans.
Freeze-Omni is dyadic → no addressee output → `addressee_pred` is all-True (exactly the failure we
measure).

**Run in Freeze-Omni's OWN env** (it pins `torch==2.2.0`, `transformers==4.45.2`), separate from
the render env — our `mpfd` package is pure-numpy and imports fine there.
```bash
# 1) set up Freeze-Omni per its README (separate conda env):
git clone https://github.com/VITA-MLLM/Freeze-Omni && cd Freeze-Omni
conda create -n freezeomni python=3.10 -y && conda activate freezeomni
pip install -r requirements.txt
#    download its checkpoints/ (audiollm/{train.yaml,global_cmvn,final.pt}) and Qwen2-7B-Instruct
# 2) run the baseline on the already-rendered audio:
python scripts/04_run_real.py --sessions data/rendered --pred freeze_omni \
  --fo_repo /path/to/Freeze-Omni \
  --model_path /path/to/Freeze-Omni/checkpoints \
  --llm_path /path/to/Qwen2-7B-Instruct
```
Notes: needs a CUDA GPU (code uses `torch.device('cuda')` + bf16). If `speech_dialogue`'s exact
kwargs differ in your checkout, the error will point to it — the call pattern follows
`bin/inference.py`'s `inference()` loop verbatim (prime with `stat='pre'`, then feed fbank chunks).

## Interpreting
- **false_bargein_rate** is the headline: a real dyadic model on multi-party audio should be high
  (it speaks during inter-human exchanges).
- Compare Level-1 (VAD) vs Level-2 (Freeze-Omni): similar failure pattern confirms it's the
  *dyadic assumption*, not a perception artifact.
- These are the numbers that motivate P2 (the addressee/floor controller).
