# Visual Firewall — Conformal Cross-Modal Defenses for Visual Prompt Injection

Research codebase for the paper:

> **Conformal Cross-Modal Firewalls: Calibrated Pre-Model Defenses for Visual Prompt Injection** *(submitted to NeurIPS 2026)*

The CMC firewall is a **pre-model** defense that sits between the input image
and the multimodal LLM. It (i) extracts OCR spans, (ii) scores each span with a
frozen SigLIP encoder using a two-component risk derived as a Neyman–Pearson
log-likelihood ratio under a von Mises–Fisher embedding model, and
(iii) calibrates the masking threshold by **inductive conformal prediction** so
that the clean-image false-positive rate is bounded by `α + 1/(n+1)` —
distribution-free, finite-sample, under exchangeability.

## Headline numbers (LLaVA-1.5-7B, MM-SafetyBench, six transforms)

| Condition | Unsafe rate (95% CI) | Attack flag | Clean flag |
|---|---|---|---|
| Base (no attack) | 15.0% [10.5, 20.5] | — | — |
| Attacked, no defense | 19.5% [17.5, 21.8] | — | — |
| Keyword blocklist | 13.9% [12.0, 15.8] | 100.0% | 100.0% (no FPR control) |
| Semantic-only (τ=0.78) | 19.0% [17.1, 21.3] | 21.5% | 4.3% (hand-tuned) |
| **CMC + conformal (α=0.20, λ=0.03)** | **15.4% [13.4, 17.5]** | **81.2%** | **12.4% < bound 21.0%** |

Cross-model validation: Qwen3.5-9B 17.2 → 13.9% (paired bootstrap −3.25 pp,
*p*=0.0008); Gemma 4 E4B-it is a non-degradation case (safety training
already binds at 0.5%). Larger split *n*=500 reproduces the gap with disjoint
CIs (Δ=−6.5 pp). MMBench MCQ utility retained at 90.0%.

See `experiments/results/aggregated/ALL_TABLES.md` for every paper table
generated from raw `summary.json` files.

## Hardware

- 1× H100-class GPU (80 GB) is enough for the headline LLaVA pipeline at
  `batch_size=32`. We developed on 2× H100 NVL with MIG slices of 47.5 GiB
  (so `cuda:0` runs the target MLLM and `cuda:1` runs Llama-Guard-3-8B + the
  SigLIP defense scorer).
- For Qwen3.5-9B and Gemma 4 E4B-it specifically, set
  `"use_two_gpus": false` in the config so the MLLM stays on `cuda:0` only;
  otherwise `accelerate`'s `device_map="auto"` will spread it onto `cuda:1`
  and OOM the judge.
- ~60 GB free disk for HuggingFace model cache, plus ~500 MB for the
  MM-SafetyBench parquets.

## Setup

```bash
# 1. Clone
git clone https://github.com/IgoyAI/visual-firewall-research.git
cd visual-firewall-research

# 2. Environment + data + models (~30 min, one-time)
bash scripts/setup.sh                  # full setup
# bash scripts/setup.sh --no-models    # if you'll fetch models manually

# 3. Smoke test (~3 min — verifies env, data, models, full pipeline)
bash scripts/smoke_test.sh

# 4. Full reproduction (~6–8 hours on 2× H100)
bash scripts/reproduce.sh
```

The setup script needs `conda` (Miniforge or Miniconda) and `hf` CLI (the
HuggingFace `hf` command, included in `huggingface_hub`). Some models are
gated (Llama-Guard, Gemma 4) — accept the licenses on HuggingFace and run
`hf auth login` before `setup.sh`.

## Repository layout

```
.
├── configs/                  # one JSON per experiment (39 total)
├── data/
│   ├── README.md             # how to fetch MM-SafetyBench
│   └── mm_safetybench/       # downloaded parquets (gitignored)
├── experiments/results/
│   ├── <run>/summary.json    # committed; aggregator reads these
│   ├── <run>/records.csv     # gitignored (heavy)
│   └── aggregated/*.md       # paper-ready tables, regenerable
├── scripts/
│   ├── setup.sh              # env + data + models
│   ├── smoke_test.sh         # ~3 min sanity check
│   ├── reproduce.sh          # full master queue
│   ├── run_real_experiment.py        # single-config runner entry point
│   ├── run_neurips_queue*.sh         # ordered sub-queues used by reproduce
│   ├── aggregate_neurips_tables.py   # summary.json → markdown tables
│   ├── verify_conformal.py / conformal_bootstrap.py  # FPR-bound checks
│   ├── verify_theorems.py            # Thm 1–4 + vMF fit
│   └── make_*figure.py               # figure regeneration
└── src/
    ├── attacks/              # visible / bounded / paraphrase / pool / GCG
    ├── defenses/
    │   ├── ocr_firewall.py   # keyword baseline
    │   ├── semantic_firewall.py  # SigLIP + reference bank only
    │   └── cmc_firewall.py   # the proposed defense
    ├── eval/                 # runner, model backends, judge, metrics
    └── transforms/           # six image transforms
```

## How to add a new experiment

1. Drop a JSON config in `configs/`. Easiest: copy
   `configs/mmsafety_llava_cmc_full.json` and tweak `defense_type`,
   `cmc_lambda`, `cmc_calibrate_fpr`, `attack_type`, `model_path`, etc.
2. Run it: `python scripts/run_real_experiment.py --config configs/<your>.json`
3. The result lands in `experiments/results/<output_dir_basename>/`.
4. Re-aggregate: `python scripts/aggregate_neurips_tables.py`

If you want it included in `scripts/reproduce.sh`, append it to one of the
`run_neurips_queue*.sh` scripts.

## How to add a new defense

Implement `apply(image) -> FirewallResult` in `src/defenses/<your>.py`,
following the shape of `cmc_firewall.py`. Then add a branch in
`src/eval/real_runner.py` near the existing `if config.defense_type == "cmc"`
block.

## Citation

```bibtex
@inproceedings{visual_firewall_2026,
  title  = {Conformal Cross-Modal Firewalls: Calibrated Pre-Model Defenses
            for Visual Prompt Injection},
  author = {Anonymous Authors},
  booktitle = {Submitted to NeurIPS 2026},
  year   = {2026}
}
```

## License

MIT — see `LICENSE`. MM-SafetyBench and all referenced model weights remain
under their respective upstream licenses.
