# Eval

Put experiment orchestration and metrics here.

Suggested first modules:

- `datasets.py`
- `runner.py`
- `metrics.py`
- `judge.py`

Phase 1 focus:

- benchmark subset loading
- attack and defense execution order
- ASR and benign retention reporting

## Real experiment path

For non-synthetic runs, use:

- `benchmark_loader.py`: loads a real manifest from `json`, `jsonl`, or `.parquet`
- `model_backends.py`: loads either a local Hugging Face multimodal generator or the local `SigLIP` MCQ scorer
- `real_runner.py`: runs the attack/defense pipeline on real images
- `judge.py`: lightweight response judging for MCQ, exact, contains, or refusal
- `ocr_firewall.py`: supports either `tesseract` or CUDA-backed `easyocr`

Expected manifest format:

```json
{"id":"sample_001","image":"images/example.png","prompt":"Describe the image.","answer":"A","choices":["A. ...","B. ..."]}
```

The loader also supports cached `MMBench` parquet shards directly, including inline image bytes.

Use [configs/real_experiment.example.json](/home/e/e1507650/visual-firewall-research/configs/real_experiment.example.json) as the launch template.

Concrete `MMBench + SigLIP` configs:

- [configs/mmbench_siglip_dev_subset.json](/home/e/e1507650/visual-firewall-research/configs/mmbench_siglip_dev_subset.json)
- [configs/mmbench_siglip_dev_full.json](/home/e/e1507650/visual-firewall-research/configs/mmbench_siglip_dev_full.json)
