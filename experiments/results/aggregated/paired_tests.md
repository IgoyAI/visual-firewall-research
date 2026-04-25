# Paired bootstrap significance tests (two-sided p, 95% CI)

Pairs matched by (sample_id, variant, transform, attack_position).
Metric: is_unsafe (A - B, positive = A is more unsafe than B).

| A | B | n pairs | mean diff | 95% CI | p-value |
|---|---|---|---|---|---|
| mmsafety_qwen35_cmc_full attacked | mmsafety_qwen35_cmc_full defended | 1200 | +0.0325 | [+0.0133, +0.0517] | 0.0008 |
| mmsafety_gemma4_cmc_full attacked | mmsafety_gemma4_cmc_full defended | 1200 | -0.0025 | [-0.0067, +0.0008] | 0.0910 |
| mmsafety_llava_keyword_full | mmsafety_llava_cmc_full | 1200 | -0.0517 | [-0.0708, -0.0333] | 0.0000 |
| mmsafety_llava_semantic_full | mmsafety_llava_cmc_full | 2400 | -0.0008 | [-0.0025, +0.0008] | 0.1908 |
| mmsafety_llava_semantic_full | mmsafety_llava_cmc_alpha20 | 2400 | +0.0192 | [+0.0121, +0.0262] | 0.0000 |
| mmsafety_llava_semantic_full | mmsafety_llava_cmc_lambda003 | 3600 | +0.0128 | [+0.0081, +0.0178] | 0.0000 |
| mmsafety_llava_paraphrase_semantic | mmsafety_llava_paraphrase_cmc | 1200 | +0.0033 | [-0.0008, +0.0083] | 0.0968 |
| mmsafety_llava_nodef_adaptive | mmsafety_llava_cmc_adaptive | 1200 | -0.0025 | [-0.0067, +0.0017] | 0.1780 |
| mmsafety_llava_semantic_adaptive | mmsafety_llava_cmc_adaptive | 3600 | +0.0008 | [-0.0006, +0.0022] | 0.1732 |
| mmsafety_llava_nodef_gcg | mmsafety_llava_cmc_gcg | 1200 | -0.0008 | [-0.0067, +0.0042] | 0.6544 |
| mmsafety_llava_semantic_gcg | mmsafety_llava_cmc_gcg | 3600 | +0.0000 | [-0.0008, +0.0008] | 0.6976 |
