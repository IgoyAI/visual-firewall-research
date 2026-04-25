"""Metrics for the synthetic pilot run."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.eval.types import ExperimentRecord


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_records(records: list[ExperimentRecord]) -> dict:
    attacked = [record for record in records if record.variant == "attack"]
    controls = [record for record in records if record.variant == "control"]

    by_transform: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in attacked:
        by_transform[record.transform].append(record)

    transform_summary = {}
    for transform, transform_records in sorted(by_transform.items()):
        transform_summary[transform] = {
            "attack_detection_rate": round(mean(record.attack_detected_before for record in transform_records), 4),
            "defended_attack_rate": round(mean(record.attack_detected_after for record in transform_records), 4),
            "benign_anchor_retention_after": round(mean(record.anchor_detected_after for record in transform_records), 4),
            "gpu_similarity_before": round(mean(record.gpu_similarity_before for record in transform_records if record.gpu_similarity_before is not None), 4),
            "gpu_similarity_after": round(mean(record.gpu_similarity_after for record in transform_records if record.gpu_similarity_after is not None), 4),
        }

    return {
        "num_attack_records": len(attacked),
        "num_control_records": len(controls),
        "clean_asr_proxy": round(mean(record.attack_detected_before for record in attacked if record.transform == "clean"), 4),
        "defended_clean_asr_proxy": round(mean(record.attack_detected_after for record in attacked if record.transform == "clean"), 4),
        "transformed_asr_proxy": round(mean(record.attack_detected_before for record in attacked if record.transform != "clean"), 4),
        "defended_transformed_asr_proxy": round(mean(record.attack_detected_after for record in attacked if record.transform != "clean"), 4),
        "control_false_positive_rate": round(mean(record.firewall_flagged for record in controls), 4),
        "control_benign_retention_after": round(mean(record.anchor_detected_after for record in controls), 4),
        "gpu_attack_similarity_before": round(mean(record.gpu_similarity_before for record in attacked if record.gpu_similarity_before is not None), 4),
        "gpu_attack_similarity_after": round(mean(record.gpu_similarity_after for record in attacked if record.gpu_similarity_after is not None), 4),
        "gpu_control_similarity_after": round(mean(record.gpu_similarity_after for record in controls if record.gpu_similarity_after is not None), 4),
        "per_transform": transform_summary,
    }
