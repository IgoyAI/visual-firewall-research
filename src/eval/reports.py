"""Output helpers for experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.eval.types import ExperimentConfig, ExperimentRecord


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, records: list[ExperimentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].to_dict().keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def write_summary_markdown(path: Path, config: ExperimentConfig, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pilot Experiment Summary",
        "",
        "This run is a synthetic OCR-proxy pilot for the visual firewall pipeline.",
        "",
        f"- Samples: {config.samples}",
        f"- Seed: {config.seed}",
        f"- Multi-view firewall: {config.use_multiview_firewall}",
        f"- Clean ASR proxy: {summary['clean_asr_proxy']:.4f}",
        f"- Defended clean ASR proxy: {summary['defended_clean_asr_proxy']:.4f}",
        f"- Transformed ASR proxy: {summary['transformed_asr_proxy']:.4f}",
        f"- Defended transformed ASR proxy: {summary['defended_transformed_asr_proxy']:.4f}",
        f"- Control false-positive rate: {summary['control_false_positive_rate']:.4f}",
        f"- Control benign retention after defense: {summary['control_benign_retention_after']:.4f}",
        f"- GPU attack similarity before defense: {summary['gpu_attack_similarity_before']:.4f}",
        f"- GPU attack similarity after defense: {summary['gpu_attack_similarity_after']:.4f}",
        f"- GPU control similarity after defense: {summary['gpu_control_similarity_after']:.4f}",
        "",
        "## Per-transform",
        "",
    ]
    for transform_name, metrics in summary["per_transform"].items():
        lines.append(
            f"- `{transform_name}`: ASR proxy {metrics['attack_detection_rate']:.4f}, "
            f"defended {metrics['defended_attack_rate']:.4f}, "
            f"anchor retention {metrics['benign_anchor_retention_after']:.4f}, "
            f"GPU sim before {metrics['gpu_similarity_before']:.4f}, "
            f"after {metrics['gpu_similarity_after']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n")
