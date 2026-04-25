"""Typed experiment configuration and record structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    samples: int = 12
    seed: int = 7
    output_dir: Path = Path("experiments/results/pilot_run")
    transforms: tuple[str, ...] = ("clean", "resize", "jpeg", "crop", "rerender", "blur")
    attack_positions: tuple[str, ...] = ("bottom_right", "top_right", "center")
    use_multiview_firewall: bool = True
    use_gpu_encoder: bool = True
    sample_preview_limit: int = 2


@dataclass
class ExperimentRecord:
    sample_id: str
    variant: str
    transform: str
    attack_text: str
    attack_detected_before: int
    attack_detected_after: int
    anchor_detected_before: int
    anchor_detected_after: int
    firewall_flagged: int
    num_suspicious_boxes: int
    before_ocr_text: str
    after_ocr_text: str
    attack_position: str
    gpu_similarity_before: float | None = None
    gpu_similarity_after: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentSummary:
    values: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.values)
