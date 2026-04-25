"""Runtime attack wrapper for the GCG-trained universal suffix.

Loads `artifacts/gcg_suffix.json` (produced by scripts/train_gcg.py)
and overlays `intent + " " + suffix` on the image. This is a fixed
universal transfer attack: the same string is used for every image,
because GCG was trained to minimize the mean CMC risk across a
training batch, yielding a string that generalizes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.attacks.visible_text_overlay import OverlayConfig, apply_visible_text_overlay


DEFAULT_ARTIFACT = Path("artifacts/gcg_suffix.json")


@dataclass(frozen=True)
class GCGAttackConfig:
    artifact_path: str | Path = DEFAULT_ARTIFACT
    position: str = "bottom_right"
    font_size: int = 30
    margin: int = 28
    max_width_ratio: float = 0.46


class GCGAttacker:
    def __init__(self, config: GCGAttackConfig) -> None:
        self.config = config
        path = Path(config.artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"GCG artifact not found at {path}. Run scripts/train_gcg.py first."
            )
        data = json.loads(path.read_text())
        self.text = data["final_text"]
        self.final_loss = float(data.get("final_loss", float("nan")))
        self.intent = data.get("intent", "")


def apply_gcg_overlay(
    image: Image.Image, attacker: GCGAttacker, position: str | None = None,
) -> tuple[Image.Image, dict]:
    overlay = OverlayConfig(
        text=attacker.text,
        position=position or attacker.config.position,
        font_size=attacker.config.font_size,
        margin=attacker.config.margin,
        max_width_ratio=attacker.config.max_width_ratio,
    )
    attacked, meta = apply_visible_text_overlay(image, overlay)
    meta["attack_family"] = "gcg_whitebox_universal"
    meta["gcg_text"] = attacker.text
    meta["gcg_training_final_loss"] = attacker.final_loss
    return attacked, meta
