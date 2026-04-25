"""Bounded low-salience text overlay attack for the visual firewall project.

Family 2 attack: smaller, lower-contrast text placed in image corners to
simulate realistic semi-visible prompt injection that may evade casual
human inspection while remaining readable to MLLMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from PIL import Image

from src.attacks.visible_text_overlay import OverlayConfig, apply_visible_text_overlay


@dataclass(frozen=True)
class BoundedOverlayConfig:
    text: str
    position: str = "bottom_right"
    font_size: int = 16
    margin: int = 8
    max_width_ratio: float = 0.25
    text_fill: Tuple[int, int, int] = (120, 120, 120)
    box_fill: Tuple[int, int, int] = (245, 245, 245)
    outline_fill: Tuple[int, int, int] = (200, 200, 200)
    box_padding_x: int = 6
    box_padding_y: int = 4


def apply_bounded_overlay(image: Image.Image, config: BoundedOverlayConfig) -> tuple[Image.Image, dict]:
    """Apply a low-salience bounded text overlay attack."""
    overlay_config = OverlayConfig(
        text=config.text,
        position=config.position,
        font_size=config.font_size,
        margin=config.margin,
        max_width_ratio=config.max_width_ratio,
        text_fill=config.text_fill,
        box_fill=config.box_fill,
        outline_fill=config.outline_fill,
        box_padding_x=config.box_padding_x,
        box_padding_y=config.box_padding_y,
    )
    attacked, meta = apply_visible_text_overlay(image, overlay_config)
    meta["attack_family"] = "bounded_overlay"
    return attacked, meta
