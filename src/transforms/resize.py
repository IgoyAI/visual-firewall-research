"""Resize-based perturbations for robustness evaluation."""

from __future__ import annotations

from PIL import Image


def apply_resize_roundtrip(image: Image.Image, scale: float = 0.55) -> Image.Image:
    width, height = image.size
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)
    return resized.resize((width, height), Image.Resampling.BICUBIC)
