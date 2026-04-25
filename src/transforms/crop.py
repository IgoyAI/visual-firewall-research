"""Crop-based perturbations for robustness evaluation."""

from __future__ import annotations

from PIL import Image


def apply_center_crop_roundtrip(image: Image.Image, crop_ratio: float = 0.82) -> Image.Image:
    width, height = image.size
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)
