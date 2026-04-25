"""Blur perturbations for robustness evaluation."""

from __future__ import annotations

from PIL import Image, ImageFilter


def apply_mild_blur(image: Image.Image, radius: float = 1.25) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
