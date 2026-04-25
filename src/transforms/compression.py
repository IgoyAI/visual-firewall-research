"""Compression-based perturbations for robustness evaluation."""

from __future__ import annotations

import io

from PIL import Image


def apply_jpeg_roundtrip(image: Image.Image, quality: int = 28) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
