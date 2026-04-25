"""Image preprocessing transform registry used in the pilot evaluation."""

from __future__ import annotations

from typing import Callable, Dict

from PIL import Image

from src.transforms.blur import apply_mild_blur
from src.transforms.compression import apply_jpeg_roundtrip
from src.transforms.crop import apply_center_crop_roundtrip
from src.transforms.rerender import apply_screenshot_rerender
from src.transforms.resize import apply_resize_roundtrip


def build_transform_registry() -> Dict[str, Callable[[Image.Image], Image.Image]]:
    return {
        "clean": lambda image: image.copy(),
        "resize": apply_resize_roundtrip,
        "jpeg": apply_jpeg_roundtrip,
        "crop": apply_center_crop_roundtrip,
        "rerender": apply_screenshot_rerender,
        "blur": apply_mild_blur,
    }
