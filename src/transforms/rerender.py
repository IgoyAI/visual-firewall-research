"""Re-render perturbations that approximate screenshot-like processing."""

from __future__ import annotations

from PIL import Image


def apply_screenshot_rerender(image: Image.Image, border: int = 26) -> Image.Image:
    width, height = image.size
    canvas = Image.new("RGB", (width + 2 * border, height + 2 * border + 40), (236, 238, 242))
    canvas.paste((248, 248, 248), (border, 12, width + border, 36))
    canvas.paste(image.copy(), (border, border + 24))
    return canvas.resize((width, height), Image.Resampling.BICUBIC)
