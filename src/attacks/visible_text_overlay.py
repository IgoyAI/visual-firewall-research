"""Visible-text attack used for the phase-1 pilot experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


@dataclass(frozen=True)
class OverlayConfig:
    text: str
    position: str = "bottom_right"
    font_size: int = 34
    margin: int = 28
    max_width_ratio: float = 0.42
    text_fill: Tuple[int, int, int] = (0, 0, 0)
    box_fill: Tuple[int, int, int] = (255, 255, 255)
    outline_fill: Tuple[int, int, int] = (30, 30, 30)
    box_padding_x: int = 18
    box_padding_y: int = 14


def apply_visible_text_overlay(image: Image.Image, config: OverlayConfig) -> tuple[Image.Image, dict]:
    """Overlay readable malicious text onto an image and return metadata."""

    attacked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(attacked)
    font = _load_font(config.font_size, bold=True)
    wrapped = _wrap_text(draw, config.text, font, int(attacked.width * config.max_width_ratio))
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6)

    box_width = (bbox[2] - bbox[0]) + 2 * config.box_padding_x
    box_height = (bbox[3] - bbox[1]) + 2 * config.box_padding_y

    if config.position == "top_right":
        x0 = attacked.width - config.margin - box_width
        y0 = config.margin
    elif config.position == "top_left":
        x0 = config.margin
        y0 = config.margin
    elif config.position == "center":
        x0 = (attacked.width - box_width) // 2
        y0 = (attacked.height - box_height) // 2
    else:
        x0 = attacked.width - config.margin - box_width
        y0 = attacked.height - config.margin - box_height

    x1 = x0 + box_width
    y1 = y0 + box_height

    draw.rectangle((x0, y0, x1, y1), fill=config.box_fill, outline=config.outline_fill, width=3)
    text_origin = (x0 + config.box_padding_x, y0 + config.box_padding_y)
    draw.multiline_text(text_origin, wrapped, font=font, fill=config.text_fill, spacing=6)

    return attacked, {
        "attack_text": config.text,
        "wrapped_text": wrapped,
        "overlay_bbox": [x0, y0, x1, y1],
        "position": config.position,
    }
