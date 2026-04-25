"""Attack implementations for visual prompt injection experiments."""

from src.attacks.bounded_overlay import BoundedOverlayConfig, apply_bounded_overlay
from src.attacks.paraphrase_overlay import (
    PARAPHRASED_ATTACKS,
    ParaphraseOverlayConfig,
    apply_paraphrase_overlay,
    get_paraphrase_for_index,
)
from src.attacks.visible_text_overlay import OverlayConfig, apply_visible_text_overlay

__all__ = [
    "OverlayConfig",
    "apply_visible_text_overlay",
    "BoundedOverlayConfig",
    "apply_bounded_overlay",
    "PARAPHRASED_ATTACKS",
    "ParaphraseOverlayConfig",
    "apply_paraphrase_overlay",
    "get_paraphrase_for_index",
]
