"""Dual-GPU local image encoder utilities for experiment instrumentation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


DEFAULT_SIGLIP_SNAPSHOT = Path(
    "/home/e/e1507650/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/"
    "snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
)


def discover_local_siglip_snapshot() -> Path | None:
    if DEFAULT_SIGLIP_SNAPSHOT.exists():
        return DEFAULT_SIGLIP_SNAPSHOT
    return None


@dataclass(frozen=True)
class GPUEncoderConfig:
    snapshot_path: Path
    devices: tuple[str, str] = ("cuda:0", "cuda:1")
    dtype: torch.dtype = torch.float16


class DualGPUSiglipImageEncoder:
    """Runs local SigLIP image embedding inference across two GPUs."""

    def __init__(self, config: GPUEncoderConfig) -> None:
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(
            str(config.snapshot_path),
            local_files_only=True,
            use_fast=False,
        )
        self.models = {
            device: AutoModel.from_pretrained(
                str(config.snapshot_path),
                local_files_only=True,
                dtype=config.dtype,
            ).to(device).eval()
            for device in config.devices
        }

    def _encode_chunk(self, pixel_values: torch.Tensor, device: str) -> torch.Tensor:
        if pixel_values.numel() == 0:
            return torch.empty((0, 1152), dtype=torch.float32)
        with torch.no_grad():
            features = self.models[device].get_image_features(pixel_values=pixel_values.to(device=device, dtype=self.config.dtype))
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = F.normalize(features.float(), dim=-1)
        return features.cpu()

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        splits = torch.tensor_split(pixel_values, len(self.config.devices), dim=0)

        with ThreadPoolExecutor(max_workers=len(self.config.devices)) as executor:
            futures = [
                executor.submit(self._encode_chunk, split, device)
                for split, device in zip(splits, self.config.devices)
            ]
        outputs = [future.result() for future in futures]
        return torch.cat(outputs, dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
