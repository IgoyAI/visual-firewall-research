"""Model-backed inference for real multimodal experiments."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVisionGenerationModel
except ImportError:  # pragma: no cover - fallback for older transformers
    from transformers import AutoModelForVision2Seq as AutoVisionGenerationModel

from src.eval.real_types import BenchmarkSample


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[key]


@dataclass(frozen=True)
class ModelBackendConfig:
    model_path: str
    backend_type: str = "hf_generate"
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    local_files_only: bool = True
    use_two_gpus: bool = True


@dataclass(frozen=True)
class BackendPrediction:
    response: str
    metadata: dict[str, Any]


class MultimodalBackend(Protocol):
    def predict(self, image: Image.Image, sample: BenchmarkSample) -> BackendPrediction:
        """Run inference for one benchmark sample."""


class HuggingFaceMultimodalGenerator:
    """Generic Hugging Face multimodal generator with dual-GPU sharding."""

    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config
        self.dtype = _resolve_dtype(config.dtype)
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
            "device_map": "auto",
            "dtype": self.dtype,
        }
        if config.use_two_gpus and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            model_kwargs["max_memory"] = self._build_max_memory()
        self.model = AutoVisionGenerationModel.from_pretrained(config.model_path, **model_kwargs).eval()
        self.primary_device = self._infer_primary_device()

    def _build_max_memory(self) -> dict[int, str]:
        max_memory = {}
        for index in range(min(2, torch.cuda.device_count())):
            total_gib = torch.cuda.get_device_properties(index).total_memory // (1024**3)
            reserve = 6 if total_gib >= 32 else 2
            max_memory[index] = f"{max(1, total_gib - reserve)}GiB"
        return max_memory

    def _infer_primary_device(self) -> str:
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            for _, device in self.model.hf_device_map.items():
                if isinstance(device, str) and device.startswith("cuda"):
                    return device
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _build_text_prompt(self, prompt: str) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.primary_device)
            else:
                moved[key] = value
        return moved

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Single-image generation (kept for backward compatibility)."""
        return self.generate_batch([image], [prompt])[0]

    def generate_batch(
        self, images: list[Image.Image], prompts: list[str],
    ) -> list[str]:
        """Batched generation across multiple (image, prompt) pairs.

        Processes all inputs through the processor in a single call and
        runs a single model.generate() for the batch. Yields substantial
        throughput gains because the H100 is severely underutilized on
        batch-size-1 calls.
        """
        if not images:
            return []
        assert len(images) == len(prompts)
        text_prompts = [self._build_text_prompt(p) for p in prompts]
        try:
            inputs = self.processor(
                images=images, text=text_prompts,
                return_tensors="pt", padding=True,
            )
        except (ValueError, TypeError):
            # Gemma 4 (and some other newer processors) expect nested
            # list-of-lists where each inner list holds the images for one
            # conversation. Retry in that format.
            inputs = self.processor(
                images=[[img] for img in images], text=text_prompts,
                return_tensors="pt", padding=True,
            )
        inputs = self._move_inputs(inputs)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids")
        if input_ids is not None and output_ids.shape[1] > input_ids.shape[1]:
            generated_ids = output_ids[:, input_ids.shape[1]:]
        else:
            generated_ids = output_ids
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]

    def predict(self, image: Image.Image, sample: BenchmarkSample) -> BackendPrediction:
        return BackendPrediction(response=self.generate(image, sample.prompt), metadata={})

    def predict_batch(
        self, images: list[Image.Image], samples: list[BenchmarkSample],
    ) -> list[BackendPrediction]:
        assert len(images) == len(samples)
        prompts = [s.prompt for s in samples]
        responses = self.generate_batch(images, prompts)
        return [BackendPrediction(response=r, metadata={}) for r in responses]


def _parse_choice(choice: str) -> tuple[str | None, str]:
    stripped = choice.strip()
    match = re.match(r"^\s*([A-Z])[\.\):\-]?\s*(.*)$", stripped)
    if not match:
        return None, stripped
    label = match.group(1).upper()
    body = match.group(2).strip() or stripped
    return label, body


class DualGPUSiglipMultipleChoiceBackend:
    """Score multiple-choice options with local SigLIP across one or two GPUs."""

    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config
        self.dtype = _resolve_dtype(config.dtype)
        self.devices = self._discover_devices(config.use_two_gpus)
        if not self.devices:
            raise RuntimeError("SigLIP backend requires at least one available device")
        self.processors = {
            device: AutoProcessor.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code,
                local_files_only=config.local_files_only,
                use_fast=False,
            )
            for device in self.devices
        }
        model_dtype = self.dtype if any(device.startswith("cuda") for device in self.devices) else torch.float32
        self.models = {
            device: AutoModel.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code,
                local_files_only=config.local_files_only,
                dtype=model_dtype,
            ).to(device).eval()
            for device in self.devices
        }

    def _discover_devices(self, use_two_gpus: bool) -> tuple[str, ...]:
        if torch.cuda.is_available():
            if use_two_gpus and torch.cuda.device_count() >= 2:
                return ("cuda:0", "cuda:1")
            return ("cuda:0",)
        return ("cpu",)

    def _build_candidate_texts(self, sample: BenchmarkSample) -> tuple[list[str], list[str]]:
        if not sample.choices:
            raise ValueError("SigLIP MCQ backend requires multiple-choice options in the sample")
        question_text = str(sample.metadata.get("question_text") or sample.prompt)
        hint = str(sample.metadata.get("hint") or "").strip()
        labels: list[str] = []
        texts: list[str] = []
        for choice in sample.choices:
            label, body = _parse_choice(choice)
            labels.append(label or f"choice_{len(labels)}")
            lines = [f"Option {label or '?'}: {body}"]
            lines.append(f"Question: {question_text}")
            if hint:
                lines.append(f"Hint: {hint}")
            texts.append("\n".join(lines))
        return labels, texts

    def _score_texts(self, image: Image.Image, texts: list[str], device: str) -> list[float]:
        if not texts:
            return []
        processor = self.processors[device]
        inputs = processor(
            text=texts,
            images=image,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        moved = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                target_dtype = self.dtype if value.is_floating_point() and device.startswith("cuda") else value.dtype
                moved[key] = value.to(device=device, dtype=target_dtype)
            else:
                moved[key] = value
        with torch.inference_mode():
            outputs = self.models[device](**moved)
        return outputs.logits_per_image.squeeze(0).float().cpu().tolist()

    def predict(self, image: Image.Image, sample: BenchmarkSample) -> BackendPrediction:
        labels, texts = self._build_candidate_texts(sample)
        index_splits = [split.tolist() for split in torch.tensor_split(torch.arange(len(texts)), len(self.devices)) if len(split) > 0]
        chunks = [(split, [texts[index] for index in split], self.devices[chunk_index]) for chunk_index, split in enumerate(index_splits)]

        if len(chunks) == 1:
            split, chunk_texts, device = chunks[0]
            chunk_scores = self._score_texts(image, chunk_texts, device)
            scores = [0.0] * len(texts)
            for local_index, score in zip(split, chunk_scores):
                scores[local_index] = score
        else:
            scores = [0.0] * len(texts)
            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                futures = [
                    executor.submit(self._score_texts, image, chunk_texts, device)
                    for _, chunk_texts, device in chunks
                ]
            for (split, _, _), future in zip(chunks, futures):
                chunk_scores = future.result()
                for local_index, score in zip(split, chunk_scores):
                    scores[local_index] = score

        best_index = max(range(len(scores)), key=lambda index: scores[index])
        choice_scores = {label: round(float(score), 4) for label, score in zip(labels, scores)}
        return BackendPrediction(
            response=labels[best_index],
            metadata={
                "backend_type": "siglip_mcq",
                "choice_scores": choice_scores,
                "backend_devices": list(self.devices),
            },
        )

    def predict_batch(
        self, images: list[Image.Image], samples: list[BenchmarkSample],
    ) -> list[BackendPrediction]:
        # SigLIP MCQ scores per (image, candidate-text) and is not naturally
        # batched across images here; loop over predict() to satisfy the
        # runner's batched interface.
        assert len(images) == len(samples)
        return [self.predict(img, s) for img, s in zip(images, samples)]


def build_model_backend(config: ModelBackendConfig) -> MultimodalBackend:
    if config.backend_type == "hf_generate":
        return HuggingFaceMultimodalGenerator(config)
    if config.backend_type == "siglip_mcq":
        return DualGPUSiglipMultipleChoiceBackend(config)
    raise ValueError(f"Unsupported backend type: {config.backend_type}")
