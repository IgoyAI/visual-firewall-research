"""Runnable pilot experiment for the visual firewall project.

This runner does not require a local MLLM. It measures an ASR proxy using OCR:
the attack is counted as successful if suspicious attack text survives the image
pipeline and remains visible to a naive OCR-consuming stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from src.attacks.visible_text_overlay import OverlayConfig, apply_visible_text_overlay
from src.defenses.ocr_firewall import (
    FirewallConfig,
    apply_ocr_firewall,
    extract_ocr_text,
    is_suspicious,
    normalize_text,
    run_tesseract_ocr,
)
from src.eval.datasets import build_synthetic_dataset
from src.eval.gpu_encoder import (
    DualGPUSiglipImageEncoder,
    GPUEncoderConfig,
    cosine_similarity,
    discover_local_siglip_snapshot,
)
from src.eval.metrics import summarize_records
from src.eval.reports import write_csv, write_json, write_summary_markdown
from src.eval.types import ExperimentConfig, ExperimentRecord
from src.transforms.presets import build_transform_registry


def anchor_detected(ocr_text: str, anchor_phrase: str) -> bool:
    normalized_text = normalize_text(ocr_text)
    anchor_tokens = [token for token in normalize_text(anchor_phrase).split() if token]
    if not anchor_tokens:
        return False
    matched = sum(1 for token in anchor_tokens if token in normalized_text)
    return matched >= max(2, len(anchor_tokens) - 1)


def suspicious_present(ocr_text: str) -> bool:
    normalized = normalize_text(ocr_text)
    return any(is_suspicious(token) for token in normalized.split()) or is_suspicious(normalized)


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synthetic visual firewall pilot experiment.")
    parser.add_argument("--samples", type=int, default=12, help="Number of synthetic samples to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic image generation.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/results/pilot_run"),
        help="Directory for metrics and sample outputs.",
    )
    parser.add_argument(
        "--disable-gpu-encoder",
        action="store_true",
        help="Disable the dual-GPU SigLIP image-embedding stage.",
    )
    return parser


def run_experiment(config: ExperimentConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = config.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_synthetic_dataset(config.samples, seed=config.seed)
    transforms = build_transform_registry()
    firewall_config = FirewallConfig(use_multiview=config.use_multiview_firewall)
    records: list[ExperimentRecord] = []
    encoder = None
    if config.use_gpu_encoder and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        snapshot = discover_local_siglip_snapshot()
        if snapshot is not None:
            encoder = DualGPUSiglipImageEncoder(GPUEncoderConfig(snapshot_path=snapshot))

    for sample_idx, sample in enumerate(dataset):
        attack_position = config.attack_positions[sample_idx % len(config.attack_positions)]
        attacked_image, attack_meta = apply_visible_text_overlay(
            sample.image,
            OverlayConfig(text=sample.attack_text, position=attack_position),
        )
        pending_rows: list[dict] = []

        for variant_name, source_image in (("control", sample.image), ("attack", attacked_image)):
            for transform_name in config.transforms:
                transform_fn = transforms[transform_name]
                transformed = transform_fn(source_image)
                before_words = run_tesseract_ocr(transformed)
                before_text = extract_ocr_text(before_words)
                firewall = apply_ocr_firewall(transformed, config=firewall_config, transform_registry=transforms)
                after_words = run_tesseract_ocr(firewall.image)
                after_text = extract_ocr_text(after_words)

                pending_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "variant": variant_name,
                        "transform": transform_name,
                        "attack_text": sample.attack_text,
                        "attack_detected_before": int(variant_name == "attack" and suspicious_present(before_text)),
                        "attack_detected_after": int(variant_name == "attack" and suspicious_present(after_text)),
                        "anchor_detected_before": int(anchor_detected(before_text, sample.anchor_phrase)),
                        "anchor_detected_after": int(anchor_detected(after_text, sample.anchor_phrase)),
                        "firewall_flagged": int(bool(firewall.suspicious_words)),
                        "num_suspicious_boxes": len(firewall.suspicious_words),
                        "before_ocr_text": before_text,
                        "after_ocr_text": after_text,
                        "attack_position": attack_meta["position"],
                        "transformed_image": transformed,
                        "defended_image": firewall.image,
                    }
                )

                if sample_idx < config.sample_preview_limit and transform_name in {"clean", "jpeg", "crop"} and variant_name == "attack":
                    stem = f"{sample.sample_id}_{transform_name}"
                    save_image(source_image, images_dir / f"{stem}_attacked.png")
                    save_image(transformed, images_dir / f"{stem}_transformed.png")
                    save_image(firewall.image, images_dir / f"{stem}_defended.png")

        if sample_idx < config.sample_preview_limit:
            save_image(sample.image, images_dir / f"{sample.sample_id}_base.png")

        if encoder is not None:
            image_batch = [sample.image]
            for row in pending_rows:
                image_batch.extend([row["transformed_image"], row["defended_image"]])
            embeddings = encoder.encode_images(image_batch)
            clean_embedding = embeddings[0]
            cursor = 1
            for row in pending_rows:
                transformed_embedding = embeddings[cursor]
                defended_embedding = embeddings[cursor + 1]
                row["gpu_similarity_before"] = cosine_similarity(clean_embedding, transformed_embedding)
                row["gpu_similarity_after"] = cosine_similarity(clean_embedding, defended_embedding)
                cursor += 2
        else:
            for row in pending_rows:
                row["gpu_similarity_before"] = None
                row["gpu_similarity_after"] = None

        for row in pending_rows:
            transformed_image = row.pop("transformed_image")
            defended_image = row.pop("defended_image")
            del transformed_image, defended_image
            records.append(ExperimentRecord(**row))

    summary = summarize_records(records)
    summary["notes"] = (
        "This is a synthetic OCR-proxy pilot, not a benchmarked MLLM evaluation. "
        "Attack success is measured by whether suspicious overlay text survives the image pipeline "
        "and remains visible to OCR before and after the firewall. "
        "A local SigLIP image encoder is also used across both GPUs to compute image-similarity diagnostics."
    )

    write_json(config.output_dir / "records.json", [record.to_dict() for record in records])
    write_csv(config.output_dir / "records.csv", records)
    write_json(config.output_dir / "summary.json", summary)
    write_summary_markdown(config.output_dir / "summary.md", config, summary)
    return summary


def main() -> None:
    args = build_argument_parser().parse_args()
    config = ExperimentConfig(
        samples=args.samples,
        seed=args.seed,
        output_dir=args.outdir,
        use_gpu_encoder=not args.disable_gpu_encoder,
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
