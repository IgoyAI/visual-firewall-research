"""Tesseract-backed OCR firewall used for the pilot experiment."""

from __future__ import annotations

import csv
import importlib.util
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw


SUSPICIOUS_PHRASES = (
    "ignore",
    "override",
    "reveal",
    "secret",
    "unsafe",
    "harmful",
    "refuse",
    "bypass",
    "policy",
)


def normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass(frozen=True)
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


@dataclass
class FirewallResult:
    image: Image.Image
    ocr_words: List[OCRWord]
    suspicious_words: List[OCRWord]
    suspicious_terms: List[str]


@dataclass(frozen=True)
class FirewallConfig:
    pad: int = 8
    psm: int = 6
    ocr_backend: str = "tesseract"
    easyocr_gpu: bool = True
    easyocr_languages: Sequence[str] = ("en",)
    easyocr_cuda_device: int = 0
    use_multiview: bool = False
    multiview_transforms: Sequence[str] = ("resize", "jpeg", "crop")
    consensus_threshold: int = 1


_EASYOCR_READERS: dict[tuple[tuple[str, ...], bool, int], Any] = {}


def _easyocr_available() -> bool:
    return importlib.util.find_spec("easyocr") is not None


def _with_selected_cuda_device(device_index: int, fn: Callable[[], Any]) -> Any:
    import torch

    if not torch.cuda.is_available():
        return fn()
    previous = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    try:
        return fn()
    finally:
        torch.cuda.set_device(previous)


def _get_easyocr_reader(
    languages: Sequence[str],
    gpu: bool,
    cuda_device: int,
):
    import torch
    import easyocr

    normalized_languages = tuple(str(language) for language in languages)
    use_gpu = bool(gpu and torch.cuda.is_available())
    key = (normalized_languages, use_gpu, int(cuda_device))
    reader = _EASYOCR_READERS.get(key)
    if reader is not None:
        return reader

    def _build_reader():
        return easyocr.Reader(list(normalized_languages), gpu=use_gpu, verbose=False)

    reader = _with_selected_cuda_device(int(cuda_device), _build_reader) if use_gpu else _build_reader()
    _EASYOCR_READERS[key] = reader
    return reader


def run_tesseract_ocr(image: Image.Image, psm: int = 6) -> List[OCRWord]:
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "ocr_input.png"
        image.save(image_path)
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", str(psm), "tsv"],
            check=True,
            capture_output=True,
            text=True,
        )
    lines = result.stdout.splitlines()
    if not lines:
        return []
    reader = csv.DictReader(lines, delimiter="\t")
    words: List[OCRWord] = []
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if row.get("level") != "5":
            continue
        try:
            conf = float(row.get("conf", "-1"))
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
        except (TypeError, ValueError, KeyError):
            continue
        if width <= 0 or height <= 0:
            continue
        words.append(OCRWord(text=text, left=left, top=top, width=width, height=height, conf=conf))
    return words


def run_easyocr_ocr(
    image: Image.Image,
    languages: Sequence[str] = ("en",),
    gpu: bool = True,
    cuda_device: int = 0,
) -> List[OCRWord]:
    reader = _get_easyocr_reader(languages=languages, gpu=gpu, cuda_device=cuda_device)
    image_array = np.array(image)

    def _read_text():
        return reader.readtext(image_array, detail=1, paragraph=False)

    raw_results = _with_selected_cuda_device(int(cuda_device), _read_text) if gpu else _read_text()
    words: List[OCRWord] = []
    for result in raw_results:
        if not isinstance(result, (list, tuple)) or len(result) < 3:
            continue
        polygon, text, conf = result[0], str(result[1]).strip(), float(result[2])
        if not text:
            continue
        try:
            xs = [int(point[0]) for point in polygon]
            ys = [int(point[1]) for point in polygon]
        except (TypeError, ValueError, IndexError):
            continue
        if not xs or not ys:
            continue
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)
        width = max(0, right - left)
        height = max(0, bottom - top)
        if width <= 0 or height <= 0:
            continue
        words.append(
            OCRWord(
                text=text,
                left=left,
                top=top,
                width=width,
                height=height,
                conf=conf,
            )
        )
    return words


def run_ocr(image: Image.Image, config: FirewallConfig) -> List[OCRWord]:
    backend = config.ocr_backend.lower()
    if backend == "auto":
        backend = "easyocr" if _easyocr_available() else "tesseract"
    if backend == "easyocr":
        if not _easyocr_available():
            raise RuntimeError("EasyOCR backend requested but easyocr is not installed")
        return run_easyocr_ocr(
            image,
            languages=config.easyocr_languages,
            gpu=config.easyocr_gpu,
            cuda_device=config.easyocr_cuda_device,
        )
    if backend == "tesseract":
        return run_tesseract_ocr(image, psm=config.psm)
    raise ValueError(f"Unsupported OCR backend: {config.ocr_backend}")


def is_suspicious(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(phrase in normalized for phrase in SUSPICIOUS_PHRASES)


def extract_ocr_text(words: Iterable[OCRWord]) -> str:
    return " ".join(word.text for word in words)


def _detect_suspicious_terms(words: Iterable[OCRWord]) -> List[str]:
    terms = {normalize_text(word.text) for word in words if is_suspicious(word.text)}
    return sorted(term for term in terms if term)


def _mask_suspicious_words(image: Image.Image, suspicious: Sequence[OCRWord], pad: int) -> Image.Image:
    filtered = image.copy()
    if suspicious:
        draw = ImageDraw.Draw(filtered)
        for word in suspicious:
            left, top, right, bottom = word.bbox
            draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=(255, 255, 255))
    return filtered


def apply_ocr_firewall(
    image: Image.Image,
    config: FirewallConfig | None = None,
    transform_registry: dict[str, Callable[[Image.Image], Image.Image]] | None = None,
) -> FirewallResult:
    config = config or FirewallConfig()
    words = run_ocr(image, config)
    suspicious_terms = set(_detect_suspicious_terms(words))

    if config.use_multiview and transform_registry:
        counts: dict[str, int] = {}
        for transform_name in config.multiview_transforms:
            transform_fn = transform_registry.get(transform_name)
            if transform_fn is None:
                continue
            view_words = run_ocr(transform_fn(image), config)
            for term in _detect_suspicious_terms(view_words):
                counts[term] = counts.get(term, 0) + 1
        suspicious_terms.update({term for term, count in counts.items() if count >= config.consensus_threshold})

    suspicious = [word for word in words if normalize_text(word.text) in suspicious_terms or is_suspicious(word.text)]
    filtered = _mask_suspicious_words(image, suspicious, pad=config.pad)
    return FirewallResult(
        image=filtered,
        ocr_words=words,
        suspicious_words=suspicious,
        suspicious_terms=sorted(suspicious_terms),
    )
