"""Cross-Modal Consistency (CMC) firewall with conformal calibration.

The key insight: benign image text describes the image, while injected text
is semantically detached from the image content. We exploit this asymmetry
by combining two signals:

  1. Instruction match score: cosine similarity between OCR text and a
     reference bank of instruction-like phrases. (existing semantic signal)
  2. Image-text inconsistency score: clipped negative cosine similarity
     between OCR text and the image embedding. (NEW cross-modal signal)

The combined risk is r(t, x) = s_inst(t) + lambda * s_inc(t, x).

We then use inductive conformal prediction to calibrate the decision
threshold so that the marginal false positive rate is bounded by
alpha + 1/(n+1) on the clean calibration/deployment distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from src.defenses.ocr_firewall import (
    FirewallConfig,
    FirewallResult,
    OCRWord,
    _mask_suspicious_words,
    normalize_text,
    run_ocr,
)
from src.defenses.semantic_firewall import (
    DEFAULT_REFERENCE_TEXTS,
    _group_words_into_spans,
    _span_text,
    _resolve_dtype,
)


@dataclass(frozen=True)
class CMCFirewallConfig:
    """Configuration for the cross-modal consistency firewall."""

    siglip_path: str | Path = ""
    device: str = "cuda:1"
    dtype: str = "float16"
    # Component weights
    lambda_inconsistency: float = 0.5
    # Reference instruction bank
    reference_texts: tuple[str, ...] = DEFAULT_REFERENCE_TEXTS
    # OCR config
    ocr_config: FirewallConfig = field(default_factory=FirewallConfig)
    # Span filtering
    min_span_length: int = 2
    # Conformal calibration parameters
    target_fpr: float = 0.10  # Target false positive rate alpha
    # Manual threshold (used if calibration not run)
    threshold: float = 0.5


@dataclass
class CMCSpanScore:
    """Individual span score with both components."""
    text: str
    s_inst: float
    s_inc: float
    risk: float


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class CMCFirewall:
    """Cross-modal consistency firewall with conformal calibration.

    Risk score for OCR-extracted text span t in image x:

        r(t, x) = s_inst(t) + lambda * s_inc(t, x)

    where
        s_inst(t) = max_{r in R} cos(E_txt(t), E_txt(r))
        s_inc(t, x) = max(-cos(E_img(x), E_txt(t)), -0.1)

    Theoretical motivation: an injected instruction overlay is asymmetric:
    its text matches instruction patterns AND is decoupled from image
    content. A benign image caption like "LIBRARY EVENT SCHEDULE" might
    score moderately on s_inst but is image-consistent, so s_inc is low.

    Conformal calibration: given a calibration set of scores from clean
    images/views drawn from the deployment distribution, set threshold tau
    as the conformal upper quantile so that future clean inputs have
    marginal flag rate at most alpha + 1/(n+1).
    """

    def __init__(self, config: CMCFirewallConfig) -> None:
        self.config = config
        self.device = config.device
        self.dtype = _resolve_dtype(config.dtype)
        siglip_path = str(config.siglip_path)

        self.tokenizer = AutoTokenizer.from_pretrained(siglip_path, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            siglip_path, local_files_only=True, use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            siglip_path, local_files_only=True, dtype=self.dtype,
        ).to(self.device).eval()

        self.reference_embeddings = self._encode_reference_bank(config.reference_texts)
        self.threshold = config.threshold
        self._calibration_scores: list[float] | None = None

    # ---------- Embedding helpers ----------

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        # transformers >=5 wraps SigLIP outputs in BaseModelOutputWithPooling;
        # older versions returned the tensor directly. Handle both.
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        features = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def _image_text_logit(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute calibrated SigLIP image-text matching probability.

        SigLIP is trained with a sigmoid loss using a learned temperature
        scale and bias; the per-pair matching probability is
            sigmoid(temperature * cos(image, text) + bias).
        We return this probability directly, which has a larger dynamic
        range than raw cosine similarity for short text spans.
        """
        # image_emb: (1, d), text_emb: (n, d) both unit-normalized
        cos = image_emb @ text_emb.T  # (1, n)
        logit_scale = getattr(self.model, "logit_scale", None)
        logit_bias = getattr(self.model, "logit_bias", None)
        if logit_scale is not None and logit_bias is not None:
            scale = logit_scale.exp().float()
            bias = logit_bias.float()
            logits = cos.float() * scale + bias
        else:
            logits = cos.float() * 100.0  # fallback temperature
        prob = torch.sigmoid(logits).squeeze(0)  # (n,)
        return prob

    def _encode_reference_bank(self, texts: Sequence[str]) -> torch.Tensor:
        all_embeddings = []
        text_list = list(texts)
        for i in range(0, len(text_list), 16):
            all_embeddings.append(self._encode_texts(text_list[i:i + 16]))
        return torch.cat(all_embeddings, dim=0)

    # ---------- Scoring ----------

    def score_components(
        self, image: Image.Image, text_spans: list[str],
    ) -> tuple[list[float], list[float]]:
        """Compute (s_inst, s_inc) for each span.

        s_inst is the max cosine similarity between the text span and any
        reference instruction for unit-normalized embeddings.

        s_inc is the image-text inconsistency: negative of the cosine
        similarity between the text and the image embedding, clamped at
        a floor of -0.1. Injected text is semantically
        detached from the image and therefore receives a higher s_inc
        than benign image-describing text. SigLIP text-image cosines for
        short spans are small but the sign and magnitude differences
        between benign captions and external instructions are consistent.
        """
        if not text_spans:
            return [], []
        text_emb = self._encode_texts(text_spans)         # (n, d)
        image_emb = self._encode_image(image)             # (1, d)

        sims_to_bank = text_emb @ self.reference_embeddings.T  # (n, |R|)
        s_inst = sims_to_bank.max(dim=1).values

        # Raw image-text cosine, then negate so that well-aligned pairs
        # produce low s_inc and poorly-aligned pairs produce larger
        # s_inc. Typical well-aligned pairs have cos ~ 0.05, poorly-
        # aligned pairs have cos ~ -0.05, so s_inc is usually small.
        sims_to_image = (text_emb @ image_emb.T).squeeze(1)    # (n,)
        s_inc = (-sims_to_image).clamp(min=-0.1)               # range ~[-0.1, 0.1]

        return s_inst.cpu().tolist(), s_inc.cpu().tolist()

    def combined_risk(self, s_inst: float, s_inc: float) -> float:
        return s_inst + self.config.lambda_inconsistency * s_inc

    def score_image_spans(
        self, image: Image.Image, text_spans: list[str],
    ) -> list[CMCSpanScore]:
        """Return per-span scores with both components and combined risk."""
        s_inst, s_inc = self.score_components(image, text_spans)
        return [
            CMCSpanScore(
                text=t,
                s_inst=round(si, 4),
                s_inc=round(sc, 4),
                risk=round(self.combined_risk(si, sc), 4),
            )
            for t, si, sc in zip(text_spans, s_inst, s_inc)
        ]

    # ---------- Conformal calibration ----------

    def calibrate_conformal(
        self,
        calibration_images: Sequence[Image.Image],
        target_fpr: float | None = None,
    ) -> float:
        """Compute conformal threshold from clean calibration images/views.

        The threshold is the (1 - alpha)-empirical quantile of per-image
        max risk scores on clean inputs drawn from the intended deployment
        distribution. Under the conformal prediction guarantee, the marginal
        false positive rate on that same distribution is bounded above by
        alpha + 1/(n+1).
        """
        alpha = target_fpr if target_fpr is not None else self.config.target_fpr
        max_scores: list[float] = []
        for img in calibration_images:
            words = run_ocr(img, self.config.ocr_config)
            spans = _group_words_into_spans(words)
            text_spans = self._collect_candidate_spans(spans)
            if not text_spans:
                max_scores.append(0.0)
                continue
            s_inst, s_inc = self.score_components(img, text_spans)
            risks = [self.combined_risk(si, sc) for si, sc in zip(s_inst, s_inc)]
            max_scores.append(max(risks))

        self._calibration_scores = sorted(max_scores)
        n = len(max_scores)
        k = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
        k = max(0, min(n - 1, k))
        self.threshold = self._calibration_scores[k]
        return self.threshold

    @property
    def calibration_size(self) -> int:
        return 0 if self._calibration_scores is None else len(self._calibration_scores)

    @staticmethod
    def conformal_coverage_bound(n: int, alpha: float) -> float:
        """Upper bound on marginal false positive rate (Vovk et al. 2005)."""
        return alpha + 1.0 / (n + 1)

    # ---------- Span detection ----------

    def _collect_candidate_spans(self, line_spans: list[list[OCRWord]]) -> list[str]:
        """Collect text spans (single + sliding windows) above min length."""
        candidates: list[str] = []
        for span in line_spans:
            t = _span_text(span)
            if len(t.split()) >= self.config.min_span_length:
                candidates.append(t)
        for w in (2, 3):
            for i in range(len(line_spans) - w + 1):
                merged: list[OCRWord] = []
                for j in range(w):
                    merged.extend(line_spans[i + j])
                t = _span_text(merged)
                if len(t.split()) >= self.config.min_span_length:
                    candidates.append(t)
        return candidates

    # ---------- Main entry point ----------

    def apply(
        self,
        image: Image.Image,
        transform_registry: dict[str, Callable[[Image.Image], Image.Image]] | None = None,  # noqa: ARG002
    ) -> FirewallResult:
        """Run the CMC firewall on an image."""
        words = run_ocr(image, self.config.ocr_config)
        line_spans = _group_words_into_spans(words)
        if not line_spans:
            return FirewallResult(image=image, ocr_words=words, suspicious_words=[], suspicious_terms=[])

        # Build candidate spans (with sliding windows)
        candidate_word_sets: list[list[OCRWord]] = []
        for span in line_spans:
            if len(_span_text(span).split()) >= self.config.min_span_length:
                candidate_word_sets.append(span)
        for w in (2, 3):
            for i in range(len(line_spans) - w + 1):
                merged: list[OCRWord] = []
                for j in range(w):
                    merged.extend(line_spans[i + j])
                if len(_span_text(merged).split()) >= self.config.min_span_length:
                    candidate_word_sets.append(merged)

        if not candidate_word_sets:
            return FirewallResult(image=image, ocr_words=words, suspicious_words=[], suspicious_terms=[])

        candidate_texts = [_span_text(s) for s in candidate_word_sets]
        s_inst, s_inc = self.score_components(image, candidate_texts)
        risks = [self.combined_risk(si, sc) for si, sc in zip(s_inst, s_inc)]

        # Pick spans above threshold; for overlapping spans, keep highest risk
        word_best: dict[int, tuple[float, int]] = {}
        for idx, (span, risk) in enumerate(zip(candidate_word_sets, risks)):
            if risk < self.threshold:
                continue
            for word in span:
                wid = id(word)
                if wid not in word_best or risk > word_best[wid][0]:
                    word_best[wid] = (risk, idx)

        seen_indices: set[int] = set()
        suspicious_word_sets: list[list[OCRWord]] = []
        suspicious_terms_set: set[str] = set()
        for _wid, (_risk, idx) in word_best.items():
            if idx not in seen_indices:
                seen_indices.add(idx)
                suspicious_word_sets.append(candidate_word_sets[idx])
                term = normalize_text(candidate_texts[idx])
                if term:
                    suspicious_terms_set.add(term)

        suspicious_words: list[OCRWord] = []
        for span in suspicious_word_sets:
            suspicious_words.extend(span)

        filtered = _mask_suspicious_words(image, suspicious_words, pad=self.config.ocr_config.pad)
        return FirewallResult(
            image=filtered,
            ocr_words=words,
            suspicious_words=suspicious_words,
            suspicious_terms=sorted(suspicious_terms_set),
        )

    @torch.no_grad()
    def _encode_images_batch(
        self, images: list[Image.Image], batch_size: int = 16,
    ) -> torch.Tensor:
        """Encode many images in batches."""
        all_embs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            inputs = self.image_processor(images=chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
            features = self.model.get_image_features(pixel_values=pixel_values)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            all_embs.append(F.normalize(features.float(), dim=-1))
        return torch.cat(all_embs, dim=0)

    def apply_batch(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
    ) -> list[FirewallResult]:
        """Batched version of apply() across many images.

        Steps:
        1. Run OCR on every image.
        2. Build candidate spans per image.
        3. Batch-encode ALL spans across all images in one run of
           SigLIP text encoder.
        4. Batch-encode ALL images in one run of SigLIP image encoder.
        5. Compute combined risk per span using vectorized ops.
        6. Mask per image.
        """
        if not images:
            return []

        # Step 1: OCR
        per_image_words: list[List[OCRWord]] = []
        for img in images:
            per_image_words.append(run_ocr(img, self.config.ocr_config))

        # Step 2: candidate spans per image
        per_image_candidate_spans: list[list[list[OCRWord]]] = []
        per_image_candidate_texts: list[list[str]] = []
        for words in per_image_words:
            line_spans = _group_words_into_spans(words)
            cand_spans: list[list[OCRWord]] = []
            cand_texts: list[str] = []
            for span in line_spans:
                text = _span_text(span)
                if len(text.split()) >= self.config.min_span_length:
                    cand_spans.append(span)
                    cand_texts.append(text)
            for window in (2, 3):
                for i in range(len(line_spans) - window + 1):
                    merged: list[OCRWord] = []
                    for j in range(window):
                        merged.extend(line_spans[i + j])
                    text = _span_text(merged)
                    if len(text.split()) >= self.config.min_span_length:
                        cand_spans.append(merged)
                        cand_texts.append(text)
            per_image_candidate_spans.append(cand_spans)
            per_image_candidate_texts.append(cand_texts)

        # Step 3: batch-encode all images (for s_inc)
        image_embeddings = self._encode_images_batch(images, batch_size=batch_size)  # (N, d)

        # Step 4: batch-encode all candidate texts
        flat_texts: list[str] = []
        per_image_offsets: list[int] = [0]
        for texts in per_image_candidate_texts:
            flat_texts.extend(texts)
            per_image_offsets.append(len(flat_texts))

        # Build per-span image index (which image each flat-span belongs to)
        span_image_idx: list[int] = []
        for img_i, texts in enumerate(per_image_candidate_texts):
            span_image_idx.extend([img_i] * len(texts))

        flat_risks: list[float] = []
        if flat_texts:
            for i in range(0, len(flat_texts), batch_size):
                chunk_texts = flat_texts[i:i + batch_size]
                chunk_img_idx = span_image_idx[i:i + batch_size]
                text_emb = self._encode_texts(chunk_texts)               # (b, d)
                sims_to_bank = text_emb @ self.reference_embeddings.T    # (b, |R|)
                s_inst = sims_to_bank.max(dim=1).values                  # (b,)
                # Gather the right image embedding for each span
                img_idx_tensor = torch.tensor(
                    chunk_img_idx, device=self.device, dtype=torch.long,
                )
                matched_img_emb = image_embeddings[img_idx_tensor]       # (b, d)
                sims_to_image = (text_emb * matched_img_emb).sum(dim=1)  # (b,)
                s_inc = (-sims_to_image).clamp(min=-0.1)
                risk = s_inst + self.config.lambda_inconsistency * s_inc
                flat_risks.extend(risk.cpu().tolist())

        # Step 5: gather risks per image, pick suspicious spans, mask
        results: list[FirewallResult] = []
        for img_idx, (img, words, cand_spans, cand_texts) in enumerate(
            zip(images, per_image_words, per_image_candidate_spans, per_image_candidate_texts)
        ):
            if not cand_spans:
                results.append(FirewallResult(
                    image=img, ocr_words=words,
                    suspicious_words=[], suspicious_terms=[],
                ))
                continue
            lo, hi = per_image_offsets[img_idx], per_image_offsets[img_idx + 1]
            risks = flat_risks[lo:hi]

            word_best: dict[int, tuple[float, int]] = {}
            for idx, (span, risk) in enumerate(zip(cand_spans, risks)):
                if risk < self.threshold:
                    continue
                for word in span:
                    wid = id(word)
                    if wid not in word_best or risk > word_best[wid][0]:
                        word_best[wid] = (risk, idx)

            seen: set[int] = set()
            suspicious_spans: list[list[OCRWord]] = []
            for _wid, (_risk, idx) in word_best.items():
                if idx not in seen:
                    seen.add(idx)
                    suspicious_spans.append(cand_spans[idx])

            suspicious_words: list[OCRWord] = []
            for span in suspicious_spans:
                suspicious_words.extend(span)
            suspicious_terms = sorted({
                normalize_text(_span_text(span))
                for span in suspicious_spans
                if normalize_text(_span_text(span))
            })
            results.append(FirewallResult(
                image=_mask_suspicious_words(img, suspicious_words, pad=self.config.ocr_config.pad),
                ocr_words=words,
                suspicious_words=suspicious_words,
                suspicious_terms=suspicious_terms,
            ))
        return results
