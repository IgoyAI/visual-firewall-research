# Proposal Outline

## Title

`Visual Firewall: Defending Multimodal Large Language Models Against Transformation-Robust Visual Prompt Injection`

## Motivation

Multimodal large language models can be manipulated by information embedded in
images. Recent work shows that image-based prompt injection and cross-modal
attacks can redirect model behavior, including in agent-like settings. However,
the field still lacks a practical understanding of which attacks remain effective
after realistic preprocessing operations such as resizing, compression, cropping,
and re-rendering. At the same time, deployable pre-model defenses remain weakly
studied compared with model-side alignment methods.

This project focuses on that deployment gap. Rather than asking only whether a
visual injection can work, it asks which injections survive a realistic input
pipeline and whether a lightweight `visual firewall` can stop them with limited
impact on normal multimodal use.

## Problem statement

Given an image-text input pair for an MLLM, study the robustness of visual prompt
injection under common image transformations and design a pre-model defense that
reduces attack success while preserving benign task performance.

## Research questions

1. Which classes of visual prompt injection remain effective after common image
   preprocessing transformations?
2. How much protection can a lightweight visual firewall provide without model
   retraining?
3. Which failure modes remain after OCR-based and consistency-based defenses?
4. How well do attacks and defenses transfer across open-source MLLM families?

## Hypotheses

1. Visible and semi-visible text-based attacks will be more transformation-robust
   than purely imperceptible perturbations.
2. An OCR-aware firewall will significantly reduce attacks that rely on explicit
   textual instructions in the image.
3. Multi-view consistency checks will catch some attacks missed by single-pass
   OCR filtering, with lower over-filtering than aggressive masking alone.

## Methodology

### Attack side

Build a benchmarkable attack suite with increasing difficulty:

- visible text overlay
- bounded or low-salience text overlay
- later, imperceptible perturbation baseline

Evaluate each attack before and after:

- resize
- JPEG compression
- crop
- screenshot-style re-render
- blur

### Defense side

Design a training-free `visual firewall` that runs before the target MLLM:

- OCR extraction
- text risk scoring
- suspicious region masking or blurring
- optional warning prompt injection to the model
- later, multi-view consistency across transformed copies

### Target models

Start with open-source MLLMs such as:

- `LLaVA-1.5-7B`
- `Qwen2-VL`
- `InternVL`

### Benchmarks

Primary:

- `MM-SafetyBench`

Secondary:

- `SafeBench`
- `SIUO`

## Evaluation

Primary metrics:

- attack success rate
- transformed attack success rate
- ASR reduction after defense
- refusal rate

Secondary metrics:

- benign utility retention
- false positive rate on harmless OCR-heavy images
- latency overhead
- transfer across model families

## Expected contributions

1. A transformation-aware evaluation protocol for visual prompt injection.
2. A practical baseline attack suite for measuring cross-modal robustness.
3. A lightweight visual firewall baseline for pre-model deployment.
4. An empirical analysis of the tradeoff between safety gain and benign utility.

## Timeline

### Stage 1

- finalize threat model
- collect core papers
- reproduce benchmark subset

### Stage 2

- implement visible-text attacks
- implement transformation pipeline
- run initial attacks on two open-source MLLMs

### Stage 3

- implement OCR-based firewall
- evaluate ASR reduction and benign retention
- analyze failure cases

### Stage 4

- add stronger attacks or consistency defense
- extend to second benchmark
- write report and thesis chapter draft

## Risks

- OCR performance may be too brittle on stylized text
- model setup cost may slow evaluation
- benchmark licensing or formatting may require adaptation work

## Ethics and scope

The project is defense-oriented red teaming. Attacks will be evaluated only in
controlled settings for measurement and defense development. The work does not
aim to operationalize real-world misuse.

## Starting references

- `MM-SafetyBench`: https://arxiv.org/abs/2311.17600
- `SafeBench`: https://arxiv.org/abs/2410.18927
- `SIUO`: https://arxiv.org/abs/2406.15279
- `CrossInject`: https://arxiv.org/abs/2504.14348
- `Adversarial Prompt Injection Attack on MLLMs`: https://arxiv.org/abs/2603.29418
- `mDPO`: https://arxiv.org/abs/2406.11839
- `Survey of Adversarial Robustness in MLLMs`: https://arxiv.org/abs/2503.13962
