# Threat Model

## Project

`Visual Firewall: Defending Multimodal Large Language Models Against Transformation-Robust Visual Prompt Injection`

## Working problem statement

Multimodal large language models and multimodal agents can be manipulated by
instructions hidden in images. The attack may use visible text, subtle overlays,
or optimized perturbations. In practice, these images are often transformed
before the target model sees them, for example by resizing, OCR, compression,
screen capture, or cropping. The main research question is:

Can we characterize which visual prompt injections remain effective after common
preprocessing transformations, and can a lightweight visual firewall block them
without materially hurting benign multimodal tasks?

## Target systems

Primary target:

- image-text MLLMs used for multimodal understanding
- optionally multimodal agents that can call tools or external services

Initial model scope:

- open-source MLLMs for reproducible experiments
- optional closed-source APIs later for transfer-only evaluation

Candidate first-wave models:

- `LLaVA-1.5-7B`
- `Qwen2-VL`
- `InternVL`

## Assets to protect

- refusal behavior on harmful requests
- tool-use integrity for multimodal agents
- confidentiality of system prompts, retrieved data, or connected tools
- reliability on benign VQA and instruction-following tasks

## Adversary goals

The attacker wants the model to do at least one of the following:

- follow malicious instructions embedded in the image
- ignore or override the system or developer policy
- answer harmful questions that should be refused
- execute unauthorized actions in an agent setting
- reveal hidden context or sensitive information

## Attacker capabilities

Assumed capabilities for the base setting:

- black-box access to the target model
- ability to submit one image and one text prompt
- knowledge that standard image preprocessing is likely to happen
- ability to optimize the attack against surrogate models
- no access to target weights, system prompt, or internal gradients

Optional stronger setting:

- limited query budget against the target API
- access to multiple surrogate models for transfer
- ability to tune attacks against an approximate preprocessing pipeline

Not assumed:

- control over target model weights
- direct compromise of the serving stack
- ability to disable safety filters manually

## Attack families in scope

### 1. Visible instruction overlays

The image contains readable malicious text or visually salient cues.

Why include it:

- strong baseline
- often survives transformations
- easy to audit

### 2. Semi-visible or bounded overlays

The attack hides instructions in small regions, blends them into the scene, or
uses typography and low-salience placement.

Why include it:

- closer to realistic attacks
- may evade casual human inspection while retaining robustness

### 3. Imperceptible or near-imperceptible perturbations

The attack optimizes image perturbations so the model receives malicious cues
that humans do not clearly notice.

Why include it:

- strongest research novelty on the attack side
- directly motivated by recent MLLM prompt injection work

### 4. Cross-modal attacks

The text prompt remains benign or weakly malicious while the image carries the
main malicious steering signal.

Why include it:

- closest to the target research question
- aligns with SIUO and MM-SafetyBench style failures

## Transformation channel

The attacker is evaluated before and after realistic transformations.

Mandatory transformations:

- resize down and up
- JPEG compression
- center crop and random crop
- screenshot or re-render
- Gaussian blur
- OCR preprocessing or text extraction

Optional later transformations:

- rotation
- color jitter
- watermarking
- aspect ratio padding
- thumbnail generation

## Defender assumptions

The defender deploys a `visual firewall` before the final MLLM call.

Allowed defense operations in phase 1:

- OCR extraction and text risk scoring
- image preprocessing randomization
- suspicious region masking or cropping
- multi-view consistency checks across transformed variants
- prompt-source attribution between user text and image text

Preferred defense constraints:

- no target model retraining in the first stage
- low latency overhead
- minimal assumptions about internal model access

Later-stage defenses may include:

- auxiliary vision encoders
- model-side refusal prompting
- lightweight fine-tuning or preference optimization

## Security game

Input:

- user text `t`
- image `x`

Attack:

- create adversarial image `x_adv`
- optionally keep `t` benign

System:

- apply preprocessing pipeline `P`
- apply visual firewall `F`
- pass result to target model `M`

Success condition:

- `M(F(P(x_adv)), t)` produces a harmful answer, policy violation, or
  unauthorized action

Defense success condition:

- attack success drops substantially after `F`
- benign task performance remains acceptable

## Evaluation metrics

Attack-side metrics:

- `ASR`: attack success rate
- transformation-robust ASR
- transfer ASR across models
- query efficiency for optimization-based attacks
- stealth score or human detectability proxy

Defense-side metrics:

- ASR reduction
- benign task accuracy retention
- over-refusal rate
- false positive rate on benign images with text
- latency overhead

## Initial benchmark plan

Primary safety benchmarks:

- `MM-SafetyBench`
- `SafeBench`
- `SIUO`

Benign utility checks:

- a small VQA or instruction-following validation split
- OCR-heavy but harmless examples to measure false positives

## Initial experiment matrix

### Phase 1

- reproduce one visible-text attack baseline
- evaluate robustness under resize, JPEG, crop, and screenshot
- implement one simple firewall: OCR plus risk scoring plus masking

### Phase 2

- add semi-visible and optimized perturbation attacks
- add multi-view consistency defense
- test transfer across at least three open-source MLLMs

### Phase 3

- evaluate on one agent-style setting
- compare training-free firewall against a model-side alignment baseline

## Main hypotheses

- visible and semi-visible attacks will remain much more transformation-robust
  than purely imperceptible attacks
- OCR-aware defenses will cut a large fraction of visible and bounded-overlay
  attacks, but will miss feature-level perturbation attacks
- multi-view consistency and prompt-source attribution will improve robustness
  with smaller benign utility loss than aggressive filtering alone

## Non-goals

- building the strongest possible offensive exploit for deployment
- bypassing real systems outside controlled evaluation
- claiming formal security guarantees

## Ethics

This project should be framed as defense-oriented red teaming. Attack artifacts
should be used only in controlled experiments, and any release should avoid
making plug-and-play exploitation easier than necessary.
