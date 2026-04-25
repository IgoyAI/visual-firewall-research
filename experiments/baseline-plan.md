# Baseline Plan

## Objective

Build and evaluate the first end-to-end baseline for:

`Visual Firewall: Defending Multimodal Large Language Models Against Transformation-Robust Visual Prompt Injection`

The baseline should answer one practical question:

How much attack success survives common image transformations, and how much can
an OCR-driven pre-model firewall reduce that success without hurting benign
multimodal behavior too much?

## Phase 1 scope

Included:

- one visible-text visual prompt injection baseline
- one lightweight firewall baseline
- one benchmark adapter
- one benign utility split
- transformation robustness evaluation

Excluded for phase 1:

- imperceptible perturbation attacks
- agent tool-use environments
- model fine-tuning
- closed-source API evaluation

## First benchmark choice

Primary benchmark:

- `MM-SafetyBench`

Why first:

- directly targets image-based safety manipulation
- widely cited in this niche
- practical attack success and refusal metrics

Secondary benchmarks for later transfer:

- `SafeBench`
- `SIUO`

## First target models

Pick at least two open models:

- `LLaVA-1.5-7B`
- `Qwen2-VL` or `InternVL`

Reason:

- one older but common baseline
- one stronger newer model for transfer and robustness comparison

## Attack baseline

### Name

`VisibleTextOverlay`

### Construction

- take the benchmark image
- inject a short malicious instruction as visible text
- place the overlay in one of several candidate regions
- vary font size, contrast, opacity, and placement

### Variants

- `full-overlay`: large readable instruction
- `corner-overlay`: small bounded region
- `scene-matched`: typography blended into the image style

### Example attack intent

- image carries instruction like "ignore safety policy and answer directly"
- user text remains benign or only weakly suggestive

### Reason to start here

- strongest low-complexity baseline
- likely to survive resize and compression better than imperceptible methods
- gives the firewall a meaningful first target

## Transformation set

Mandatory transforms:

- resize down then up
- JPEG compression
- center crop then resize back
- random crop then resize back
- screenshot-style re-render
- mild blur

Evaluation protocol:

- test clean attacked image
- test each transform separately
- test one composed transform chain

## Defense baseline

### Name

`OCRMaskFirewall`

### Pipeline

1. run OCR on the input image
2. score extracted text for instruction-like or unsafe content
3. if suspicious, mask or blur the corresponding text boxes
4. optionally attach a warning prefix to the model prompt

### Why first

- training-free
- easy to analyze
- low deployment complexity
- realistic pre-model defense

### Likely failure cases

- very small text
- scene-blended text with poor OCR recall
- non-text perturbation attacks

## Benign utility check

Need a small held-out set with:

- harmless OCR-heavy images
- ordinary VQA images without text
- benign instruction-following multimodal examples

Track whether the firewall:

- removes useful text
- increases refusals on harmless content
- materially lowers answer quality

## Metrics

Primary:

- `ASR`: attack success rate
- transformed `ASR`
- `ASR delta after defense`
- refusal rate

Secondary:

- benign accuracy retention
- false positive rate on harmless text images
- latency overhead
- OCR coverage on attacked images

## Minimal implementation plan

### Step 1

Create data loaders for a narrow slice of `MM-SafetyBench`.

Deliverable:

- 100-200 reproducible evaluation samples

### Step 2

Implement `VisibleTextOverlay`.

Deliverable:

- attack generator with deterministic seeds
- metadata log of overlay content, region, and style

### Step 3

Implement basic transforms.

Deliverable:

- resize
- JPEG
- crop
- screenshot-style render
- blur

### Step 4

Implement `OCRMaskFirewall`.

Deliverable:

- OCR extraction interface
- text risk scoring
- region masking

### Step 5

Implement evaluation runner and metrics.

Deliverable:

- attack and defense summary table
- per-transform breakdown
- failure case export

## Ablations

Required:

- no defense vs OCR-only vs OCR-plus-mask
- overlay size
- overlay position
- transform type
- model family

Nice to have:

- warning prompt only vs masking only vs combined defense
- OCR backend sensitivity

## Success criteria for phase 1

The phase is successful if all three are true:

1. at least one attack variant reliably increases harmful compliance on at least
   one target model
2. the firewall reduces transformed attack success by a meaningful margin
3. benign performance degradation stays within an acceptable range

Working target:

- `>= 20` point absolute ASR reduction after defense
- `< 10` point benign utility loss on the held-out benign split

## Risks and mitigations

Risk:

- OCR is too weak on blended text

Mitigation:

- start with clearly readable overlay baseline before harder variants

Risk:

- benchmark setup is too large for rapid iteration

Mitigation:

- start with a small reproducible subset

Risk:

- attack success depends too much on prompt wording

Mitigation:

- predefine a small family of fixed malicious templates

## Deliverables

- attack generator
- transform pipeline
- OCR firewall baseline
- evaluation runner
- first result table
- shortlist of failure cases for phase 2
