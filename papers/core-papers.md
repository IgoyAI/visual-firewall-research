# Core Papers

This file tracks the minimum reading set for the project:

`Visual Firewall: Defending Multimodal Large Language Models Against Transformation-Robust Visual Prompt Injection`

## Reading order

1. benchmark and problem framing
2. attack papers
3. defense and alignment papers
4. broader surveys

## Benchmark and framing

### MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models

- Authors: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao
- Date: November 29, 2023; revised June 19, 2024
- Link: https://arxiv.org/abs/2311.17600
- Why it matters:
  Establishes a safety benchmark for image-based manipulation of MLLMs with 13
  scenarios and 5,040 text-image pairs. This is one of the most directly useful
  starting points for evaluating image-driven jailbreak behavior.
- What to extract:
  threat model, attack construction pipeline, metrics, simple prompting defense

### SafeBench: A Safety Evaluation Framework for Multimodal Large Language Models

- Authors: Zonghao Ying, Aishan Liu, Siyuan Liang, Lei Huang, Jinyang Guo, Wenbo Zhou, Xianglong Liu, Dacheng Tao
- Date: October 24, 2024
- Link: https://arxiv.org/abs/2410.18927
- Why it matters:
  Expands evaluation breadth with 23 risk scenarios and 2,300 multimodal harmful
  query pairs, plus a stronger automated judging protocol.
- What to extract:
  risk taxonomy, automated evaluation setup, cross-model comparison protocol

### Safe Inputs but Unsafe Output: Benchmarking Cross-modality Safety Alignment of Large Vision-Language Model

- Authors: Siyin Wang, Xingsong Ye, Qinyuan Cheng, Junwen Duan, Shimin Li, Jinlan Fu, Xipeng Qiu, Xuanjing Huang
- Date: June 21, 2024; revised February 17, 2025
- Link: https://arxiv.org/abs/2406.15279
- Why it matters:
  Defines the `SIUO` setting, where each modality is harmless alone but unsafe
  when combined. This matches the project’s interest in cross-modal prompt
  injection rather than purely textual jailbreaks.
- What to extract:
  cross-modal composition failure modes, benchmark domains, evaluation protocol

## Attack papers

### Manipulating Multimodal Agents via Cross-Modal Prompt Injection

- Authors: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu
- Date: April 19, 2025; revised July 27, 2025
- Link: https://arxiv.org/abs/2504.14348
- Why it matters:
  Introduces `CrossInject`, a strong attack framework for multimodal agents that
  combines visual latent alignment and textual guidance enhancement.
- What to extract:
  attack assumptions, multimodal agent setting, transfer behavior, strongest
  baseline components

### Adversarial Prompt Injection Attack on Multimodal Large Language Models

- Authors: Meiwen Ding, Song Xia, Chenqi Kong, Xudong Jiang
- Date: March 31, 2026
- Link: https://arxiv.org/abs/2603.29418
- Why it matters:
  Studies imperceptible visual prompt injection against closed-source MLLMs.
  This is currently one of the clearest papers motivating the “transformation
  robustness versus stealth” tension.
- What to extract:
  perturbation construction, bounded overlay design, transfer evaluation, attack
  limits under preprocessing

## Defense and alignment papers

### mDPO: Conditional Preference Optimization for Multimodal Large Language Models

- Authors: Fei Wang, Wenxuan Zhou, James Y. Huang, Nan Xu, Sheng Zhang, Hoifung Poon, Muhao Chen
- Date: June 17, 2024; revised October 7, 2024
- Link: https://arxiv.org/abs/2406.11839
- Why it matters:
  Shows that multimodal alignment can ignore the image condition, then proposes
  conditional preference optimization to force the model to use visual evidence.
- What to extract:
  conditioning failure analysis, image-dependent preference construction, defense
  ideas for model-side grounding

### MM-SafetyBench defense prompt

- Source paper: `MM-SafetyBench`
- Link: https://arxiv.org/abs/2311.17600
- Why it matters:
  It is a simple but relevant training-free baseline. The firewall should beat
  or complement this kind of defense, not ignore it.
- What to extract:
  exact prompt strategy, strengths, limits, cost profile

## Broader surveys and adjacent work

### Survey of Adversarial Robustness in Multimodal Large Language Models

- Authors: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui
- Date: March 18, 2025
- Link: https://arxiv.org/abs/2503.13962
- Why it matters:
  Useful for organizing the literature into modality-specific and cross-modal
  attacks, datasets, and metrics.
- What to extract:
  taxonomy, benchmark landscape, missing defenses, open problems

### SneakyPrompt: Jailbreaking Text-to-image Generative Models

- Authors: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao
- Date: May 20, 2023; revised November 10, 2023
- Link: https://arxiv.org/abs/2305.12082
- Why it matters:
  Not an MLLM paper, but highly relevant for thinking about prompt filtering,
  adversarial prompt adaptation, and how safety filters fail under iterative
  attack.
- What to extract:
  black-box attack logic, query-efficient optimization, defense evaluation style

## Immediate synthesis questions

Use the reading to answer these before implementation:

1. Which attacks explicitly test robustness after resize, crop, or screenshot?
2. Which defenses are model-side versus pipeline-side?
3. Which benchmark best matches a visual firewall rather than training-time alignment?
4. What benign OCR-heavy examples are needed to measure false positives?
5. Where do current papers stop short of a deployable pre-model defense?

## First implementation takeaway

The initial codebase should support:

- visible-text attacks
- transformed-image evaluation
- OCR extraction
- a simple risk-based masking firewall
- benchmark adapters for `MM-SafetyBench`, `SafeBench`, and `SIUO`
