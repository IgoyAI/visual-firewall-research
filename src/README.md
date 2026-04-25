# Source

Research code lives here.

## Current layout

- `attacks/`: attack generation and wrappers
- `defenses/`: visual firewall components
- `transforms/`: preprocessing and robustness transforms
- `eval/`: dataset adapters, metrics, and experiment runners

## First milestone

Implement the smallest end-to-end pipeline:

1. load a benchmark sample
2. apply a visible-text attack
3. run transformation variants
4. apply an OCR-based firewall
5. score attack success and benign retention
