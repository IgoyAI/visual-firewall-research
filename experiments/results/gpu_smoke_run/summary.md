# Pilot Experiment Summary

This run is a synthetic OCR-proxy pilot for the visual firewall pipeline.

- Samples: 2
- Seed: 7
- Multi-view firewall: True
- Clean ASR proxy: 1.0000
- Defended clean ASR proxy: 0.0000
- Transformed ASR proxy: 0.8000
- Defended transformed ASR proxy: 0.0000
- Control false-positive rate: 0.0000
- Control benign retention after defense: 0.8333
- GPU attack similarity before defense: 0.8831
- GPU attack similarity after defense: 0.8942
- GPU control similarity after defense: 0.9715

## Per-transform

- `blur`: ASR proxy 1.0000, defended 0.0000, anchor retention 0.5000, GPU sim before 0.8770, after 0.8931
- `clean`: ASR proxy 1.0000, defended 0.0000, anchor retention 1.0000, GPU sim before 0.8976, after 0.9061
- `crop`: ASR proxy 0.5000, defended 0.0000, anchor retention 0.0000, GPU sim before 0.8464, after 0.8586
- `jpeg`: ASR proxy 0.5000, defended 0.0000, anchor retention 1.0000, GPU sim before 0.8842, after 0.8883
- `rerender`: ASR proxy 1.0000, defended 0.0000, anchor retention 1.0000, GPU sim before 0.9003, after 0.9075
- `resize`: ASR proxy 1.0000, defended 0.0000, anchor retention 0.5000, GPU sim before 0.8931, after 0.9116
