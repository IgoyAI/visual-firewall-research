# Pilot Experiment Summary

This run is a synthetic OCR-proxy pilot for the visual firewall pipeline.

- Samples: 12
- Seed: 7
- Multi-view firewall: True
- Clean ASR proxy: 1.0000
- Defended clean ASR proxy: 0.0833
- Transformed ASR proxy: 0.9333
- Defended transformed ASR proxy: 0.1333
- Control false-positive rate: 0.0000
- Control benign retention after defense: 0.8194
- GPU attack similarity before defense: 0.8968
- GPU attack similarity after defense: 0.9167
- GPU control similarity after defense: 0.9700

## Per-transform

- `blur`: ASR proxy 0.9167, defended 0.2500, anchor retention 0.5833, GPU sim before 0.8984, after 0.9170
- `clean`: ASR proxy 1.0000, defended 0.0833, anchor retention 1.0000, GPU sim before 0.9142, after 0.9370
- `crop`: ASR proxy 0.9167, defended 0.0000, anchor retention 0.0000, GPU sim before 0.8359, after 0.8546
- `jpeg`: ASR proxy 0.9167, defended 0.0833, anchor retention 1.0000, GPU sim before 0.9037, after 0.9256
- `rerender`: ASR proxy 1.0000, defended 0.0833, anchor retention 1.0000, GPU sim before 0.9148, after 0.9344
- `resize`: ASR proxy 0.9167, defended 0.2500, anchor retention 0.5833, GPU sim before 0.9136, after 0.9319
