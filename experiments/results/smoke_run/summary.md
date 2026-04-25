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

## Per-transform

- `blur`: ASR proxy 1.0000, defended 0.0000, anchor retention 0.5000
- `clean`: ASR proxy 1.0000, defended 0.0000, anchor retention 1.0000
- `crop`: ASR proxy 0.5000, defended 0.0000, anchor retention 0.0000
- `jpeg`: ASR proxy 0.5000, defended 0.0000, anchor retention 1.0000
- `rerender`: ASR proxy 1.0000, defended 0.0000, anchor retention 1.0000
- `resize`: ASR proxy 1.0000, defended 0.0000, anchor retention 0.5000
