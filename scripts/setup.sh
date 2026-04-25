#!/bin/bash
# One-time environment + data + model setup for the visual-firewall codebase.
#
# Usage:
#   bash scripts/setup.sh             # full setup (env+data+models)
#   bash scripts/setup.sh --no-models # skip the 50+ GB model downloads
#
# Idempotent: skips what already exists.

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --no-models) SKIP_MODELS=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

# --- 1. Conda env --------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniforge / Miniconda first."; exit 1
fi
if ! conda env list | grep -q "^visual-firewall "; then
    step "Creating conda env 'visual-firewall'"
    conda env create -f environment.yml
else
    step "Conda env 'visual-firewall' already exists; updating pinned pip layer"
    conda run -n visual-firewall pip install -r requirements.txt
fi

# Activate inside this script for the remaining steps
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate visual-firewall

# --- 2. Dataset ---------------------------------------------------------------
step "Fetching MM-SafetyBench parquet"
python scripts/download_data.py

# --- 3. Models ----------------------------------------------------------------
if [ $SKIP_MODELS -eq 1 ]; then
    step "Skipping model downloads (--no-models). Run later with:"
    echo "    bash scripts/setup.sh"
    exit 0
fi

step "Downloading models from HuggingFace (≈ 60 GB total)"
# Headline target + non-degradation reference + cross-model targets + judge + scorer.
hf download llava-hf/llava-1.5-7b-hf
hf download Qwen/Qwen2-VL-7B-Instruct
hf download Qwen/Qwen3.5-9B
hf download google/gemma-4-E4B-it
hf download meta-llama/Llama-Guard-3-8B
hf download google/siglip-so400m-patch14-384

step "Setup complete."
echo
echo "Next steps:"
echo "  bash scripts/smoke_test.sh           # quick (~5 min) sanity check"
echo "  bash scripts/reproduce.sh            # full paper reproduction (~6 hours)"
