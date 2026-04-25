"""Entrypoint for the real non-synthetic experiment path."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.real_runner import main


if __name__ == "__main__":
    main()
