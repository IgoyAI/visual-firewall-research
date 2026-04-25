"""Generate ROC and precision-recall curves from the threshold sweep data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    data = json.loads(Path("experiments/results/threshold_sweep/threshold_sweep.json").read_text())
    rows = data["rows"]
    clean_scores = data["clean_scores"]
    attack_scores = data["attack_scores"]

    fprs = [r["fpr"] for r in rows]
    tprs = [r["recall"] for r in rows]
    precisions = [r["precision"] for r in rows]
    recalls = [r["recall"] for r in rows]
    thresholds = [r["threshold"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ROC
    ax = axes[0]
    ax.plot(fprs, tprs, "o-", color="#2b5ea0", linewidth=2, markersize=6)
    for r in rows:
        if r["threshold"] in (0.70, 0.78, 0.82):
            ax.annotate(
                f"$\\tau$={r['threshold']}",
                (r["fpr"], r["recall"]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=9,
            )
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Precision-Recall
    ax = axes[1]
    ax.plot(recalls, precisions, "s-", color="#c03030", linewidth=2, markersize=6)
    for r in rows:
        if r["threshold"] in (0.70, 0.78, 0.82):
            ax.annotate(
                f"$\\tau$={r['threshold']}",
                (r["recall"], r["precision"]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=9,
            )
    ax.set_xlabel("Recall (Attack Detection Rate)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Score distribution
    ax = axes[2]
    # Filter out the 0.0 scores (no text detected) for readable histogram
    clean_nonzero = [s for s in clean_scores if s > 0.01]
    attack_nonzero = [s for s in attack_scores if s > 0.01]
    bins = [i / 20 for i in range(11, 20)]
    ax.hist(clean_nonzero, bins=bins, color="#2b5ea0", alpha=0.6, label="Clean", density=True)
    ax.hist(attack_nonzero, bins=bins, color="#c03030", alpha=0.6, label="Attacked", density=True)
    ax.axvline(0.78, color="black", linestyle="--", linewidth=1.5, label=r"$\tau = 0.78$")
    ax.set_xlabel("Max Span Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution (clean vs. attacked)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_calibration.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    png_path = out_dir / "threshold_calibration.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {png_path}")
    print(f"Clean scores > 0: {len(clean_nonzero)}/{len(clean_scores)}")
    print(f"Attack scores > 0: {len(attack_nonzero)}/{len(attack_scores)}")


if __name__ == "__main__":
    main()
