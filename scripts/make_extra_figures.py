"""Produce the four additional paper figures:

1. paper/figures/alpha_dial.pdf       — α-dial tradeoff (clean, attack, unsafe)
2. paper/figures/lambda_sweep.pdf     — λ ablation at fixed α=0.10 (+ vMF point)
3. paper/figures/conformal_bootstrap.pdf — B=2000 FPR histograms vs bound
4. paper/figures/category_breakdown.pdf  — per-category unsafe reduction

All figures are generated from files already on disk (experiments/results/...).
No GPU work.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_summary(name: str) -> dict:
    return json.loads((RES / name / "summary.json").read_text())


# -------- Figure 1: α-dial tradeoff --------------------------------------

def fig_alpha_dial() -> Path:
    alphas = [0.05, 0.10, 0.20]
    runs = [load_summary(n) for n in (
        "mmsafety_llava_cmc_alpha05",
        "mmsafety_llava_cmc_full",
        "mmsafety_llava_cmc_alpha20",
    )]
    clean_flag = [r["variants"]["defended_control"]["firewall_flag_rate"] for r in runs]
    attack_flag = [r["variants"]["defended_attack"]["firewall_flag_rate"] for r in runs]
    unsafe_defended = [r["variants"]["defended_attack"]["unsafe_rate"] for r in runs]
    bound = [a + 1.0 / 101 for a in alphas]
    attacked_baseline = 0.1958

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    ax.plot(alphas, clean_flag, "o-", color="#2b5ea0", linewidth=2.2, markersize=8, label="Empirical clean flag")
    ax.plot(alphas, bound, "s--", color="#c03030", linewidth=1.8, markersize=7, label=r"Bound $\alpha + \frac{1}{n+1}$")
    ax.plot(alphas, alphas, ":", color="gray", linewidth=1.4, label=r"Target $\alpha$")
    ax.set_xlabel(r"Target FPR $\alpha$")
    ax.set_ylabel("Clean-image flag rate")
    ax.set_title("Conformal bound holds across the dial")
    ax.set_xticks(alphas)
    ax.set_ylim(0, 0.26)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.plot(alphas, attack_flag, "o-", color="#1f8a4c", linewidth=2.2, markersize=8)
    for a, f in zip(alphas, attack_flag):
        ax.annotate(f"{f*100:.1f}%", (a, f), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xlabel(r"Target FPR $\alpha$")
    ax.set_ylabel("Attack flag rate")
    ax.set_title("Attack detection grows with budget")
    ax.set_xticks(alphas)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.axhline(attacked_baseline, color="#c03030", linestyle="--", linewidth=1.6,
               label=f"Attacked ({attacked_baseline*100:.1f}%)")
    ax.axhline(0.15, color="gray", linestyle=":", linewidth=1.3,
               label="Clean (15.0%)")
    ax.plot(alphas, unsafe_defended, "o-", color="#2b5ea0", linewidth=2.2, markersize=8, label="Defended unsafe")
    for a, u in zip(alphas, unsafe_defended):
        ax.annotate(f"{u*100:.1f}%", (a, u), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xlabel(r"Target FPR $\alpha$")
    ax.set_ylabel("Unsafe response rate")
    ax.set_title(r"Unsafe rate closes at $\alpha=0.20$")
    ax.set_xticks(alphas)
    ax.set_ylim(0.14, 0.22)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG / "alpha_dial.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(FIG / "alpha_dial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# -------- Figure 2: λ sweep ----------------------------------------------

def fig_lambda_sweep() -> Path:
    # All at α=0.10 except the vMF-implied λ=0.03 point at α=0.20.
    alpha10 = [
        ("mmsafety_llava_cmc_lambda025", 0.25),
        ("mmsafety_llava_cmc_full",      0.50),
        ("mmsafety_llava_cmc_lambda100", 1.00),
    ]
    lambdas_10 = [lam for _, lam in alpha10]
    runs_10 = [load_summary(n) for n, _ in alpha10]
    attack_10 = [r["variants"]["defended_attack"]["firewall_flag_rate"] for r in runs_10]
    clean_10 = [r["variants"]["defended_control"]["firewall_flag_rate"] for r in runs_10]
    unsafe_10 = [r["variants"]["defended_attack"]["unsafe_rate"] for r in runs_10]

    vmf = load_summary("mmsafety_llava_cmc_lambda003")
    vmf_attack = vmf["variants"]["defended_attack"]["firewall_flag_rate"]
    vmf_clean = vmf["variants"]["defended_control"]["firewall_flag_rate"]
    vmf_unsafe = vmf["variants"]["defended_attack"]["unsafe_rate"]

    # alpha20 baseline for the vMF point comparison
    a20 = load_summary("mmsafety_llava_cmc_alpha20")
    a20_attack = a20["variants"]["defended_attack"]["firewall_flag_rate"]
    a20_unsafe = a20["variants"]["defended_attack"]["unsafe_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.plot(lambdas_10, attack_10, "o-", color="#1f8a4c", linewidth=2.2, markersize=8, label=r"Attack flag (@$\alpha$=0.10)")
    ax.plot(lambdas_10, clean_10, "o-", color="#2b5ea0", linewidth=2.2, markersize=8, label=r"Clean flag (@$\alpha$=0.10)")
    ax.scatter([0.03], [vmf_attack], marker="*", s=260, color="#c03030", zorder=5,
               label=r"$\lambda{=}0.03$ (vMF, @$\alpha$=0.20)")
    ax.scatter([0.03], [vmf_clean], marker="*", s=260, color="#2b5ea0", zorder=5, edgecolors="#c03030", linewidths=1.4)
    ax.scatter([0.5], [a20_attack], marker="D", s=90, color="#1f8a4c", zorder=4, alpha=0.6,
               label=r"$\lambda{=}0.5$ (@$\alpha$=0.20)")
    ax.annotate(f"{vmf_attack*100:.1f}%", (0.03, vmf_attack), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.annotate(f"{a20_attack*100:.1f}%", (0.5, a20_attack), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"Inconsistency weight $\lambda$")
    ax.set_ylabel("Flag rate")
    ax.set_title(r"Flag rates vs $\lambda$")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="center left", fontsize=8)

    ax = axes[1]
    ax.axhline(0.1958, color="#c03030", linestyle="--", linewidth=1.6, label="Attacked (19.6%)")
    ax.axhline(0.15, color="gray", linestyle=":", linewidth=1.3, label="Clean (15.0%)")
    ax.plot(lambdas_10, unsafe_10, "o-", color="#2b5ea0", linewidth=2.2, markersize=8, label=r"Defended (@$\alpha$=0.10)")
    ax.scatter([0.03], [vmf_unsafe], marker="*", s=260, color="#c03030", zorder=5,
               label=r"$\lambda{=}0.03$ (@$\alpha$=0.20)")
    ax.scatter([0.5], [a20_unsafe], marker="D", s=90, color="#2b5ea0", zorder=4, alpha=0.6,
               label=r"$\lambda{=}0.5$ (@$\alpha$=0.20)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Inconsistency weight $\lambda$")
    ax.set_ylabel("Unsafe response rate")
    ax.set_title(r"Defended unsafe rate vs $\lambda$")
    ax.set_ylim(0.13, 0.22)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out = FIG / "lambda_sweep.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(FIG / "lambda_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# -------- Figure 3: bootstrap FPR histograms ------------------------------

def fig_conformal_bootstrap() -> Path:
    data = json.loads((RES / "conformal_verification" / "results.json").read_text())
    pool = np.asarray(list(data["cal_risks"]) + list(data["test_risks"]), dtype=float)
    n_cal, n_test = 100, 200
    B = 2000
    rng = np.random.default_rng(7)
    alphas = [0.05, 0.10, 0.20]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, alpha in zip(axes, alphas):
        fprs = np.empty(B)
        for b in range(B):
            perm = rng.permutation(len(pool))
            cal = pool[perm[:n_cal]]
            test = pool[perm[n_cal:n_cal + n_test]]
            k = int(math.ceil((n_cal + 1) * (1.0 - alpha))) - 1
            k = max(0, min(n_cal - 1, k))
            tau = np.sort(cal)[k]
            fprs[b] = (test >= tau).mean()
        mean_fpr = fprs.mean()
        lo, hi = np.quantile(fprs, [0.025, 0.975])
        bound = alpha + 1.0 / (n_cal + 1)

        ax.hist(fprs, bins=25, color="#2b5ea0", alpha=0.7, edgecolor="white")
        ax.axvline(alpha, color="gray", linestyle=":", linewidth=1.5, label=rf"Target $\alpha={alpha}$")
        ax.axvline(bound, color="#c03030", linestyle="--", linewidth=1.8, label=rf"Bound $={bound:.3f}$")
        ax.axvline(mean_fpr, color="#1f8a4c", linestyle="-", linewidth=1.8, label=rf"Mean $={mean_fpr:.3f}$")
        ax.axvspan(lo, hi, color="#1f8a4c", alpha=0.12, label=f"95% CI [{lo:.2f}, {hi:.2f}]")
        ax.set_title(rf"$\alpha={alpha}$")
        ax.set_xlabel("Empirical FPR (single re-split)")
        if ax is axes[0]:
            ax.set_ylabel(f"Count (B={B} re-splits)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Bootstrap re-split verification of the conformal FPR bound", y=1.02, fontsize=12)
    plt.tight_layout()
    out = FIG / "conformal_bootstrap.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(FIG / "conformal_bootstrap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# -------- Figure 4: per-category unsafe reduction -------------------------

def fig_category_breakdown() -> Path:
    # Use λ=0.03 / α=0.20 as the paper's main operating point (best Pareto).
    s = load_summary("mmsafety_llava_cmc_lambda003")
    cats = s["categories"]
    # Sort by absolute reduction (attacked - defended), skipping zero-count.
    rows = []
    for name, v in cats.items():
        au = v.get("attacked_unsafe")
        du = v.get("defended_unsafe")
        n = v.get("count") or 0
        if au is None or du is None or n == 0:
            continue
        rows.append((name.replace("_", " "), au, du, au - du, n))
    rows.sort(key=lambda r: r[3])  # ascending; largest reduction (most negative sign) at bottom

    labels = [r[0] for r in rows]
    attacked = np.array([r[1] for r in rows])
    defended = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    y = np.arange(len(rows))
    h = 0.38
    ax.barh(y - h / 2, attacked * 100, height=h, color="#c03030", alpha=0.85, label="Attacked")
    ax.barh(y + h / 2, defended * 100, height=h, color="#2b5ea0", alpha=0.85, label="Defended (CMC, λ=0.03, α=0.20)")

    for i, (_, au, du, delta, n) in enumerate(rows):
        pp = (au - du) * 100
        txt = f"{pp:+.1f} pp" if pp != 0 else ""
        color = "#1f8a4c" if pp > 0 else ("#c03030" if pp < 0 else "gray")
        x = max(au, du) * 100 + 1.2
        ax.text(x, i, txt, va="center", fontsize=9, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Unsafe response rate (%)")
    ax.set_title("Per-category unsafe reduction (MM-SafetyBench, LLaVA-1.5-7B)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(attacked.max(), defended.max()) * 100 + 12)
    plt.tight_layout()
    out = FIG / "category_breakdown.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(FIG / "category_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    for fn in (fig_alpha_dial, fig_lambda_sweep, fig_conformal_bootstrap, fig_category_breakdown):
        out = fn()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
