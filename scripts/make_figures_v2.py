"""v2 figures from artifacts/master_table_v2.json -> artifacts/figures_v2/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "figures_v2"
FIG.mkdir(exist_ok=True)

ORDER = ["e0", "e2", "e3", "e5", "e6"]
LABEL = {"e0": "E0 Base", "e2": "E2 Binary", "e3": "E3 +len-pen", "e5": "E5 Tiny-GAR", "e6": "E6 Budget-GAR"}
COLORS = {"e0": "#9aa0a6", "e2": "#8ab4f8", "e3": "#fdd663", "e5": "#c58af9", "e6": "#34a853"}


def main() -> None:
    t = json.loads((ART / "master_table_v2.json").read_text(encoding="utf-8"))
    arms = [a for a in ORDER if a in t["arms"]]
    x = range(len(arms))
    passes = [t["arms"][a]["pass_mean"] for a in arms]
    pstd = [t["arms"][a]["pass_std"] for a in arms]

    # fig1: main comparison with seed scatter
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x, passes, yerr=pstd, capsize=4, color=[COLORS[a] for a in arms],
           alpha=0.85, edgecolor="black", linewidth=0.6)
    for a, xi in zip(arms, x):
        seeds = t["arms"][a]["per_seed"]
        ax.scatter([xi] * len(seeds), [s["pass_at_1"] for s in seeds],
                   color="black", s=14, zorder=3)
    ax.set_xticks(list(x), [LABEL[a] for a in arms], fontsize=9)
    ax.set_ylabel("Pass@1 · MATH-500 L4/L5 numeric (256-tok eval)")
    ax.set_title("Fair-budget comparison: reward designs vs base (dots = seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_v2_main.png", dpi=200)
    plt.close(fig)

    # fig2: L4/L5 strata
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    w = 0.38
    l4, l5 = [], []
    for a in arms:
        s = t["arms"][a]["per_seed"][0]
        l4.append(s.get("pass_L4", 0))
        l5.append(s.get("pass_L5", 0))
    ax.bar([i - w / 2 for i in x], l4, w, label="Level 4", color="#8ab4f8", edgecolor="black", linewidth=0.6)
    ax.bar([i + w / 2 for i in x], l5, w, label="Level 5", color="#f28b82", edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x), [LABEL[a] for a in arms], fontsize=9)
    ax.set_ylabel("Pass@1")
    ax.set_title("Difficulty strata (seed 42)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_v2_strata.png", dpi=200)
    plt.close(fig)

    # fig3: variance vs mean (the stability story)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for a in arms:
        d = t["arms"][a]
        ax.errorbar(d["pass_mean"], d["pass_std"], fmt="o", markersize=10,
                    color=COLORS[a], label=LABEL[a], ecolor="black", capsize=3)
    ax.set_xlabel("Pass@1 mean across seeds")
    ax.set_ylabel("Seed std (pp)")
    ax.set_title("Mean–variance trade-off across reward designs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_v2_meanvar.png", dpi=200)
    plt.close(fig)

    print(f"[figures_v2] {FIG}")


if __name__ == "__main__":
    main()
