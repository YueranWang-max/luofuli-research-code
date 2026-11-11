"""P6: generate paper figures from artifacts/master_table.json.

Outputs to artifacts/figures/*.png (200 dpi):
  fig1_pass_at_1.png   - Pass@1 per arm with per-seed scatter + seed-std bars
  fig2_token_eff.png   - avg tokens per successful task per arm
  fig3_pareto.png      - Pass@1 vs token cost Pareto view
  fig4_domains.png     - math / code Pass@1 per arm
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["e0", "e2", "e3", "e4", "e5", "e6", "e7l"]
LABEL = {
    "e0": "E0 Base",
    "e2": "E2 Binary\nGRPO",
    "e3": "E3 +length\npenalty",
    "e4": "E4 Tiny-GRS",
    "e5": "E5 Tiny-GAR",
    "e6": "E6 Budget-GAR\n(ours)",
    "e7l": "E7 +unc.\ngate",
}
COLORS = {
    "e0": "#9aa0a6", "e2": "#8ab4f8", "e3": "#fdd663", "e4": "#f28b82",
    "e5": "#c58af9", "e6": "#34a853", "e7l": "#46bdc6",
}


def main() -> None:
    table = json.loads((ART / "master_table.json").read_text(encoding="utf-8"))
    arms = [a for a in ORDER if a in table["arms"]]
    if not arms:
        print("master_table.json has no arms yet")
        return
    x = range(len(arms))
    passes = [table["arms"][a]["pass_at_1_mean"] for a in arms]
    pstd = [table["arms"][a]["pass_at_1_std"] for a in arms]
    toks = [table["arms"][a]["avg_tokens_success_mean"] for a in arms]
    tstd = [table["arms"][a]["avg_tokens_success_std"] for a in arms]
    colors = [COLORS[a] for a in arms]

    # fig1: Pass@1
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x, passes, yerr=pstd, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    for a, xi in zip(arms, x):
        seeds = table["arms"][a]["per_seed"]
        ax.scatter([xi] * len(seeds), [s["pass_at_1"] for s in seeds],
                   color="black", s=12, zorder=3, label="_seed" if xi == 0 else "_seed")
    ax.set_xticks(list(x), [LABEL[a] for a in arms], fontsize=8)
    ax.set_ylabel("Pass@1 (test set)")
    ax.set_title("Verifiable pass rate by reward design (dots = seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_pass_at_1.png", dpi=200)
    plt.close(fig)

    # fig2: tokens per success
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x, toks, yerr=tstd, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x), [LABEL[a] for a in arms], fontsize=8)
    ax.set_ylabel("Avg tokens per successful task")
    ax.set_title("Answer efficiency on successful tasks (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_token_eff.png", dpi=200)
    plt.close(fig)

    # fig3: Pareto
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for a, xi in zip(arms, x):
        seeds = table["arms"][a]["per_seed"]
        ax.scatter([s["avg_tokens_success"] for s in seeds], [s["pass_at_1"] for s in seeds],
                   s=48, color=COLORS[a], label=LABEL[a].replace("\n", " "), edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Avg tokens per successful task")
    ax.set_ylabel("Pass@1")
    ax.set_title("Quality-efficiency Pareto view")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_pareto.png", dpi=200)
    plt.close(fig)

    # fig4: per-domain
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.38
    math_v, code_v = [], []
    for a in arms:
        seeds = table["arms"][a]["per_seed"]
        mv = [s["math_pass"] for s in seeds if s["math_pass"] is not None]
        cv = [s["code_pass"] for s in seeds if s["code_pass"] is not None]
        math_v.append(sum(mv) / len(mv) if mv else 0)
        code_v.append(sum(cv) / len(cv) if cv else 0)
    ax.bar([i - width / 2 for i in x], math_v, width, label="math", color="#8ab4f8", edgecolor="black", linewidth=0.6)
    ax.bar([i + width / 2 for i in x], code_v, width, label="code", color="#fdd663", edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x), [LABEL[a] for a in arms], fontsize=8)
    ax.set_ylabel("Pass@1")
    ax.set_title("Per-domain pass rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_domains.png", dpi=200)
    plt.close(fig)

    print(f"[figures] {FIG}")


if __name__ == "__main__":
    main()
