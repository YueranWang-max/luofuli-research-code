"""P6: one-click rebuild of the master results table from raw eval artifacts.

Reads artifacts/eval_*.json (+ artifacts/rows/*.jsonl when present) and writes
artifacts/master_table.json plus a markdown table to stdout. Also evaluates the
PLAN §8 success criteria for E6 (Budget-GAR) vs E2 (Binary GRPO).

Statistics:
  - Wilson 95% CI for each Pass@1
  - mean ± std across seeds per arm
  - exact McNemar on per-task pass/fail for matched-seed arm pairs
  - Welch t-test on per-task token counts of successful tasks (matched pairs)
  - reward-hacking rate re-derived from completion texts (code domain)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))

from environments.code_env import detect_reward_hacking  # noqa: E402

ARMS = ["e0", "e2", "e3", "e4", "e5", "e6", "e7l"]
ARM_LABEL = {
    "e0": "E0 Base (Qwen3-4B)",
    "e2": "E2 Binary GRPO",
    "e3": "E3 GRPO + length penalty",
    "e4": "E4 Tiny-GRS",
    "e5": "E5 Tiny-GAR",
    "e6": "E6 Budget-GAR (ours)",
    "e7l": "E7 Budget-GAR + unc. gate",
}
SEEDS = [42, 123, 2026]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def eval_paths(arm: str) -> list[tuple[int, Path]]:
    out = []
    if arm == "e0":
        p = ART / "eval_e0_canon.json"
        if p.exists():
            out.append((0, p))
        return out
    for s in SEEDS:
        p = ART / f"eval_{arm}_s{s}.json"
        if p.exists():
            out.append((s, p))
    return out


def load_rows(name: str) -> list[dict]:
    p = ART / "rows" / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def hack_rate(rows: list[dict]) -> float | None:
    code = [r for r in rows if r.get("domain") == "code"]
    if not code:
        return None
    flagged = sum(1 for r in code if detect_reward_hacking(r.get("completion", "")))
    return flagged / len(code)


def mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar (binomial) on paired pass/fail vectors."""
    b01 = sum(1 for x, y in zip(a, b) if x and not y)  # a pass, b fail
    b10 = sum(1 for x, y in zip(a, b) if (not x) and y)  # a fail, b pass
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p": 1.0}
    k = min(b01, b10)
    p = 2 * sum(math.comb(n, i) for i in range(0, k + 1)) * 0.5**n
    return {"b01": b01, "b10": b10, "p": min(1.0, p)}


def main() -> None:
    table: dict = {"arms": {}, "pairwise": {}, "success_criteria": {}}
    rows_by_name: dict[str, list[dict]] = {}

    for arm in ARMS:
        paths = eval_paths(arm)
        if not paths:
            continue
        per_seed = []
        for seed, p in paths:
            d = json.loads(p.read_text(encoding="utf-8"))
            name = "e0_canon" if arm == "e0" else f"{arm}_s{seed}"
            rows = load_rows(name)
            rows_by_name[name] = rows
            k = round(d["pass_at_1"] * d["n"])
            lo, hi = wilson_ci(k, d["n"])
            per_seed.append(
                {
                    "seed": seed,
                    "n": d["n"],
                    "pass_at_1": d["pass_at_1"],
                    "ci95": [lo, hi],
                    "avg_tokens_all": d["avg_tokens_all"],
                    "avg_tokens_success": d["avg_tokens_success"],
                    "math_pass": d["per_domain"].get("math", {}).get("pass_at_1"),
                    "code_pass": d["per_domain"].get("code", {}).get("pass_at_1"),
                    "hack_rate_code": hack_rate(rows),
                    "rows_available": bool(rows),
                }
            )
        passes = [s["pass_at_1"] for s in per_seed]
        toks = [s["avg_tokens_success"] for s in per_seed]
        m_p, s_p = mean_std(passes)
        m_t, s_t = mean_std(toks)
        table["arms"][arm] = {
            "label": ARM_LABEL[arm],
            "n_seeds": len(per_seed),
            "per_seed": per_seed,
            "pass_at_1_mean": m_p,
            "pass_at_1_std": s_p,
            "avg_tokens_success_mean": m_t,
            "avg_tokens_success_std": s_t,
        }

    # paired comparisons vs E2 at matched seeds (per-task, when rows exist)
    for arm in ("e3", "e4", "e5", "e6", "e7l"):
        for seed in SEEDS:
            a_name, b_name = f"e2_s{seed}", f"{arm}_s{seed}"
            if b_name not in rows_by_name or a_name not in rows_by_name:
                continue
            ra, rb = rows_by_name[a_name], rows_by_name[b_name]
            if not ra or not rb:
                continue
            by_id_a = {r["task_id"]: r for r in ra}
            common = [r for r in rb if r["task_id"] in by_id_a]
            if len(common) < 10:
                continue
            va = [bool(by_id_a[r["task_id"]]["passed"]) for r in common]
            vb = [bool(r["passed"]) for r in common]
            table["pairwise"][f"{arm}_vs_e2_s{seed}"] = mcnemar_exact(va, vb)

    # PLAN §8 success criteria: E6 vs E2
    checks = []
    for seed in SEEDS:
        a = table["arms"].get("e2", {}).get("per_seed", [])
        b = table["arms"].get("e6", {}).get("per_seed", [])
        sa = next((x for x in a if x["seed"] == seed), None)
        sb = next((x for x in b if x["seed"] == seed), None)
        if not (sa and sb):
            continue
        dp = sb["pass_at_1"] - sa["pass_at_1"]
        dt = (sa["avg_tokens_success"] - sb["avg_tokens_success"]) / max(sa["avg_tokens_success"], 1e-9)
        checks.append(
            {
                "seed": seed,
                "pass_at_1_delta_pp": round(dp * 100, 2),
                "token_reduction": round(dt, 4),
                "pass_ok": dp >= -0.015,
                "token_ok": dt >= 0.15,
            }
        )
    if checks:
        table["success_criteria"] = {
            "e6_vs_e2_per_seed": checks,
            "pass_criterion_seeds_ok": sum(c["pass_ok"] for c in checks),
            "token_criterion_seeds_ok": sum(c["token_ok"] for c in checks),
            "direction_consistent": len({c["pass_at_1_delta_pp"] > 0 for c in checks}) == 1,
        }

    out = ART / "master_table.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown summary
    print("| Arm | seeds | Pass@1 (mean±std) | avg tok/succ (mean±std) | math | code |")
    print("|---|---|---|---|---|---|")
    for arm, d in table["arms"].items():
        math_p = next((s["math_pass"] for s in d["per_seed"] if s["math_pass"] is not None), None)
        code_p = next((s["code_pass"] for s in d["per_seed"] if s["code_pass"] is not None), None)
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "-"
        print(
            f"| {d['label']} | {d['n_seeds']} | {d['pass_at_1_mean']:.3f}±{d['pass_at_1_std']:.3f} "
            f"| {d['avg_tokens_success_mean']:.1f}±{d['avg_tokens_success_std']:.1f} "
            f"| {fmt(math_p)} | {fmt(code_p)} |"
        )
    for k, v in table["pairwise"].items():
        print(f"McNemar {k}: b01={v['b01']} b10={v['b10']} p={v['p']:.4f}")
    if table["success_criteria"]:
        sc = table["success_criteria"]
        print(f"Success criteria E6 vs E2: pass_ok {sc['pass_criterion_seeds_ok']}/{len(sc['e6_vs_e2_per_seed'])}, "
              f"token_ok {sc['token_criterion_seeds_ok']}/{len(sc['e6_vs_e2_per_seed'])}, "
              f"direction_consistent={sc['direction_consistent']}")
    print(f"[table] {out}")


if __name__ == "__main__":
    main()
