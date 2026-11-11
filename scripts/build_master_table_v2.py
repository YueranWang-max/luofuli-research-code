"""v2 master table: real-benchmark results with bootstrap CIs and mechanisms.

Reads artifacts/bench_eval_{bench}_{arm}_s{seed}.json (+ bench_rows/, bench_reward_stats/)
and writes artifacts/master_table_v2.json with:
  - per-arm Pass@1 mean±std across seeds, bootstrap 95% CI
  - paired McNemar vs E2 at matched seeds
  - difficulty-stratified (L4 vs L5) pass rates
  - token efficiency on successes
  - reward-pipeline activation rates from instrumented runs
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

BENCH = "mathhard"       # eval naming uses train bench prefix from run_matrix_v2
EVAL_KEY = "math500"     # rows carry task ids from the math500 test set
ARMS = ["e0", "e2", "e3", "e5", "e6"]
ARM_LABEL = {
    "e0": "E0 Base",
    "e2": "E2 Binary GRPO",
    "e3": "E3 +length penalty",
    "e5": "E5 Tiny-GAR",
    "e6": "E6 Budget-GAR (ours)",
}
SEEDS = [42, 123, 2026]


def load_rows(name: str) -> list[dict]:
    p = ART / "bench_rows" / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def bootstrap_ci(passes: list[bool], n_boot: int = 10000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(passes)
    if n == 0:
        return (0.0, 0.0)
    means = []
    vals = [1.0 if p else 0.0 for p in passes]
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def paired_bootstrap_diff(a: list[bool], b: list[bool], n_boot: int = 10000, seed: int = 11) -> dict:
    """P(diff<=0) and CI for pass-rate difference a-b on the same tasks."""
    rng = random.Random(seed)
    n = len(a)
    va = [1.0 if x else 0.0 for x in a]
    vb = [1.0 if x else 0.0 for x in b]
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(va[i] for i in idx) / n
        db = sum(vb[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    p_le0 = sum(1 for d in diffs if d <= 0) / len(diffs)
    return {"ci95": [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]], "p_diff_le_0": p_le0}


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if (not x) and y)
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p": 1.0}
    k = min(b01, b10)
    p = 2 * sum(math.comb(n, i) for i in range(0, k + 1)) * 0.5**n
    return {"b01": b01, "b10": b10, "p": min(1.0, p)}


def activation_stats(name: str) -> dict | None:
    p = ART / "bench_reward_stats" / f"{name}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    g = sum(r["n_groups"] for r in rows)
    sg = sum(r["n_success_groups"] for r in rows)
    rg = sum(r["n_redist_groups"] for r in rows)
    s = sum(r["n_success"] for r in rows)
    n = sum(r["n_rollouts"] for r in rows)
    return {
        "groups": g,
        "frac_groups_with_success": sg / max(g, 1),
        "frac_groups_redistributed": rg / max(g, 1),
        "rollout_pass_rate": s / max(n, 1),
    }


def mean_std(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return m, math.sqrt(v)


def main() -> None:
    # difficulty lookup from the eval set
    diff_of = {}
    for l in (ART / "benchmarks" / "math500" / "test.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            diff_of[r["task_id"]] = r.get("difficulty", "?")

    table = {"arms": {}, "pairwise": {}, "meta": {"bench": "MATH-500 L4/L5 numeric", "n_test": len(diff_of)}}
    rows_by: dict[str, list[dict]] = {}

    for arm in ARMS:
        names = ["e0_math500_256"] if arm == "e0" else [f"{BENCH}_{arm}_s{s}" for s in SEEDS]
        per_seed = []
        for name in names:
            p = ART / f"bench_eval_{name}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            rows = load_rows("e0_math500_256" if arm == "e0" else name)
            rows_by[name] = rows
            entry = {
                "name": name,
                "pass_at_1": d["pass_at_1"],
                "avg_tokens_success": d["avg_tokens_success"],
                "rows_available": bool(rows),
            }
            if rows:
                lo, hi = bootstrap_ci([r["passed"] for r in rows])
                entry["boot_ci95"] = [lo, hi]
                for level in ("L4", "L5"):
                    sub = [r for r in rows if diff_of.get(r["task_id"]) == level]
                    if sub:
                        entry[f"pass_{level}"] = sum(r["passed"] for r in sub) / len(sub)
                        entry[f"n_{level}"] = len(sub)
            act = activation_stats(name)
            if act:
                entry["activation"] = act
            per_seed.append(entry)
        if not per_seed:
            continue
        passes = [s["pass_at_1"] for s in per_seed]
        toks = [s["avg_tokens_success"] for s in per_seed]
        mp, sp = mean_std(passes)
        mt, st = mean_std(toks)
        table["arms"][arm] = {
            "label": ARM_LABEL[arm],
            "n_seeds": len(per_seed),
            "per_seed": per_seed,
            "pass_mean": mp, "pass_std": sp,
            "tok_succ_mean": mt, "tok_succ_std": st,
        }

    # paired comparisons vs E2
    for arm in ("e3", "e5", "e6"):
        for s in SEEDS:
            a_name, b_name = f"{BENCH}_e2_s{s}", f"{BENCH}_{arm}_s{s}"
            if a_name not in rows_by or b_name not in rows_by:
                continue
            ra, rb = rows_by[a_name], rows_by[b_name]
            if not ra or not rb:
                continue
            by_id = {r["task_id"]: r for r in ra}
            common = [r for r in rb if r["task_id"] in by_id]
            if len(common) < 20:
                continue
            va = [bool(by_id[r["task_id"]]["passed"]) for r in common]
            vb = [bool(r["passed"]) for r in common]
            key = f"{arm}_vs_e2_s{s}"
            table["pairwise"][key] = mcnemar_exact(va, vb)
            table["pairwise"][key]["boot"] = paired_bootstrap_diff(vb, va)

    out = ART / "master_table_v2.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")

    print("| Arm | seeds | Pass@1 mean±std | boot CI (s42) | L4 | L5 | tok/succ | 激活率(重分配) |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, d in table["arms"].items():
        s42 = next((x for x in d["per_seed"] if "s42" in x["name"] or arm == "e0"), d["per_seed"][0])
        ci = s42.get("boot_ci95")
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and isinstance(ci[0], float) else "-"
        l4 = f"{s42.get('pass_L4', 0):.3f}" if "pass_L4" in s42 else "-"
        l5 = f"{s42.get('pass_L5', 0):.3f}" if "pass_L5" in s42 else "-"
        act = s42.get("activation", {}).get("frac_groups_redistributed")
        act_s = f"{act:.1%}" if act is not None else "-"
        print(
            f"| {d['label']} | {d['n_seeds']} | {d['pass_mean']:.3f}±{d['pass_std']:.3f} "
            f"| {ci_s} | {l4} | {l5} | {d['tok_succ_mean']:.0f}±{d['tok_succ_std']:.0f} | {act_s} |"
        )
    for k, v in table["pairwise"].items():
        b = v.get("boot", {})
        bc = b.get("ci95")
        bc_s = f"[{bc[0]:.3f},{bc[1]:.3f}]" if bc and isinstance(bc[0], float) else "-"
        pd = b.get("p_diff_le_0")
        pd_s = f"{pd:.3f}" if isinstance(pd, float) else "-"
        print(f"McNemar {k}: b01={v['b01']} b10={v['b10']} p={v['p']:.4f} | boot diff CI {bc_s} P(Δ≤0)={pd_s}")
    print(f"[table] {out}")


if __name__ == "__main__":
    main()
