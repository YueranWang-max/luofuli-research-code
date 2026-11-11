"""Groupwise advantage redistribution: Tiny-GAR, Budget-GAR, uncertainty gating.

These methods are project innovations, not verbatim MiMo-V2-Flash mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import TrajectoryRecord


@dataclass
class GroupwiseConfig:
    lambda_gar: float = 0.2
    lambda_0: float = 0.3
    lambda_min_ratio: float = 0.2
    eta: float = 1.0
    c_token: float = 0.5
    c_action: float = 0.3
    c_time: float = 0.2
    min_success_for_ranking: int = 2
    uncertainty_fallback: float = 0.6


def _centered_unit(values: list[float]) -> list[float]:
    """Map values to [-0.5, 0.5] via rank/normalized score centering."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) - 0.5 for v in values]


def apply_tiny_gar(
    records: list[TrajectoryRecord],
    cfg: GroupwiseConfig | None = None,
) -> list[TrajectoryRecord]:
    """Redistribute advantage inside a group using quality ranks of successful rollouts.

    Invariant: GAR delta sums to ~0 over successful members; failures get 0.
    """
    cfg = cfg or GroupwiseConfig()
    successes = [r for r in records if r.graders.passed and not r.graders.reward_hacking]
    if len(successes) < cfg.min_success_for_ranking:
        for r in records:
            r.gar_delta = 0.0
        return records

    scores = [r.quality_score for r in successes]
    centered = _centered_unit(scores)
    for r, c in zip(successes, centered):
        r.group_rank = c + 0.5
        r.gar_delta = cfg.lambda_gar * c

    for r in records:
        if r not in successes:
            r.gar_delta = 0.0
            r.group_rank = None
    return records


def apply_budget_gar(
    records: list[TrajectoryRecord],
    cfg: GroupwiseConfig | None = None,
) -> list[TrajectoryRecord]:
    """Budget-aware dynamic GAR: scale quality redistribution by difficulty and cost.

    Efficiency is compared only among successful trajectories so that short failures
    cannot farm positive advantage.
    """
    cfg = cfg or GroupwiseConfig()
    n = len(records)
    if n == 0:
        return records

    successes = [r for r in records if r.graders.passed and not r.graders.reward_hacking]
    group_pass_rate = len(successes) / n
    difficulty = 1.0 - group_pass_rate
    if len(successes) < cfg.min_success_for_ranking:
        for r in records:
            r.gar_delta = 0.0
        return records

    tokens = [max(r.graders.generated_tokens, 1) for r in successes]
    actions = [max(r.graders.tool_calls, 0) for r in successes]
    times = [max(r.graders.wall_time_seconds, 1e-3) for r in successes]
    t_span = max(tokens) - min(tokens) or 1.0
    a_span = max(actions) - min(actions) or 1.0
    w_span = max(times) - min(times) or 1.0

    efficiencies: list[float] = []
    for r, t, a, w in zip(successes, tokens, actions, times):
        nt = (t - min(tokens)) / t_span
        na = (a - min(actions)) / a_span
        nw = (w - min(times)) / w_span
        cost = cfg.c_token * nt + cfg.c_action * na + cfg.c_time * nw + 1e-6
        efficiencies.append(1.0 / cost)

    centered = _centered_unit(efficiencies)
    for r, c in zip(successes, centered):
        lam = cfg.lambda_0 * max(cfg.lambda_min_ratio, 1.0 - (difficulty ** cfg.eta))
        r.group_rank = c + 0.5
        r.gar_delta = lam * c

    for r in records:
        if r not in successes:
            r.gar_delta = 0.0
            r.group_rank = None
    return records


def compute_uncertainty(record: TrajectoryRecord) -> float:
    """Disagreement among cheap rankers + test/quality conflict, clipped to [0, 1]."""
    graders = record.graders
    parts: list[float] = []

    scores = list(graders.ranker_scores.values())
    if len(scores) >= 2:
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        parts.append(min(var * 8.0, 1.0))

    # test/quality conflict: passed but low quality, or failed but high quality
    q = record.quality_score
    if graders.passed and q < 0.35:
        parts.append(0.35 - q)
    if (not graders.passed) and q > 0.65:
        parts.append(q - 0.65)

    if not parts:
        return 0.0
    # max: either strong ranker disagreement or strong test/quality conflict is enough
    return min(max(parts), 1.0)


def apply_uncertainty_gate(
    records: list[TrajectoryRecord],
    cfg: GroupwiseConfig | None = None,
) -> list[TrajectoryRecord]:
    """Shrink GAR deltas when graders disagree; fall back to base reward if too uncertain."""
    cfg = cfg or GroupwiseConfig()
    for r in records:
        u = compute_uncertainty(r)
        r.uncertainty = u
        if u >= cfg.uncertainty_fallback:
            r.gar_delta = 0.0
        else:
            r.gar_delta = r.gar_delta * (1.0 - u)
    return records
