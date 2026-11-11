"""Tiny-GRS: quality score for successful trajectories only."""

from __future__ import annotations

from dataclasses import dataclass

from .base import GraderOutputs


@dataclass
class QualityConfig:
    coverage: float = 0.35
    minimality: float = 0.20
    complexity: float = 0.15
    style: float = 0.10
    efficiency: float = 0.10
    scope: float = 0.10
    alpha: float = 0.7
    beta: float = 0.3


def compute_quality_score(graders: GraderOutputs, cfg: QualityConfig | None = None) -> float:
    """Weighted quality in [0, 1]. Callers should only use this on passing trajectories."""
    cfg = cfg or QualityConfig()
    return (
        cfg.coverage * graders.coverage
        + cfg.minimality * graders.minimality
        + cfg.complexity * graders.complexity
        + cfg.style * graders.style
        + cfg.efficiency * graders.efficiency
        + cfg.scope * graders.scope
    )


def compute_grs_reward(
    graders: GraderOutputs,
    quality: float,
    cfg: QualityConfig | None = None,
) -> float:
    """R_grs = R_pass * (alpha + beta * Q) for successes; 0 for failures.

    Failed trajectories must never receive a positive quality bonus.
    """
    cfg = cfg or QualityConfig()
    if not graders.passed or graders.prohibited_action or graders.reward_hacking:
        return 0.0
    q = min(max(quality, 0.0), 1.0)
    return cfg.alpha + cfg.beta * q
