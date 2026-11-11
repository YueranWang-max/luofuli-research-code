"""Wire Ollama multi-judge into quality scoring + uncertainty gate."""

from __future__ import annotations

from dataclasses import dataclass

from .groupwise import compute_uncertainty
from .ollama_judge import DEFAULT_JUDGES, MultiJudgeResult, multi_judge
from .base import TrajectoryRecord


@dataclass
class JudgeBlendConfig:
    enabled: bool = False
    weight: float = 0.5
    judges: list[str] | None = None
    fallback_uncertainty: float = 0.8


def judge_quality(task: str, answer: str, cfg: JudgeBlendConfig | None = None) -> MultiJudgeResult:
    cfg = cfg or JudgeBlendConfig()
    return multi_judge(task, answer, cfg.judges or DEFAULT_JUDGES)


def blend_quality(local_q: float, judge_q: float, cfg: JudgeBlendConfig | None = None) -> float:
    cfg = cfg or JudgeBlendConfig()
    w = cfg.weight
    return float((1.0 - w) * local_q + w * judge_q)


def augment_with_judges(
    record: TrajectoryRecord,
    task_prompt: str,
    answer: str,
    cfg: JudgeBlendConfig | None = None,
) -> TrajectoryRecord:
    """Blend Ollama judge scores into quality and uncertainty.

    Falls back to high uncertainty when judges disagree strongly or fail.
    """
    cfg = cfg or JudgeBlendConfig()
    if not cfg.enabled:
        return record

    res = judge_quality(task_prompt, answer, cfg)
    n_ok = sum(1 for s in res.scores if s.ok)
    if n_ok == 0:
        record.uncertainty = cfg.fallback_uncertainty
        return record

    # Blend quality only for successful trajectories.
    if record.graders.passed:
        blended = blend_quality(record.quality_score, res.mean_quality, cfg)
        record.quality_score = blended

    # Uncertainty: max(local, judge disagreement)
    judge_u = res.uncertainty if n_ok >= 2 else cfg.fallback_uncertainty
    local_u = compute_uncertainty(record)
    record.uncertainty = min(max(local_u, judge_u), 1.0)
    return record
