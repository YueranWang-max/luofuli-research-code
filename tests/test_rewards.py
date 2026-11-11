"""Unit tests for reward invariants from PLAN.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewards import (  # noqa: E402
    GroupwiseConfig,
    GraderOutputs,
    QualityConfig,
    RewardPipeline,
    RewardPipelineConfig,
    RewardMode,
    TrajectoryRecord,
    apply_budget_gar,
    apply_tiny_gar,
    apply_uncertainty_gate,
    compute_base_reward,
    compute_grs_reward,
    compute_quality_score,
    compute_uncertainty,
)


def _rec(tid: str, **kwargs) -> TrajectoryRecord:
    g = GraderOutputs(**kwargs)
    return TrajectoryRecord(
        trajectory_id=tid,
        task_id="t0",
        group_id="g0",
        seed=42,
        graders=g,
    )


def test_failed_short_trajectory_gets_no_efficiency_bonus():
    recs = [
        _rec("a", passed=False, generated_tokens=1),
        _rec("b", passed=True, generated_tokens=500, efficiency=1.0, coverage=1.0),
        _rec("c", passed=True, generated_tokens=200, efficiency=1.0, coverage=1.0),
    ]
    apply_budget_gar(recs)
    assert recs[0].gar_delta == 0.0


def test_single_success_has_zero_gar():
    recs = [
        _rec("a", passed=True, coverage=0.9, generated_tokens=100),
        _rec("b", passed=False, generated_tokens=50),
    ]
    apply_tiny_gar(recs)
    assert recs[0].gar_delta == 0.0


def test_identical_quality_zero_gar():
    recs = [
        _rec("a", passed=True, coverage=0.5, generated_tokens=100),
        _rec("b", passed=True, coverage=0.5, generated_tokens=100),
    ]
    apply_tiny_gar(recs)
    assert abs(recs[0].gar_delta) < 1e-9
    assert abs(recs[1].gar_delta) < 1e-9


def test_gar_zero_mean_among_successes():
    recs = [
        _rec("a", passed=True, coverage=0.2, generated_tokens=100),
        _rec("b", passed=True, coverage=0.8, generated_tokens=120),
        _rec("c", passed=True, coverage=0.5, generated_tokens=110),
    ]
    for r in recs:
        r.quality_score = compute_quality_score(r.graders)
    apply_tiny_gar(recs)
    total = sum(r.gar_delta for r in recs)
    assert abs(total) < 1e-9


def test_failed_trajectory_quality_does_not_boost_reward():
    g = GraderOutputs(passed=False, coverage=1.0, minimality=1.0, complexity=1.0, style=1.0, efficiency=1.0, scope=1.0)
    q = compute_quality_score(g)
    assert compute_grs_reward(g, q) == 0.0
    assert compute_base_reward(g) == 0.0


def test_high_uncertainty_falls_back():
    rec = _rec(
        "a",
        passed=True,
        coverage=0.5,
        ranker_scores={"r1": 0.9, "r2": 0.1, "r3": 0.8, "r4": 0.2},
    )
    rec.quality_score = compute_quality_score(rec.graders)
    rec.gar_delta = 0.1
    apply_uncertainty_gate([rec])
    assert rec.uncertainty > 0.6
    assert rec.gar_delta == 0.0


def test_prohibited_action_penalized():
    g = GraderOutputs(passed=True, prohibited_action=True)
    assert compute_base_reward(g) <= -0.5


def test_pipeline_binary_matches_base():
    recs = [_rec("a", passed=True), _rec("b", passed=False)]
    RewardPipeline(RewardPipelineConfig(mode=RewardMode.BINARY)).score_group(recs)
    assert recs[0].final_reward == 1.0
    assert recs[1].final_reward == 0.0


def test_pipeline_tiny_gar_changes_only_success_rank():
    recs = [
        _rec("a", passed=True, coverage=0.1, generated_tokens=100),
        _rec("b", passed=True, coverage=0.9, generated_tokens=100),
        _rec("c", passed=False, generated_tokens=10),
    ]
    pipe = RewardPipeline(RewardPipelineConfig(mode=RewardMode.TINY_GAR, groupwise=GroupwiseConfig(lambda_gar=0.2)))
    pipe.score_group(recs)
    assert recs[2].gar_delta == 0.0
    assert abs(recs[0].gar_delta + recs[1].gar_delta) < 1e-9


def test_timeout_reward_deterministic():
    g = GraderOutputs(timed_out=True)
    assert compute_base_reward(g) == -0.1
    assert compute_base_reward(g) == -0.1


def test_safe_eval_none_propagation():
    """Sub-expressions that cannot evaluate must yield None, not crash (regression)."""
    from environments.math_env import safe_eval_expr

    assert safe_eval_expr("2 + x") is None
    assert safe_eval_expr("-(2 + a)") is None
    assert safe_eval_expr("1/0") is None
    assert safe_eval_expr("2+3*4") == 14.0
