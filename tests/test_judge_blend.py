"""Tests for Ollama judge blending (offline, no live calls)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewards.base import GraderOutputs, TrajectoryRecord
from rewards.judge_blend import JudgeBlendConfig, blend_quality
from rewards.groupwise import compute_uncertainty


def _rec(passed: bool = True, q: float = 0.5) -> TrajectoryRecord:
    g = GraderOutputs(passed=passed, coverage=1.0, minimality=q, efficiency=q)
    r = TrajectoryRecord("a", "t", "g", 0, g)
    r.quality_score = q
    return r


def test_blend_quality_weight():
    cfg = JudgeBlendConfig(enabled=True, weight=0.5)
    assert abs(blend_quality(0.2, 0.8, cfg) - 0.5) < 1e-9
    cfg2 = JudgeBlendConfig(enabled=True, weight=1.0)
    assert abs(blend_quality(0.2, 0.8, cfg2) - 0.8) < 1e-9


def test_uncertainty_high_when_rankers_disagree():
    r = _rec(passed=True, q=0.5)
    r.graders.ranker_scores = {"a": 0.9, "b": 0.1, "c": 0.85, "d": 0.15}
    assert compute_uncertainty(r) > 0.5


def test_uncertainty_low_when_rankers_agree():
    r = _rec(passed=True, q=0.5)
    r.graders.ranker_scores = {"a": 0.5, "b": 0.5, "c": 0.5}
    assert compute_uncertainty(r) < 0.2
