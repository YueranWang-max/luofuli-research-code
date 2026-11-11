"""Reward suite: binary outcome, Tiny-GRS quality, Tiny-GAR / Budget-GAR redistribution."""

from .base import (
    BaseRewardConfig,
    GraderOutputs,
    TrajectoryRecord,
    compute_base_reward,
)
from .quality import QualityConfig, compute_quality_score, compute_grs_reward
from .groupwise import (
    GroupwiseConfig,
    apply_tiny_gar,
    apply_budget_gar,
    apply_uncertainty_gate,
    compute_uncertainty,
)
from .pipeline import RewardMode, RewardPipeline, RewardPipelineConfig

__all__ = [
    "BaseRewardConfig",
    "GraderOutputs",
    "TrajectoryRecord",
    "compute_base_reward",
    "QualityConfig",
    "compute_quality_score",
    "compute_grs_reward",
    "GroupwiseConfig",
    "apply_tiny_gar",
    "apply_budget_gar",
    "apply_uncertainty_gate",
    "compute_uncertainty",
    "RewardMode",
    "RewardPipeline",
    "RewardPipelineConfig",
]
