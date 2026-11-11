"""End-to-end reward computation for one rollout group."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .base import BaseRewardConfig, TrajectoryRecord, compute_base_reward
from .groupwise import (
    GroupwiseConfig,
    apply_budget_gar,
    apply_tiny_gar,
    apply_uncertainty_gate,
    compute_uncertainty,
)
from .quality import QualityConfig, compute_grs_reward, compute_quality_score


class RewardMode(str, Enum):
    BINARY = "binary"
    LENGTH_PENALTY = "length_penalty"
    TINY_GRS = "tiny_grs"
    TINY_GAR = "tiny_gar"
    BUDGET_GAR = "budget_gar"
    UNCERTAIN_BUDGET_GAR = "uncertain_budget_gar"


@dataclass
class RewardPipelineConfig:
    mode: RewardMode = RewardMode.BINARY
    base: BaseRewardConfig | None = None
    quality: QualityConfig | None = None
    groupwise: GroupwiseConfig | None = None
    length_penalty_coef: float = 0.0005
    length_target_tokens: int = 256


class RewardPipeline:
    def __init__(self, cfg: RewardPipelineConfig | None = None) -> None:
        self.cfg = cfg or RewardPipelineConfig()

    def score_group(self, records: list[TrajectoryRecord]) -> list[TrajectoryRecord]:
        mode = self.cfg.mode
        for r in records:
            r.base_reward = compute_base_reward(r.graders, self.cfg.base)
            r.quality_score = (
                compute_quality_score(r.graders, self.cfg.quality) if r.graders.passed else 0.0
            )
            r.grs_reward = (
                compute_grs_reward(r.graders, r.quality_score, self.cfg.quality)
                if mode in {RewardMode.TINY_GRS, RewardMode.TINY_GAR, RewardMode.BUDGET_GAR, RewardMode.UNCERTAIN_BUDGET_GAR}
                else 0.0
            )
            r.gar_delta = 0.0
            r.uncertainty = compute_uncertainty(r)

        if mode == RewardMode.BINARY:
            for r in records:
                r.final_reward = r.base_reward
        elif mode == RewardMode.LENGTH_PENALTY:
            for r in records:
                excess = max(r.graders.generated_tokens - self.cfg.length_target_tokens, 0)
                r.final_reward = r.base_reward - self.cfg.length_penalty_coef * excess
        elif mode == RewardMode.TINY_GRS:
            for r in records:
                r.final_reward = r.grs_reward if r.graders.passed else r.base_reward
        elif mode == RewardMode.TINY_GAR:
            apply_tiny_gar(records, self.cfg.groupwise)
            for r in records:
                r.final_reward = (r.grs_reward if r.graders.passed else r.base_reward) + r.gar_delta
        elif mode in {RewardMode.BUDGET_GAR, RewardMode.UNCERTAIN_BUDGET_GAR}:
            apply_budget_gar(records, self.cfg.groupwise)
            if mode == RewardMode.UNCERTAIN_BUDGET_GAR:
                apply_uncertainty_gate(records, self.cfg.groupwise)
            for r in records:
                r.final_reward = (r.grs_reward if r.graders.passed else r.base_reward) + r.gar_delta
        else:
            raise ValueError(f"unknown mode: {mode}")

        return records
