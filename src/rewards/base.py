"""Base outcome reward and shared trajectory/verifier data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseRewardConfig:
    pass_reward: float = 1.0
    fail_reward: float = 0.0
    invalid_format: float = -0.1
    timeout: float = -0.1
    prohibited_action: float = -0.5


@dataclass
class GraderOutputs:
    """Raw outputs from deterministic graders for one trajectory."""

    passed: bool = False
    hidden_passed: bool = False
    invalid_format: bool = False
    timed_out: bool = False
    prohibited_action: bool = False
    reward_hacking: bool = False

    # quality components in [0, 1]
    coverage: float = 0.0
    minimality: float = 0.0
    complexity: float = 0.0
    style: float = 0.0
    efficiency: float = 0.0
    scope: float = 0.0

    # resource usage
    generated_tokens: int = 0
    tool_calls: int = 0
    wall_time_seconds: float = 0.0
    files_changed: int = 0
    lines_changed: int = 0

    # ranker votes in [0, 1] for uncertainty estimation
    ranker_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
    trajectory_id: str
    task_id: str
    group_id: str
    seed: int
    graders: GraderOutputs

    # filled by reward pipeline
    base_reward: float = 0.0
    quality_score: float = 0.0
    grs_reward: float = 0.0
    gar_delta: float = 0.0
    uncertainty: float = 0.0
    final_reward: float = 0.0
    group_rank: float | None = None
    checkpoint_id: str = "base"


def compute_base_reward(graders: GraderOutputs, cfg: BaseRewardConfig | None = None) -> float:
    """Binary / sparse outcome reward. Safety and format penalties stack on top."""
    cfg = cfg or BaseRewardConfig()
    if graders.prohibited_action or graders.reward_hacking:
        return cfg.prohibited_action
    if graders.invalid_format:
        return cfg.invalid_format
    if graders.timed_out:
        return cfg.timeout
    return cfg.pass_reward if graders.passed else cfg.fail_reward
