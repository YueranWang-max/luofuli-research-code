"""Data schemas for verifiable tasks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskRecord:
    task_id: str
    template_id: str
    prompt: str
    reference_answer: str | None = None
    public_tests: list[str] = field(default_factory=list)
    hidden_tests: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    domain: str = "math"
    meta: dict = field(default_factory=dict)


@dataclass
class SplitBundle:
    train: list[TaskRecord] = field(default_factory=list)
    val: list[TaskRecord] = field(default_factory=list)
    test: list[TaskRecord] = field(default_factory=list)
    template_overlap: dict[str, list[str]] = field(default_factory=dict)
