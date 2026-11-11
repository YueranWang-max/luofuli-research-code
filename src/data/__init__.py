"""Data splits and prompt construction for verifiable math/code tasks."""

from .schema import SplitBundle, TaskRecord
from .split import contamination_report, dedupe_by_normalized_text, template_split

__all__ = [
    "TaskRecord",
    "SplitBundle",
    "template_split",
    "dedupe_by_normalized_text",
    "contamination_report",
]
