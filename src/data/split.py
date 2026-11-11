"""Template-aware splits and contamination checks."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from .schema import SplitBundle, TaskRecord

_NUM = re.compile(r"\d+")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _NUM.sub("#", text)
    text = re.sub(r"\s+", " ", text)
    return text


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def dedupe_by_normalized_text(tasks: list[TaskRecord]) -> tuple[list[TaskRecord], int]:
    """Drop exact-duplicate prompts only; number variants are distinct problems."""
    seen: set[str] = set()
    kept: list[TaskRecord] = []
    dropped = 0
    for t in tasks:
        h = content_hash(t.prompt)
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        kept.append(t)
    return kept, dropped


def template_split(
    tasks: list[TaskRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SplitBundle:
    by_template: dict[str, list[TaskRecord]] = defaultdict(list)
    for t in tasks:
        by_template[t.template_id].append(t)

    templates = sorted(by_template.keys())
    n = len(templates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(templates[:n_train])
    val_ids = set(templates[n_train : n_train + n_val])
    test_ids = set(templates[n_train + n_val :])

    bundle = SplitBundle(
        template_overlap={
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        }
    )
    for tid, items in by_template.items():
        if tid in train_ids:
            bundle.train.extend(items)
        elif tid in val_ids:
            bundle.val.extend(items)
        else:
            bundle.test.extend(items)
    return bundle


def contamination_report(bundle: SplitBundle) -> dict:
    hashes = {
        "train": {content_hash(t.prompt) for t in bundle.train},
        "val": {content_hash(t.prompt) for t in bundle.val},
        "test": {content_hash(t.prompt) for t in bundle.test},
    }
    return {
        "train_val": sorted(hashes["train"] & hashes["val"]),
        "train_test": sorted(hashes["train"] & hashes["test"]),
        "val_test": sorted(hashes["val"] & hashes["test"]),
        "sizes": {
            "train": len(bundle.train),
            "val": len(bundle.val),
            "test": len(bundle.test),
        },
    }
