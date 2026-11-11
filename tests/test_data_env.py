"""Tests for data splits and verifiers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import TaskRecord, contamination_report, dedupe_by_normalized_text, template_split  # noqa: E402
from environments import CodeVerifier, MathVerifier  # noqa: E402
from environments.code_env import complexity_delta, detect_reward_hacking  # noqa: E402
from environments.math_env import safe_eval_expr  # noqa: E402


def test_template_split_no_overlap():
    titles = ["add", "sum", "total", "plus", "combine", "accumulate"]
    tasks = []
    for t in range(6):
        for i in range(3):
            tasks.append(
                TaskRecord(f"t{t}_{i}", f"tpl{t}", f"Please {titles[t]} the value {t} and {i}")
            )
    bundle = template_split(tasks, train_ratio=0.5, val_ratio=0.17)
    report = contamination_report(bundle)
    tr = set(bundle.template_overlap["train"])
    va = set(bundle.template_overlap["val"])
    te = set(bundle.template_overlap["test"])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert report["train_val"] == []
    assert report["train_test"] == []
    assert report["val_test"] == []


def test_dedupe():
    tasks = [
        TaskRecord("a", "t1", "What is 2+2?"),
        TaskRecord("b", "t1", "What is 2+2?"),
        TaskRecord("c", "t1", "What is 3+3?"),
    ]
    kept, dropped = dedupe_by_normalized_text(tasks)
    assert dropped == 1
    assert len(kept) == 2


def test_math_verifier():
    v = MathVerifier()
    assert v.verify("The answer is 4", "4").passed
    assert v.verify("\\boxed{42}", "42").passed
    assert not v.verify("I don't know", "4").passed


def test_math_verifier_rejects_integer_beyond_float_range():
    huge = "9" * 10000
    assert safe_eval_expr(huge) is None
    result = MathVerifier().verify(f"The answer is {huge}", "4")
    assert not result.passed


def test_code_verifier_pass_and_fail():
    v = CodeVerifier(timeout_s=5.0)
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    assert v.verify(good, tests, tests).passed
    assert not v.verify(bad, tests, tests).passed


def test_reward_hacking_detected():
    assert detect_reward_hacking("import os\nprint(os.environ)")
    assert detect_reward_hacking("except:\n    pass")
    assert not detect_reward_hacking("def f(x):\n    return x + 1")


def test_complexity_delta():
    a = "def f(x):\n    return x\n"
    b = "def f(x):\n    if x > 0:\n        return x\n    return -x\n"
    assert complexity_delta(a, b) > 0
