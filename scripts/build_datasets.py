"""Generate harder verifiable math/code datasets with domain-stratified splits."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import TaskRecord, contamination_report, dedupe_by_normalized_text


def make_math_tasks(n_templates: int = 48, per_template: int = 10, seed: int = 42) -> list[TaskRecord]:
    rng = random.Random(seed)
    tasks: list[TaskRecord] = []

    ops = [
        ("mul3", lambda a, b, c: a * b * c, "{a} times {b} times {c}"),
        ("addmul", lambda a, b, c: (a + b) * c, "({a} plus {b}) times {c}"),
        ("mulsub", lambda a, b, c: a * b - c, "{a} times {b} minus {c}"),
        ("mixed", lambda a, b, c: a * b + c * c, "{a} times {b} plus {c} squared"),
        ("negmix", lambda a, b, c: -a * b + c, "minus {a} times {b} plus {c}"),
        ("divmix", lambda a, b, c: (a * b) // c, "product of {a} and {b} integer-divided by {c}"),
    ]
    for t in range(n_templates):
        name, fn, ph = ops[t % len(ops)]
        for i in range(per_template):
            a = rng.randint(11, 99)
            b = rng.randint(11, 40)
            c = rng.randint(3, 17)
            ans = fn(a, b, c)
            prompt = f"Compute the {ph.format(a=a, b=b, c=c)}. Answer with a single integer only."
            difficulty = "easy" if t < 12 else ("medium" if t < 30 else "hard")
            tasks.append(
                TaskRecord(
                    task_id=f"math_{name}_{t}_{i}",
                    template_id=f"math_{name}_{t}",
                    prompt=prompt,
                    reference_answer=str(ans),
                    difficulty=difficulty,
                    domain="math",
                )
            )
    return tasks


CODE_TEMPLATES = [
    (
        "add_fn",
        "Fill in the blank so the function returns the sum of a and b:\n```\ndef add(a, b):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        ["assert add(-1, 1) == 0", "assert add(10, 5) == 15"],
    ),
    (
        "double_fn",
        "Fill in the blank so the function returns two times x:\n```\ndef double(x):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert double(3) == 6", "assert double(0) == 0"],
        ["assert double(-2) == -4", "assert double(10) == 20"],
    ),
    (
        "is_positive",
        "Fill in the blank so the function returns True when n > 0:\n```\ndef is_positive(n):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert is_positive(1) is True", "assert is_positive(-1) is False"],
        ["assert is_positive(0) is False", "assert is_positive(5) is True"],
    ),
    (
        "last_char",
        "Fill in the blank so the function returns the last character of s:\n```\ndef last_char(s):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert last_char('abc') == 'c'", "assert last_char('a') == 'a'"],
        ["assert last_char('') == '' if False else last_char('xy') == 'y'"],
    ),
    (
        "square_fn",
        "Fill in the blank so the function returns x squared:\n```\ndef square(x):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert square(3) == 9", "assert square(0) == 0"],
        ["assert square(-2) == 4", "assert square(5) == 25"],
    ),
    (
        "min_fn",
        "Fill in the blank so the function returns the smaller of a and b:\n```\ndef my_min(a, b):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert my_min(1, 2) == 1", "assert my_min(5, 5) == 5"],
        ["assert my_min(-3, -1) == -3", "assert my_min(0, -10) == -10"],
    ),
    (
        "first_elem",
        "Fill in the blank so the function returns the first element of xs:\n```\ndef first(xs):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert first([1,2,3]) == 1", "assert first(['a']) == 'a'"],
        ["assert first([7,8]) == 7"],
    ),
    (
        "len_fn",
        "Fill in the blank so the function returns the length of s:\n```\ndef slen(s):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert slen('abc') == 3", "assert slen('') == 0"],
        ["assert slen('hi') == 2", "assert slen('hello') == 5"],
    ),
    (
        "abs_fn",
        "Fill in the blank so the function returns the absolute value of x:\n```\ndef my_abs(x):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert my_abs(-3) == 3", "assert my_abs(2) == 2"],
        ["assert my_abs(0) == 0", "assert my_abs(-10) == 10"],
    ),
    (
        "not_fn",
        "Fill in the blank so the function returns the opposite boolean:\n```\ndef my_not(x):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert my_not(True) is False", "assert my_not(False) is True"],
        ["assert my_not(1 > 2) is True"],
    ),
    (
        "concat_fn",
        "Fill in the blank so the function returns a and b joined as strings:\n```\ndef join_str(a, b):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert join_str('a', 'b') == 'ab'", "assert join_str('', 'x') == 'x'"],
        ["assert join_str('hi', '!') == 'hi!'"],
    ),
    (
        "div_fn",
        "Fill in the blank so the function returns a divided by b as a float:\n```\ndef div(a, b):\n    return ___\n```\nAnswer with just the expression for the blank.",
        ["assert div(4, 2) == 2.0", "assert div(3, 2) == 1.5"],
        ["assert div(10, 5) == 2.0"],
    ),
]


def make_code_tasks(per_template: int = 12, seed: int = 42) -> list[TaskRecord]:
    tasks: list[TaskRecord] = []
    for t, (name, desc, pub, hid) in enumerate(CODE_TEMPLATES):
        for i in range(per_template):
            tasks.append(
                TaskRecord(
                    task_id=f"code_{name}_{i}",
                    template_id=f"code_{name}",
                    prompt=(
                        "Example:\n"
                        "Task: Write a function `double(x)` that returns 2*x.\n"
                        "Solution:\n```\ndef double(x):\n    return 2 * x\n```\n\n"
                        + desc
                        + "\nSolution:"
                    ),
                    public_tests=list(pub),
                    hidden_tests=list(hid),
                    difficulty="easy" if t < 4 else ("medium" if t < 8 else "hard"),
                    domain="code",
                )
            )
    return tasks


def stratified_template_split(
    tasks: list[TaskRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    from collections import defaultdict

    from data import SplitBundle

    by_domain_tpl: dict[str, dict[str, list[TaskRecord]]] = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        by_domain_tpl[t.domain][t.template_id].append(t)

    bundle = SplitBundle()
    for domain, tpls in by_domain_tpl.items():
        names = sorted(tpls.keys())
        n = len(names)
        n_train = max(int(n * train_ratio), 1)
        n_val = max(int(n * val_ratio), 0)
        train_ids = names[:n_train]
        val_ids = names[n_train : n_train + n_val]
        test_ids = names[n_train + n_val :]
        if not test_ids and names:
            test_ids = [names[-1]]
            if names[-1] in train_ids:
                train_ids = train_ids[:-1]
        bundle.template_overlap[f"{domain}_train"] = train_ids
        bundle.template_overlap[f"{domain}_val"] = val_ids
        bundle.template_overlap[f"{domain}_test"] = test_ids
        for tid, items in tpls.items():
            if tid in train_ids:
                bundle.train.extend(items)
            elif tid in val_ids:
                bundle.val.extend(items)
            else:
                bundle.test.extend(items)
    return bundle


def main() -> None:
    out = ROOT / "artifacts" / "datasets"
    out.mkdir(parents=True, exist_ok=True)

    math_tasks, dropped = dedupe_by_normalized_text(make_math_tasks())
    code_tasks = make_code_tasks()
    all_tasks = math_tasks + code_tasks

    bundle = stratified_template_split(all_tasks)
    report = contamination_report(bundle)

    def dump(name: str, items: list[TaskRecord]) -> None:
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for t in items:
                f.write(json.dumps(t.__dict__, ensure_ascii=False) + "\n")
        by_d: dict[str, int] = {}
        for t in items:
            by_d[t.domain] = by_d.get(t.domain, 0) + 1
        print(f"[write] {path.name}: {len(items)} {by_d}")

    dump("train", bundle.train)
    dump("val", bundle.val)
    dump("test", bundle.test)
    (out / "contamination_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[dedupe] dropped={dropped}")


if __name__ == "__main__":
    main()
