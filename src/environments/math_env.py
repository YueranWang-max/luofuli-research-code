"""Deterministic math answer verification."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass


@dataclass
class VerifyResult:
    passed: bool
    extracted: str | None = None
    reason: str = ""


_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_ANS = re.compile(r"(?:answer|final answer|答案)\s*[:：]?\s*(-?[\d.,]+/?[\d.,]*)", re.I)


def _clean_num(s: str) -> str:
    return s.strip().rstrip(".").replace(",", "")


def extract_answer(text: str) -> str | None:
    m = _BOXED.findall(text)
    if m:
        return _clean_num(m[-1])
    m = _ANS.findall(text)
    if m:
        return _clean_num(m[-1])
    # last integer/float in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return _clean_num(nums[-1]) if nums else None


def _to_float(expr: str) -> float | None:
    expr = expr.strip()
    if "/" in expr and re.fullmatch(r"-?\d+/\d+", expr):
        a, b = expr.split("/")
        b = float(b)
        value = float(a) / b if b else None
        return value if value is not None and math.isfinite(value) else None
    try:
        value = float(expr)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def safe_eval_expr(expr: str) -> float | None:
    """Evaluate a tiny arithmetic expression safely via AST."""
    expr = expr.strip().replace("^", "**")
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return _to_float(expr)

    def _ev(n):
        if isinstance(n, ast.Expression):
            return _ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = _ev(n.operand)
            if v is None:
                return None
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = _ev(n.left), _ev(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            return a / b if b else None
        return None

    try:
        value = _ev(node)
        return value if value is not None and math.isfinite(value) else None
    except (OverflowError, ZeroDivisionError):
        # Arbitrarily large model-emitted numerals and overflowing arithmetic are
        # invalid numeric answers for this float-based legacy verifier.
        return None


class MathVerifier:
    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def verify(self, completion: str, reference: str) -> VerifyResult:
        got = extract_answer(completion)
        if got is None:
            return VerifyResult(False, None, "no_answer")
        ref = _clean_num(reference)
        g = safe_eval_expr(got)
        r = safe_eval_expr(ref)
        if g is None or r is None:
            ok = got.strip() == ref.strip()
            return VerifyResult(ok, got, "string_match" if ok else "mismatch")
        ok = abs(g - r) <= self.tol * max(1.0, abs(r))
        return VerifyResult(ok, got, "ok" if ok else "mismatch")
