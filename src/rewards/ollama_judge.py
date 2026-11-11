"""Ollama-based multi-judge client for cheap quality scoring and uncertainty.

Judges run outside training VRAM (GGUF via local Ollama API).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_JUDGES = ["qwen3:8b", "qwen3.5:4b", "openbmb/minicpm5-2b:latest"]
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


@dataclass
class JudgeScore:
    judge: str
    quality: float
    ok: bool
    raw: str = ""


@dataclass
class MultiJudgeResult:
    scores: list[JudgeScore] = field(default_factory=list)

    @property
    def mean_quality(self) -> float:
        vals = [s.quality for s in self.scores if s.ok]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def uncertainty(self) -> float:
        vals = [s.quality for s in self.scores if s.ok]
        if len(vals) < 2:
            return 1.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return min(max(var * 8.0, 0.0), 1.0)


def _extract_score(text: str) -> float | None:
    m = re.search(r"(?i)score\s*[:=]?\s*([01](?:\.\d+)?)", text)
    if m:
        v = float(m.group(1))
        return min(max(v, 0.0), 1.0)
    m = re.search(r"\b([01](?:\.\d+)?)\b", text)
    if m:
        v = float(m.group(1))
        return min(max(v, 0.0), 1.0)
    return None


PROMPT = (
    "Rate the following answer quality from 0.0 to 1.0 for correctness and conciseness.\n"
    "Reply with a single line: Score: <number>\n\n"
    "Task: {task}\nAnswer: {answer}\n"
)


def judge_once(judge: str, task: str, answer: str, timeout: float = 60.0) -> JudgeScore:
    payload = {
        "model": judge,
        "prompt": PROMPT.format(task=task, answer=answer),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 32},
        "think": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw = body.get("response", "")
        score = _extract_score(raw)
        return JudgeScore(judge, score if score is not None else 0.0, score is not None, raw)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return JudgeScore(judge, 0.0, False, str(e))


def multi_judge(
    task: str,
    answer: str,
    judges: list[str] | None = None,
) -> MultiJudgeResult:
    judges = judges or DEFAULT_JUDGES
    out = MultiJudgeResult()
    for j in judges:
        out.scores.append(judge_once(j, task, answer))
    return out


if __name__ == "__main__":
    demo = multi_judge("What is 2+2?", "4")
    print(json.dumps({
        "mean": demo.mean_quality,
        "uncertainty": demo.uncertainty,
        "scores": [(s.judge, s.quality, s.ok) for s in demo.scores],
    }, ensure_ascii=False, indent=2))
