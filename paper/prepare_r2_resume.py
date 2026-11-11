"""Archive failed formal attempt 2 and construct the audited checkpoint-50 resume directory."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts" / "v3"
KEY = "Qwen2.5-1.5B-Instruct_centered_budget_s42"
SOURCE = V3 / "runs" / KEY
ARCHIVE = V3 / "failed_attempts" / f"{KEY}_attempt2"
REVISION = V3 / "technical_revisions" / "20261017_legacy_overflow"


def filtered_jsonl(source: Path, target: Path, predicate) -> int:
    kept = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if predicate(row):
            kept.append(json.dumps(row, ensure_ascii=False))
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(kept)


def main() -> None:
    assert SOURCE.is_dir() and not ARCHIVE.exists()
    assert (SOURCE / "checkpoint-50" / "trainer_state.json").is_file()
    state = json.loads((SOURCE / "checkpoint-50" / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 50 and state["max_steps"] == 100
    REVISION.mkdir(parents=True, exist_ok=False)
    for name in ("training_manifest.json", "frozen_sources"):
        path = V3 / name
        assert path.exists()
        shutil.move(str(path), str(REVISION / name))
    log = V3 / "logs" / f"{KEY}.log"
    assert log.is_file()
    shutil.copy2(log, REVISION / f"{KEY}_attempt2.log")
    shutil.move(str(SOURCE), str(ARCHIVE))
    SOURCE.mkdir(parents=True)
    shutil.copytree(ARCHIVE / "checkpoint-50", SOURCE / "checkpoint-50")
    groups = filtered_jsonl(ARCHIVE / "groups.jsonl", SOURCE / "groups.jsonl",
                            lambda row: row["global_step"] < 50)
    phases = filtered_jsonl(ARCHIVE / "phase_times.jsonl", SOURCE / "phase_times.jsonl",
                            lambda row: row["step"] < 50)
    assert groups == 100 and phases == 100
    plan = {
        "checkpoint": "checkpoint-50",
        "resume_global_step": 50,
        "retained_groups": groups,
        "retained_phase_events": phases,
        "failed_attempt_archive": str(ARCHIVE.relative_to(ROOT)),
        "technical_amendment": "paper/TECHNICAL_AMENDMENT_R2.md",
    }
    (SOURCE / "resume_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
