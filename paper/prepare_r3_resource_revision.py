"""Archive the R2 matrix and resource-cancelled non-core run before freezing R3."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts" / "v3"
REVISION = V3 / "technical_revisions" / "20261018_resource_bounded_core"
KEY = "Qwen2.5-1.5B-Instruct_gr3_fixed_s42"


def main() -> None:
    revision = REVISION
    assert not revision.exists()
    revision.mkdir(parents=True)
    lock = ROOT / "artifacts" / ".gpu_lock"
    assert lock.is_file(), "Expected stale lock from interrupted R2 queue"
    text = lock.read_text(encoding="utf-8")
    pid = int(text.split("pid=", 1)[1].split()[0])
    try:
        os.kill(pid, 0)
    except OSError:
        pass
    else:
        raise RuntimeError(f"Refusing revision while GPU owner pid {pid} is alive")
    (revision / "stale_gpu_lock.txt").write_text(text, encoding="utf-8")
    lock.unlink()
    for name in ("training_manifest.json", "frozen_sources"):
        source = V3 / name
        assert source.exists()
        shutil.move(str(source), str(revision / name))
    partial = V3 / "runs" / KEY
    assert partial.is_dir() and not (partial / "completed.json").exists()
    archive = V3 / "resource_cancelled_attempts" / KEY
    archive.parent.mkdir(exist_ok=True)
    assert not archive.exists()
    shutil.move(str(partial), str(archive))
    log = V3 / "logs" / f"{KEY}.log"
    if log.exists():
        shutil.copy2(log, revision / log.name)
    groups = sum(1 for line in (archive / "groups.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip())
    record = {
        "amendment": "paper/TECHNICAL_AMENDMENT_R3.md",
        "observed_unix": time.time(),
        "reason": "Prospective resource reduction before any confirmation evaluation",
        "cancelled_key": KEY,
        "saved_groups": groups,
        "entered_primary_analysis": False,
        "completed_formal_evaluations_at_revision": 0,
    }
    assert groups == 96
    (revision / "revision_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
