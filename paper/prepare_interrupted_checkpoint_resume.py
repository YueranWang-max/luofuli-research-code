"""Archive an interrupted run and reconstruct an audited checkpoint-50 resume directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts" / "v3"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("--repair-archive")
    parser.add_argument("--invalidate-completed-evidence", action="store_true")
    parser.add_argument("--restart-from-scratch", action="store_true")
    args = parser.parse_args()
    source = V3 / "runs" / args.key
    if args.restart_from_scratch:
        assert source.is_dir() and not (source / "completed.json").exists()
        lock = ROOT / "artifacts" / ".gpu_lock"
        lock_text = lock.read_text(encoding="utf-8") if lock.is_file() else "lock absent"
        if lock.is_file():
            pid = int(lock_text.split("pid=", 1)[1].split()[0])
            try:
                os.kill(pid, 0)
            except OSError:
                pass
            else:
                raise RuntimeError(f"Refusing recovery while pid {pid} is alive")
        archive_root = V3 / "interrupted_attempts"
        archive_root.mkdir(exist_ok=True)
        attempt = 1
        while (archive_root / f"{args.key}_attempt{attempt}").exists():
            attempt += 1
        archive = archive_root / f"{args.key}_attempt{attempt}"
        lock.unlink(missing_ok=True)
        shutil.move(str(source), str(archive))
        run_log = V3 / "logs" / f"{args.key}.log"
        if run_log.exists():
            shutil.move(str(run_log), str(archive / "training.log"))
        report = {
            "restart": "from_scratch",
            "interrupted_attempt_archive": str(archive.relative_to(ROOT)),
            "stale_lock": lock_text,
            "prepared_unix": time.time(),
            "reason": "Process ended before the first formal checkpoint; archive and rerun the frozen job",
        }
        print(json.dumps(report, indent=2))
        return
    if args.repair_archive:
        archive = ROOT / args.repair_archive
        assert archive.is_dir() and source.is_dir() and not (source / "completed.json").exists()
        checkpoint = archive / "checkpoint-50"
    else:
        assert source.is_dir() and (
            args.invalidate_completed_evidence or not (source / "completed.json").exists()
        )
        checkpoint = source / "checkpoint-50"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 50 and state["max_steps"] == 100
    if args.repair_archive:
        lock_text = "already archived before interrupted checkpoint copy"
    else:
        lock = ROOT / "artifacts" / ".gpu_lock"
        if lock.is_file():
            lock_text = lock.read_text(encoding="utf-8")
            pid = int(lock_text.split("pid=", 1)[1].split()[0])
            try:
                os.kill(pid, 0)
            except OSError:
                pass
            else:
                raise RuntimeError(f"Refusing recovery while pid {pid} is alive")
        else:
            lock_text = "lock absent after external GPU/process interruption"
        archive_root = V3 / "interrupted_attempts"
        archive_root.mkdir(exist_ok=True)
        attempt = 1
        while (archive_root / f"{args.key}_attempt{attempt}").exists():
            attempt += 1
        archive = archive_root / f"{args.key}_attempt{attempt}"
        lock.unlink(missing_ok=True)
        shutil.move(str(source), str(archive))
        source.mkdir(parents=True)
    shutil.copytree(archive / "checkpoint-50", source / "checkpoint-50", dirs_exist_ok=True)
    groups = filtered_jsonl(archive / "groups.jsonl", source / "groups.jsonl",
                            lambda row: row["global_step"] < 50)
    phases = filtered_jsonl(archive / "phase_times.jsonl", source / "phase_times.jsonl",
                            lambda row: row["step"] < 50)
    assert groups == 100 and phases == 100
    plan = {
        "checkpoint": "checkpoint-50",
        "resume_global_step": 50,
        "retained_groups": groups,
        "retained_phase_events": phases,
        "interrupted_attempt_archive": str(archive.relative_to(ROOT)),
        "stale_lock": lock_text,
        "prepared_unix": time.time(),
        "reason": (
            "Completed parameters invalidated because raw group evidence was incomplete; "
            "resume from validated checkpoint"
            if args.invalidate_completed_evidence
            else "Training process absent without a completed endpoint; resume from validated checkpoint"
        ),
    }
    (source / "resume_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
