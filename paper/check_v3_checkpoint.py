"""Validate an intermediate TRL/PEFT checkpoint without resuming or selecting it."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.run / f"checkpoint-{args.step}"
    assert checkpoint.is_dir(), f"Missing checkpoint directory: {checkpoint}"
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["global_step"] == args.step
    assert state["max_steps"] == args.max_steps

    history = state["log_history"]
    gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
    assert len(gradients) == args.step
    assert all(math.isfinite(value) for value in gradients)

    adapter_path = checkpoint / "adapter_model.safetensors"
    assert adapter_path.is_file(), f"Missing adapter weights: {adapter_path}"
    from safetensors import safe_open

    tensor_count = 0
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            assert tensor.isfinite().all().item(), f"Nonfinite tensor: {key}"
            tensor_count += 1
    assert tensor_count > 0

    required = [
        state_path,
        adapter_path,
        checkpoint / "adapter_config.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "rng_state.pth",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"Missing checkpoint files: {missing}"
    files = {
        str(path.relative_to(args.run)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    }
    report = {
        "run": str(args.run),
        "checkpoint": str(checkpoint),
        "step": args.step,
        "max_steps": state["max_steps"],
        "finite_gradient_records": len(gradients),
        "finite_adapter_tensors": tensor_count,
        "files": files,
        "scope": (
            "Engineering integrity check only. The checkpoint is not an analysis endpoint, "
            "and this report does not authorize outcome-based stopping or selection."
        ),
    }
    destination = args.output or checkpoint / "integrity_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
