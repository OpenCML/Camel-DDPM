"""Validate existing native training results and checkpoint rejection behavior."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import os


def interrupted_restart(trainer, resumed, expected, temporary):
    run = Path(temporary) / "interrupted"
    run.mkdir()
    for name in ("config.json", "train_ids.txt"):
        shutil.copy2(resumed / name, run / name)
    source = resumed / "checkpoints/step-10"
    checkpoint = run / "checkpoints/step-10"
    shutil.copytree(source, checkpoint)
    trainer.atomic_json(run / "latest.json", {"checkpoint": "step-10",
        "state_sha256": trainer.digest(checkpoint / "state.json")})
    config = json.loads((run / "config.json").read_text())
    command = ["python3", str(trainer.PROJECT / "tools/text-image.py"), "train",
        "--out", str(run), "--dataset", config["dataset"], "--resume",
        "--seed", str(config["seed"]), "--rate", str(config["rate"]),
        "--timesteps", str(config["timesteps"])]
    process = subprocess.Popen(command + ["--steps", "100", "--checkpoint-every", "90"],
        cwd=trainer.PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    interrupted = False
    try:
        for line in process.stdout:
            if line.startswith("RGB DDPM step=11 "):
                process.send_signal(signal.SIGINT)
                interrupted = True
                break
        process.communicate(timeout=30)
        assert interrupted and process.returncode != 0, "training was not interrupted"
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    _, saved = trainer.read_checkpoint(run)
    assert saved == 10, "partial checkpoint replaced the published checkpoint"
    result = subprocess.run(command + ["--steps", "20", "--checkpoint-every", "10"],
        cwd=trainer.PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    if result.returncode:
        raise AssertionError(result.stdout)
    final, step = trainer.read_checkpoint(run)
    assert step == 20 and trainer.validate_arrays(final) == expected, "interrupted restart differs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("--exercise-interruption", action="store_true")
    args = parser.parse_args()
    import sys
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("trainer", tools / "text-image.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    a, step_a = trainer.read_checkpoint(args.continuous.resolve())
    b, step_b = trainer.read_checkpoint(args.resumed.resolve())
    assert step_a == step_b
    hashes = trainer.validate_arrays(a)
    assert len(hashes) == 28 and hashes == trainer.validate_arrays(b), "restart trajectory differs"
    with tempfile.TemporaryDirectory() as temporary:
        if args.exercise_interruption:
            interrupted_restart(trainer, args.resumed.resolve(), hashes, temporary)
        copied = Path(temporary) / "run"
        shutil.copytree(args.resumed, copied)
        checkpoint, _ = trainer.read_checkpoint(copied)
        path = checkpoint / "adam_m/w1.npy"
        original = path.read_bytes()
        changed = bytearray(original)
        changed[-1] ^= 1
        path.write_bytes(changed)
        try:
            trainer.read_checkpoint(copied)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt tensor accepted")
        path.write_bytes(original)
        pointer = copied / "latest.json"
        value = json.loads(pointer.read_text())
        value["checkpoint"] = "../../outside"
        pointer.write_text(json.dumps(value))
        try:
            trainer.read_checkpoint(copied)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid checkpoint pointer accepted")
    print(json.dumps({"global_step": step_a, "byte_identical_tensor_files": len(hashes),
        "trees": list(trainer.TREES), "corruption_rejected": True, "invalid_pointer_rejected": True,
        "mid_segment_interruption_and_restart": args.exercise_interruption}))


if __name__ == "__main__":
    main()
