"""Compare actual native trajectories and reject corrupted checkpoint tensors."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from checkpoint_store import read
from model_schema import shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous", type=Path)
    parser.add_argument("resumed", type=Path)
    args = parser.parse_args()
    config = json.loads((args.continuous / "config.json").read_text())
    assert config == json.loads((args.resumed / "config.json").read_text())
    expected_shapes = shapes(config["kind"])
    continuous, step = read(args.continuous, expected_shapes)
    resumed, resumed_step = read(args.resumed, expected_shapes)
    assert step == resumed_step
    first = json.loads((continuous / "state.json").read_text())["sha256"]
    second = json.loads((resumed / "state.json").read_text())["sha256"]
    assert first == second, "native continuation diverged"
    with tempfile.TemporaryDirectory(prefix="camel-recovery-") as name:
        damaged = Path(name)
        shutil.copy2(args.continuous / "config.json", damaged)
        shutil.copy2(args.continuous / "latest.json", damaged)
        target = damaged / "checkpoints" / continuous.name
        shutil.copytree(continuous, target)
        tensor = target / next(iter(first))
        contents = bytearray(tensor.read_bytes())
        contents[-1] ^= 1
        tensor.write_bytes(contents)
        try:
            read(damaged, expected_shapes)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt tensor was accepted")
    print(f"RECOVERY PASS kind={config['kind']} step={step} identical_tensors={len(first)} corruption_rejected=true")


if __name__ == "__main__":
    main()
