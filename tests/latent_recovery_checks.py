"""Compare actual native trajectories and reject corrupted checkpoint tensors."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from checkpoint_store import read, validate
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
    first = validate(continuous, expected_shapes)
    second = validate(resumed, expected_shapes)
    assert first == second, "native model/Adam/EMA/control state continuation diverged"
    timings = list(args.continuous.glob('train-*.timing.json'))
    assert len(timings) == 1 and json.loads(timings[0].read_text())['native_processes'] == 1
    assert len(list((args.continuous / 'checkpoints').glob('step-*'))) >= 2
    with tempfile.TemporaryDirectory(prefix="camel-recovery-", dir=args.continuous.resolve().parent) as name:
        damaged = Path(name)
        shutil.copy2(args.continuous / "config.json", damaged)
        shutil.copy2(args.continuous / "latest.json", damaged)
        target = damaged / "checkpoints" / continuous.name
        shutil.copytree(continuous, target)
        tensor = target / next(name for name in first if name.endswith('.npy'))
        contents = bytearray(tensor.read_bytes())
        contents[-1] ^= 1
        tensor.write_bytes(contents)
        try:
            read(damaged, expected_shapes)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt tensor was accepted")
        tensor.write_bytes(bytes(contents[:-1]) + bytes([contents[-1] ^ 1]))
        # A kill between directory publication and pointer replacement leaves a valid orphan.
        (damaged / 'latest.json').unlink()
        assert read(damaged, expected_shapes)[1] == step
        pending = damaged / 'checkpoints' / ('step-' + str(step + 1) + '.pending-interrupted')
        pending.mkdir()
        (pending / 'manifest.json').write_text('{')
        assert read(damaged, expected_shapes)[1] == step, 'partial directory became visible'
    print(f"RECOVERY PASS kind={config['kind']} step={step} identical_files={len(first)} single_process=true corruption_rejected=true orphan_recovered=true")


if __name__ == "__main__":
    main()
