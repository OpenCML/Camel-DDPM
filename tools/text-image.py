"""Train/resume/sample the RGB64 integration model using native Camel CUDA.

Python validates files and publishes checkpoints; it never evaluates the model.
Fresh processes between checkpoint segments are intentional for recovery tests.
"""
import argparse
import ast
import fcntl
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import uuid

from text_image_runtime import PROJECT, digest, run_native

LEAVES = {"w1": (192, 64), "wt": (259, 64), "b1": (64, 64), "w2": (64, 64),
          "b2": (64, 64), "w3": (64, 192), "b3": (16, 64)}
TREES = ("model", "adam_m", "adam_v", "ema")


def sync_directory(path):
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def fingerprints():
    files = list((PROJECT / "src").glob("*.cml"))
    files += [PROJECT / ".camel/bin/camel"]
    files += [path for path in (PROJECT / ".camel/libs").glob("*.so*") if path.is_file()]
    return {str(path.relative_to(PROJECT)): digest(path) for path in sorted(files)}


def validate_arrays(folder):
    hashes = {}
    for tree in TREES:
        for leaf, shape in LEAVES.items():
            path = folder / tree / (leaf + ".npy")
            with path.open("rb") as stream:
                if stream.read(6) != b"\x93NUMPY":
                    raise ValueError("not an NPY file: " + str(path))
                version = stream.read(2)
                if version not in (b"\x01\x00", b"\x02\x00"):
                    raise ValueError("unsupported NPY version")
                length = struct.unpack("<H" if version[0] == 1 else "<I", stream.read(2 if version[0] == 1 else 4))[0]
                if length > 65536:
                    raise ValueError("excessive NPY header")
                header = ast.literal_eval(stream.read(length).decode("latin-1"))
                if header["shape"] != shape or header["descr"] != "<f4" or header["fortran_order"]:
                    raise ValueError("checkpoint tensor layout mismatch: " + str(path))
                remaining = path.stat().st_size - stream.tell()
                if remaining != math.prod(shape) * 4:
                    raise ValueError("truncated checkpoint: " + str(path))
                # Validate numerics without loading any framework or running a model.
                for (value,) in struct.iter_unpack("<f", stream.read()):
                    if not math.isfinite(value):
                        raise ValueError("non-finite checkpoint tensor: " + str(path))
            hashes[str(path.relative_to(folder))] = digest(path)
    return hashes


def read_checkpoint(run):
    pointer = json.loads((run / "latest.json").read_text())
    name = pointer["checkpoint"]
    if not name.startswith("step-") or not name.removeprefix("step-").isdigit():
        raise ValueError("invalid checkpoint pointer")
    folder = run / "checkpoints" / name
    if digest(folder / "state.json") != pointer["state_sha256"]:
        raise ValueError("checkpoint state checksum mismatch")
    state = json.loads((folder / "state.json").read_text())
    if validate_arrays(folder) != state["sha256"]:
        raise ValueError("checkpoint tensor checksum mismatch")
    if state["config_sha256"] != digest(run / "config.json"):
        raise ValueError("checkpoint/config mismatch")
    return folder, state["step"]


def snapshot_data(dataset, run):
    manifest = dataset / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    ids = []
    for row in rows:
        if row["split"] != "train":
            continue
        if row["image_license"].get("access") != "CC0" or row["metadata_license"].get("access") != "CC0":
            raise ValueError("pilot admission requires CC0 image and metadata")
        if digest(dataset / row["rgb64"]) != row["rgb64_sha256"]:
            raise ValueError("dataset RGB checksum mismatch")
        if (dataset / row["rgb64"]).with_suffix(".caption.txt").read_text(encoding="utf-8") != row["caption"]:
            raise ValueError("dataset caption mismatch")
        ids.append(row["sha256"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate training ids")
    content = "\n".join(ids) + "\n"
    index = run / "train_ids.txt"
    if index.exists() and index.read_text() != content:
        raise ValueError("training data order changed")
    if not index.exists():
        index.write_text(content)
    return digest(manifest), digest(index)


def train(args):
    run = args.out.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / "writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = args.dataset.resolve()
        manifest_hash, index_hash = snapshot_data(dataset, run)
        gpu = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        config = {"schema": 1, "model": "rgb64-joint-v2", "objective": "x0_mse", "batch": args.batch,
                  "seed": args.seed, "rate": args.rate, "timesteps": args.timesteps,
                  "dataset": str(dataset), "manifest_sha256": manifest_hash,
                  "train_index_sha256": index_hash, "fingerprints": fingerprints(),
                  "rng": "tensor-and-host-reseed:seed+global-step+1", "ema_decay": 0.99,
                  "caption_dropout": 0.1, "gpu_uuid": gpu, "production_ready": False}
        existing = run / "config.json"
        if args.resume:
            if json.loads(existing.read_text()) != config:
                raise ValueError("resume source/runtime/data/hyperparameter identity changed")
            previous, start = read_checkpoint(run) if (run / "latest.json").exists() else (None, 0)
        else:
            if existing.exists():
                raise ValueError("run exists; explicitly use --resume or a new output")
            atomic_json(existing, config)
            previous, start = None, 0
        if args.steps <= start:
            raise ValueError("target steps must exceed the saved global step")
        (run / "checkpoints").mkdir(exist_ok=True)
        while start < args.steps:
            end = min(start + args.checkpoint_every, args.steps)
            stage = run / "checkpoints" / (".pending-" + uuid.uuid4().hex)
            stage.mkdir()
            run_native("src/train_text_image.cml", {
                "data": dataset, "index": run / "train_ids.txt", "out": stage,
                "resume": previous if previous else "", "start": start, "end": end,
                "timesteps": args.timesteps, "initial_seed": args.seed, "rate": args.rate, "batch_size": args.batch,
            }, stage / "train.log", args.gpu)
            hashes = validate_arrays(stage)
            for relative in hashes:
                with (stage / relative).open("rb") as stream:
                    os.fsync(stream.fileno())
            atomic_json(stage / "state.json", {"schema": 1, "step": end,
                "config_sha256": digest(existing), "sha256": hashes})
            destination = run / "checkpoints" / ("step-" + str(end))
            if destination.exists():
                # A crash after directory publication but before the pointer
                # update leaves an orphan; never overwrite it silently.
                recovered = json.loads((destination / "state.json").read_text())
                if recovered != json.loads((stage / "state.json").read_text()) or validate_arrays(destination) != hashes:
                    raise ValueError("conflicting orphan checkpoint")
            else:
                os.replace(stage, destination)
            sync_directory(run / "checkpoints")
            atomic_json(run / "latest.json", {"checkpoint": destination.name,
                "state_sha256": digest(destination / "state.json")})
            previous, start = destination, end
            print("Published complete checkpoint " + str(destination), flush=True)


def sample(args):
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    if config["fingerprints"] != fingerprints():
        raise ValueError("sampling source/runtime identity changed")
    checkpoint, step = read_checkpoint(run)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "sample.json", {"step": step, "prompt": args.prompt,
        "sample_seed": args.seed, "guidance": args.guidance, "weights": args.weights,
        "timesteps": config["timesteps"], "checkpoint": str(checkpoint),
        "config_sha256": digest(run / "config.json"), "production_ready": False})
    run_native("src/sample_text_image.cml", {"model": checkpoint / args.weights,
        "prompt": args.prompt, "out": out / "sample", "timesteps": config["timesteps"],
        "sample_seed": args.seed, "guidance": args.guidance}, out / "sample.log", args.gpu)
    atomic_json(out / "sha256.json", {name: digest(out / name) for name in ("sample.npy", "sample.ppm")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", type=Path, default=Path("data/openaccess-pilot"))
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--steps", type=int, default=20)
    training.add_argument("--checkpoint-every", type=int, default=10)
    training.add_argument("--batch", type=int, default=2)
    training.add_argument("--timesteps", type=int, default=100)
    training.add_argument("--seed", type=int, default=20261018)
    training.add_argument("--rate", type=float, default=0.001)
    training.add_argument("--resume", action="store_true")
    training.set_defaults(action=train)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--run", type=Path, required=True)
    sampling.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--prompt", required=True)
    sampling.add_argument("--seed", type=int, default=12345)
    sampling.add_argument("--guidance", type=float, default=1.0)
    sampling.add_argument("--weights", choices=TREES[::3], default="ema")
    sampling.set_defaults(action=sample)
    for command in (training, sampling):
        command.add_argument("--gpu", default="2")
    args = parser.parse_args()
    if args.command == "train" and (args.steps < 1 or args.batch < 1 or args.checkpoint_every < 1 or args.timesteps < 2 or not math.isfinite(args.rate) or args.rate <= 0 or args.seed < 0):
        parser.error("invalid training sizes, seed or rate")
    if args.command == "sample" and (not math.isfinite(args.guidance) or args.guidance < 0 or args.seed < 0):
        parser.error("invalid guidance or seed")
    args.action(args)


if __name__ == "__main__":
    main()
