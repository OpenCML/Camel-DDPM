"""Versioned, atomic FP32 parameter-tree snapshots for native training."""
import ast
import json
import math
import os
from pathlib import Path
import struct
import uuid
from text_image_runtime import digest

TREES = ("model", "adam_m", "adam_v", "ema")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def validate(folder, shapes):
    hashes = {}
    for tree in TREES:
        for field, shape in shapes.items():
            path = folder / tree / (field + ".npy")
            with path.open("rb") as stream:
                if stream.read(6) != b"\x93NUMPY":
                    raise ValueError("invalid tensor magic")
                version = stream.read(2)
                if version not in (b"\x01\x00", b"\x02\x00"):
                    raise ValueError("unsupported tensor format")
                size = 2 if version[0] == 1 else 4
                length = struct.unpack("<H" if size == 2 else "<I", stream.read(size))[0]
                if length > 65536:
                    raise ValueError("excessive tensor header")
                header = ast.literal_eval(stream.read(length).decode("latin-1"))
                if header["descr"] != "<f4" or header["fortran_order"] or header["shape"] != tuple(shape):
                    raise ValueError("tensor dtype/layout/shape mismatch: " + str(path))
                payload = stream.read()
                if len(payload) != math.prod(shape) * 4:
                    raise ValueError("truncated tensor")
                if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", payload)):
                    raise ValueError("non-finite tensor")
            hashes[str(path.relative_to(folder))] = digest(path)
    return hashes


def read(run, shapes):
    pointer = json.loads((run / "latest.json").read_text())
    name = pointer["checkpoint"]
    if not name.startswith("step-") or not name[5:].isdigit():
        raise ValueError("invalid checkpoint pointer")
    folder = run / "checkpoints" / name
    if digest(folder / "state.json") != pointer["state_sha256"]:
        raise ValueError("checkpoint state checksum mismatch")
    state = json.loads((folder / "state.json").read_text())
    if state["config_sha256"] != digest(run / "config.json") or state["sha256"] != validate(folder, shapes):
        raise ValueError("checkpoint/config/tensor checksum mismatch")
    return folder, state["step"]


def publish(run, stage, step, shapes):
    hashes = validate(stage, shapes)
    for name in hashes:
        with (stage / name).open("rb") as stream:
            os.fsync(stream.fileno())
    state = {"schema": 2, "step": step, "config_sha256": digest(run / "config.json"), "sha256": hashes}
    atomic_json(stage / "state.json", state)
    destination = run / "checkpoints" / ("step-" + str(step))
    if destination.exists():
        if json.loads((destination / "state.json").read_text()) != state or validate(destination, shapes) != hashes:
            raise ValueError("conflicting orphan checkpoint")
    else:
        os.replace(stage, destination)
    sync_directory(destination.parent)
    atomic_json(run / "latest.json", {"checkpoint": destination.name, "state_sha256": digest(destination / "state.json")})
    return destination
