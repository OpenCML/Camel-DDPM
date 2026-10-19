"""Read-only schema-2 importer. Active training uses native Camel snapshots."""
import ast
import json
import math
from pathlib import Path
import struct
from text_image_runtime import digest

TREES = ("model", "adam_m", "adam_v", "ema")


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


