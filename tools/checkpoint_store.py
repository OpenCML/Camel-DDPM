"""Inspect native Camel snapshots. Camel alone writes/publishes training checkpoints."""
import ast
import json
import math
import os
from pathlib import Path
import re
import struct
import uuid
from text_image_runtime import digest

TREES = ("model", "adam_m", "adam_v", "ema")
SCALARS = {"step": "int", "seed": "int", "data_cursor": "int",
           "dataset_sha256": "string", "config_sha256": "string"}


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def leaves(node, path=""):
    if node["path"] != path:
        raise ValueError("native checkpoint type path differs: " + path)
    if node["kind"] in ("struct", "tuple"):
        members = node["members"]
        names = [member["name"] for member in members]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tree member: " + path)
        result = {}
        for member in members:
            child = (path + "." if path else "") + member["name"]
            result.update(leaves(member["type"], child))
        return result
    return {path: node}


def validate(folder, shapes):
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("checkpoint manifest must be a regular file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != 1 or manifest["tree"]["kind"] != "struct":
        raise ValueError("unsupported native checkpoint schema")
    fields = leaves(manifest["tree"])
    expected = {tree + "." + field for tree in TREES for field in shapes} | set(SCALARS)
    if set(fields) != expected:
        raise ValueError("checkpoint parameter/control tree differs")
    hashes = {}
    for name, leaf in fields.items():
        extension = ".txt" if name in SCALARS else ".npy"
        filename = name + extension
        if leaf["file"] != filename:
            raise ValueError("invalid checkpoint filename: " + name)
        path = folder / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint leaf is not a regular file: " + filename)
        if path.stat().st_size != leaf["bytes"] or digest(path) != leaf["sha256"]:
            raise ValueError("checkpoint leaf checksum mismatch: " + filename)
        if name in SCALARS:
            if leaf["kind"] != SCALARS[name]:
                raise ValueError("checkpoint scalar type differs: " + name)
        else:
            shape = shapes[name.split(".", 1)[1]]
            if leaf["kind"] != "tensor" or leaf["dtype"] != "float32" or leaf["shape"] != list(shape):
                raise ValueError("checkpoint model dtype/shape differs: " + name)
            with path.open("rb") as stream:
                if stream.read(8) != b"\x93NUMPY\x01\x00":
                    raise ValueError("native writer must produce NPY v1")
                length = struct.unpack("<H", stream.read(2))[0]
                header = ast.literal_eval(stream.read(length).decode("latin-1"))
                if header["descr"] != "<f4" or header["fortran_order"] or header["shape"] != tuple(shape):
                    raise ValueError("NPY storage differs from the model schema: " + name)
                payload = stream.read()
                if len(payload) != math.prod(shape) * 4 or any(not math.isfinite(v) for (v,) in struct.iter_unpack("<f", payload)):
                    raise ValueError("invalid or non-finite model tensor: " + name)
        hashes[filename] = leaf["sha256"]
    return hashes


def scalar(folder, name):
    value = (folder / (name + ".txt")).read_text(encoding="utf-8")
    if SCALARS[name] == "int":
        parsed = int(value)
        if str(parsed) != value or parsed < 0:
            raise ValueError("invalid checkpoint control integer: " + name)
        return parsed
    return value


def read(run, shapes):
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["schema"] != 3:
        raise ValueError("legacy run requires explicit migration; use its archived source to reproduce it")
    candidates = []
    for path in (run / "checkpoints").iterdir():
        if re.fullmatch(r"step-[0-9]+", path.name):
            if path.is_symlink() or not path.is_dir():
                raise ValueError("checkpoint must be a native published directory")
            candidates.append((int(path.name[5:]), path))
    if not candidates:
        raise ValueError("no published checkpoint")
    step, folder = max(candidates)
    pointer_path = run / "latest.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text())
        if not re.fullmatch(r"step-[0-9]+", pointer["checkpoint"]):
            raise ValueError("invalid checkpoint pointer")
        if digest(run / "checkpoints" / pointer["checkpoint"] / "manifest.json") != pointer["manifest_sha256"]:
            raise ValueError("checkpoint pointer checksum mismatch")
    validate(folder, shapes)
    manifest = json.loads((folder / "manifest.json").read_text())
    data_hash = config["dataset_summary"]["manifest_sha256"]
    if (scalar(folder, "step") != step or scalar(folder, "data_cursor") != step or
        scalar(folder, "seed") != config["seed"] or scalar(folder, "config_sha256") != digest(run / "config.json") or
        scalar(folder, "dataset_sha256") != data_hash or manifest["dataset_manifest_sha256"] != data_hash):
        raise ValueError("checkpoint/config/data/seed/cursor identity differs")
    return folder, step
