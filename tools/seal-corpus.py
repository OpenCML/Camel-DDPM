"""Freeze a transport corpus with object/exact/near-duplicate grouped splits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image
from text_image_runtime import digest
from checkpoint_store import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if not json.loads((source / "summary.json").read_text())["limit_reached"]:
        raise ValueError("preparation has not reached its requested limit")
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a, b):
        parent[root(a)] = root(b)
    objects, exact, perceptual = {}, {}, []
    for i, row in enumerate(rows):
        if row["image_license"].get("access") != "CC0" or row["metadata_license"].get("access") != "CC0":
            raise ValueError("unlicensed pilot record")
        for key, checksum in (("image", "sha256"), ("rgb64", "rgb64_sha256")):
            if digest(source / row[key]) != row[checksum]:
                raise ValueError("source file checksum mismatch")
        key = (row["source"], row["record_id"])
        if key in objects: join(i, objects[key])
        objects[key] = i
        if row["sha256"] in exact: join(i, exact[row["sha256"]])
        exact[row["sha256"]] = i
        image = Image.frombytes("RGB", (64, 64), (source / row["rgb64"]).read_bytes()).convert("L").resize((9, 8))
        pixels = list(image.getdata())
        value = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))
        for j, previous in enumerate(perceptual):
            if (value ^ previous).bit_count() <= 3: join(i, j)
        perceptual.append(value)
        row["dhash64"] = format(value, "016x")
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row["sha256"])
    for i, row in enumerate(rows):
        group = hashlib.sha256(("camel-split-v1:" + min(groups[root(i)])).encode()).hexdigest()
        row.update(split_group=group, split="validation" if int(group[:8], 16) % 10 == 0 else "train")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation")}
    if not all(counts.values()):
        raise ValueError("sealed corpus needs nonempty training and validation splits")
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "rgb64", "metadata"):
        (out / folder).mkdir()
    for row in rows:
        for relative in (row["image"], row["rgb64"], row["rgb64"].removesuffix(".rgb") + ".caption.txt"):
            shutil.copy2(source / relative, out / relative)
    for metadata in (source / "metadata").iterdir():
        if metadata.is_file(): shutil.copy2(metadata, out / "metadata" / metadata.name)
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"schema": 2, "sealed": True, "images": len(rows), **counts,
        "groups": len(groups), "grouping": "record-id + exact-image + 64-bit-dhash-distance<=3",
        "manifest_sha256": digest(manifest), "source_manifest_sha256": digest(source / "manifest.jsonl"),
        "languages": sorted({row["language"] for row in rows})}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
