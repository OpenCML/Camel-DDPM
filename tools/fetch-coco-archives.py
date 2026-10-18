"""Download official COCO archives locally and transport verified bytes to the server.

No local image decoding or model execution. Archive/pixel validation stays on the server.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SOURCE = "https://s3.amazonaws.com/images.cocodataset.org/"
OFFICIAL_DOWNLOADS = "https://cocodataset.org/#download"
ARCHIVES = (
    "annotations/annotations_trainval2017.zip",
    "zips/val2017.zip",
    "zips/train2017.zip",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/coco2017/archives"))
    parser.add_argument("--ssh-host", default="88zjwei")
    parser.add_argument("--server-root", default="/mnt/data3/zjwei/Camel-DDPM-data/coco2017/archives")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.ssh_host) or not re.fullmatch(r"/[a-zA-Z0-9_./-]+", args.server_root):
        parser.error("invalid transport destination")
    args.out.mkdir(parents=True, exist_ok=True)
    curl = "curl.exe" if os.name == "nt" else "curl"
    for item in ARCHIVES:
        url = SOURCE + item
        name = item.split("/")[-1]
        destination = args.out / name
        head = subprocess.check_output([curl, "--http1.1", "--fail", "--silent", "--show-error", "--head",
            "--connect-timeout", "20", "--max-time", "60", url], text=True)
        lengths = re.findall(r"^content-length:\s*(\d+)", head, re.I | re.M)
        if not lengths:
            raise ValueError("official object has no content length")
        expected = int(lengths[-1])
        print(f"DOWNLOAD {name} expected_bytes={expected}", flush=True)
        if not destination.exists():
            partial = destination.with_suffix(destination.suffix + ".part")
            subprocess.run([curl, "--http1.1", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "20", "--retry", "5", "--retry-delay", "3", "--retry-all-errors",
                "--continue-at", "-", "--output", str(partial), url], check=True)
            if partial.stat().st_size != expected:
                raise ValueError("archive length differs from official object")
            partial.replace(destination)
        if destination.stat().st_size != expected:
            raise ValueError("cached archive length differs from official object")
        checksum = digest(destination)
        provenance = {"source": url, "official_downloads": OFFICIAL_DOWNLOADS,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(), "bytes": expected,
            "sha256": checksum, "head": head, "training_admitted": False}
        record = destination.with_suffix(destination.suffix + ".json")
        record.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        remote = args.server_root
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.ssh_host, f"mkdir -p {remote}"], check=True)
        print(f"TRANSFER {name} sha256={checksum}", flush=True)
        subprocess.run(["scp", str(destination), f"{args.ssh_host}:{remote}/{name}.incoming"], check=True)
        actual = subprocess.check_output(["ssh", "-o", "BatchMode=yes", args.ssh_host,
            f"sha256sum {remote}/{name}.incoming"], text=True).split()[0]
        if actual != checksum:
            raise ValueError("server archive checksum differs")
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.ssh_host,
            f"mv {remote}/{name}.incoming {remote}/{name}"], check=True)
        subprocess.run(["scp", str(record), f"{args.ssh_host}:{remote}/{record.name}"], check=True)
        print(f"VERIFIED {name} bytes={expected}", flush=True)


if __name__ == "__main__":
    main()
