"""Download official image/provenance files locally and transfer verified batches.

No image decoding, resizing, model work or training admission happens here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(url, destination, maximum, host):
    parsed = urlsplit(url)
    allowed = (host,) if isinstance(host, str) else host
    if parsed.scheme != "https" or parsed.hostname not in allowed:
        raise ValueError("unexpected HTTPS source host")
    partial = destination.with_suffix(destination.suffix + ".part")
    command = ["curl.exe" if shutil.which("curl.exe") else "curl", "--http1.1", "--fail", "--silent", "--show-error",
        "--location", "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "8", "--max-time", "45",
        "--max-filesize", str(maximum), "--retry", "1", "--retry-all-errors",
        "--user-agent", "Camel-DDPM/0.1 (https://github.com/OpenCML/Camel-DDPM)", "--output", str(partial), url]
    subprocess.run(command, check=True)
    if partial.stat().st_size > maximum:
        raise ValueError("oversized source response")
    partial.replace(destination)


def transfer(args, root, ids):
    if not ids:
        return
    bundles = root / "bundles"
    bundles.mkdir(exist_ok=True)
    temporary = bundles / "pending.tar"
    subprocess.run(["tar", "-cf", str(temporary), "-C", str(root), *ids], check=True)
    checksum = digest(temporary)
    bundle = bundles / (checksum + ".tar")
    temporary.replace(bundle)
    remote = args.server_root
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.ssh_host, f"mkdir -p {remote}/bundles {remote}/images"], check=True)
    subprocess.run(["scp", str(bundle), f"{args.ssh_host}:{remote}/bundles/{bundle.name}.incoming"], check=True)
    response = subprocess.check_output(["ssh", "-o", "BatchMode=yes", args.ssh_host,
        f"sha256sum {remote}/bundles/{bundle.name}.incoming"], text=True)
    if response.split()[0] != checksum:
        raise ValueError("server bundle checksum mismatch")
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.ssh_host,
        f"mv {remote}/bundles/{bundle.name}.incoming {remote}/bundles/{bundle.name} && tar -xf {remote}/bundles/{bundle.name} -C {remote}/images"], check=True)
    print(f"TRANSFERRED images={len(ids)} bundle={bundle.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--ssh-host", default="88zjwei")
    parser.add_argument("--server-root", default="/mnt/data3/zjwei/Camel-DDPM-data/wit/assets")
    args = parser.parse_args()
    if min(args.limit, args.batch) < 1 or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.ssh_host) or not re.fullmatch(r"/[a-zA-Z0-9_./-]+", args.server_root):
        parser.error("invalid transport configuration")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pending, completed, failures, consecutive = [], 0, 0, 0
    for item in json.loads(args.queue.read_text(encoding="utf-8"))[:args.limit]:
        image_id = item["image_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", image_id):
            raise ValueError("invalid image identity")
        folder = root / image_id
        folder.mkdir(exist_ok=True)
        try:
            ready = folder / "ready.json"
            if ready.exists():
                record = json.loads(ready.read_text(encoding="utf-8"))
                if record["candidate"] != item or any(digest(folder / name) != value for name, value in record["sha256"].items()):
                    raise ValueError("cached candidate/files changed")
            else:
                fetch(item["api_url"], folder / "imageinfo.json", 4 * 1024 * 1024, "commons.wikimedia.org")
                api = json.loads((folder / "imageinfo.json").read_text(encoding="utf-8"))
                pages = list(api.get("query", {}).get("pages", {}).values())
                if len(pages) != 1 or len(pages[0].get("imageinfo", [])) != 1:
                    raise ValueError("missing or ambiguous Commons file")
                info = pages[0]["imageinfo"][0]
                if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                    raise ValueError("unsupported media type")
                url = info.get("thumburl", info["url"])
                fetch(url, folder / "image.bin", 32 * 1024 * 1024, ("upload.wikimedia.org", "thumb.wikimedia.org"))
                record = {"schema": 1, "candidate": item, "download_url": url,
                    "derivative": "thumburl" in info, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": {name: digest(folder / name) for name in ("imageinfo.json", "image.bin")}, "training_admitted": False}
                part = folder / "ready.json.part"
                part.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                part.replace(ready)
            pending.append(image_id)
            completed += 1
            consecutive = 0
            print(f"FETCHED {completed} {image_id}", flush=True)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            failures += 1
            consecutive += 1
            (folder / "failure.json").write_text(json.dumps({"error": str(error), "training_admitted": False}) + "\n")
            print(f"FAILED {image_id}: {error}", flush=True)
            if consecutive >= 5:
                transfer(args, root, pending)
                raise RuntimeError("stopping after five consecutive transport failures") from error
        else:
            # Transport failures abort the job; do not mislabel a downloaded
            # image as bad or keep accumulating untransferred batches.
            if len(pending) == args.batch:
                transfer(args, root, pending)
                pending = []
    transfer(args, root, pending)
    print(json.dumps({"downloaded_and_transferred": completed, "failed": failures, "training_admitted": 0}), flush=True)


if __name__ == "__main__":
    main()
