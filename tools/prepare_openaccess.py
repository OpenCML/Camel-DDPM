"""Fetch a bounded, auditable CC0 image/text pilot on the training server.

This is transport and image decoding only. It never imports a model or weights.
Captions are museum metadata, not generated visual descriptions. This pilot is
deliberately not presented as a sufficient corpus for general text-to-image.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.parse

from PIL import Image, ImageOps

ROOT = "https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/"
AGENT = "Camel-DDPM/0.1 (https://github.com/OpenCML/Camel-DDPM)"
MAX_BYTES = 32 * 1024 * 1024


def fetch(url, limit=MAX_BYTES, redirected=False):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "smithsonian-open-access.s3-us-west-2.amazonaws.com", "ids.si.edu"
    }:
        raise ValueError("unrecognized source host")
    for attempt in range(2):
        try:
            # curl has a wall-clock deadline including name resolution and uses
            # the same configured network path as the server's shell tools.
            with tempfile.TemporaryDirectory() as temporary:
                headers = Path(temporary) / "headers.txt"
                result = subprocess.run([
                    "curl", "--fail", "--silent", "--show-error",
                    "--dump-header", str(headers), "--proto", "=https",
                    "--connect-timeout", "8", "--max-time", "60",
                    "--max-filesize", str(limit), "--user-agent", AGENT, url,
                ], capture_output=True, timeout=65, check=False)
                header_lines = headers.read_text(encoding="latin-1").splitlines() if headers.exists() else []
            status = 0
            location = None
            for line in header_lines:
                if line.startswith("HTTP/"):
                    status = int(line.split()[1])
                    location = None
                elif line.lower().startswith("location:"):
                    location = line.split(":", 1)[1].strip()
            if result.returncode:
                raise OSError(result.stderr.decode("utf-8", errors="replace").strip())
            if 300 <= status < 400:
                if not location:
                    raise ValueError("redirect omitted location")
                # The published IDS endpoint redirects to the same public S3
                # bucket. Validate the new host before following it.
                destination = urllib.parse.urljoin(url, location)
                if redirected or not destination.startswith("https://smithsonian-open-access.s3-us-west-2.amazonaws.com/media/"):
                    raise ValueError("unexpected image redirect")
                return fetch(destination, limit, redirected=True)
            if len(result.stdout) > limit:
                raise ValueError("source exceeds configured size limit")
            return result.stdout
        except (OSError, ValueError, subprocess.TimeoutExpired):
            if attempt == 1:
                raise
            time.sleep(attempt + 1)


def atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def captions(record):
    content = record.get("content", {})
    free = content.get("freetext", {})
    title = record.get("title", "").strip()
    topics = [item["content"].replace("\\", ", ") for item in free.get("topic", [])]
    medium = [item["content"] for item in free.get("physicalDescription", [])
              if item.get("label") == "Medium"]
    parts = [title] + topics[:8] + medium[:1]
    return ". ".join(part for part in parts if part)


def candidates(record):
    desc = record.get("content", {}).get("descriptiveNonRepeating", {})
    # Metadata openness alone never establishes the image's license.
    if desc.get("metadata_usage", {}).get("access") != "CC0":
        return
    caption = captions(record)
    if not caption or len(caption) < 8:
        return
    for media in desc.get("online_media", {}).get("media", []):
        if media.get("type") != "Images" or media.get("usage", {}).get("access") != "CC0":
            continue
        url = media.get("content")
        if not url:
            continue
        yield media, url, caption, desc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/openaccess-pilot"))
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--shards-per-unit", type=int, default=1)
    parser.add_argument("--image-max", type=int, default=512)
    parser.add_argument("--units", nargs="+", default=["saam", "nasm"])
    args = parser.parse_args()
    if args.limit < 1 or args.shards_per_unit < 1 or args.image_max < 512:
        parser.error("limits must be positive")
    out = args.out
    for folder in ("metadata", "images", "rgb64"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()] if manifest.exists() else []
    seen = {row["media_id"] for row in rows}
    hashes = {row["sha256"] for row in rows}
    # Verify cached data rather than silently trusting an interrupted prior run.
    for row in rows:
        for key, digest in (("image", "sha256"), ("rgb64", "rgb64_sha256")):
            if hashlib.sha256((out / row[key]).read_bytes()).hexdigest() != row[digest]:
                raise ValueError("cached artifact checksum mismatch: " + row[key])
        caption_file = out / "rgb64" / (row["sha256"] + ".caption.txt")
        if not caption_file.exists():
            atomic(caption_file, row["caption"].encode("utf-8"))
        elif caption_file.read_text(encoding="utf-8") != row["caption"]:
            raise ValueError("cached caption mismatch")
    errors = out / "rejections.jsonl"
    inspected = 0
    with manifest.open("a", encoding="utf-8") as accepted, errors.open("a", encoding="utf-8") as rejected:
        # Interleave units so a small pilot is not entirely one museum's images.
        indexes = []
        for unit in args.units:
            if not unit.isalnum():
                parser.error("unit must be alphanumeric")
            index_url = ROOT + unit.lower() + "/index.txt"
            print("INDEX " + index_url, flush=True)
            index_cache = out / "metadata" / (unit.lower() + "_index.txt")
            if not index_cache.exists():
                atomic(index_cache, fetch(index_url))
            index = index_cache.read_text(encoding="utf-8").splitlines()
            indexes.append(index[:args.shards_per_unit])
        for group in zip(*indexes):
            records = []
            for source in group:
                if not source.startswith(ROOT):
                    raise ValueError("unexpected metadata location")
                cached = out / "metadata" / (source.removeprefix(ROOT).replace("/", "_"))
                if not cached.exists():
                    print("METADATA " + source, flush=True)
                    atomic(cached, fetch(source))
                records.append((source, cached.read_text(encoding="utf-8").splitlines()))
            maximum = max(len(lines) for _, lines in records)
            for index in range(maximum):
                for source, lines in records:
                    if index >= len(lines) or len(rows) >= args.limit:
                        continue
                    record = json.loads(lines[index])
                    inspected += 1
                    for media, url, caption, desc in candidates(record):
                        if media["id"] in seen or len(rows) >= args.limit:
                            continue
                        try:
                            published_url = url
                            url += ("&" if "?" in url else "?") + "max=" + str(args.image_max)
                            payload = fetch(url)
                            digest = hashlib.sha256(payload).hexdigest()
                            seen.add(media["id"])
                            if digest in hashes:
                                continue
                            with Image.open(io.BytesIO(payload)) as image:
                                if image.format != "JPEG":
                                    raise ValueError("pilot currently accepts JPEG only")
                                image.load()
                                image = ImageOps.exif_transpose(image).convert("RGB")
                                width, height = image.size
                                if min(width, height) < 256 or max(width, height) / min(width, height) > 3:
                                    raise ValueError("resolution or aspect ratio rejected")
                                # Keep original pixels; this square view is only the 64px pipeline fixture.
                                rgb = ImageOps.fit(image, (64, 64), method=Image.Resampling.LANCZOS).tobytes()
                            row = {
                                "schema": 1, "source": "smithsonian-open-access", "language": "en",
                                "record_id": desc.get("record_ID"), "media_id": media["id"],
                                "unit": record.get("unitCode"), "caption": caption,
                                "caption_kind": "museum-metadata", "record_url": desc.get("record_link"),
                                "metadata_url": source, "metadata_record_hash": record.get("hash"),
                                "published_image_url": published_url, "image_url": url,
                                "image_variant": "delivery-service-max-" + str(args.image_max),
                                "image_license": media["usage"],
                                "metadata_license": desc["metadata_usage"],
                                "credit": record.get("content", {}).get("freetext", {}).get("creditLine", []),
                                "width": width, "height": height, "sha256": digest,
                                "image": "images/" + digest + ".jpg", "rgb64": "rgb64/" + digest + ".rgb",
                                "rgb64_sha256": hashlib.sha256(rgb).hexdigest(),
                                "split": "validation" if int(digest[:8], 16) % 20 == 0 else "train",
                                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            }
                            atomic(out / row["image"], payload)
                            atomic(out / row["rgb64"], rgb)
                            atomic(out / "rgb64" / (digest + ".caption.txt"), caption.encode("utf-8"))
                            accepted.write(json.dumps(row, ensure_ascii=False) + "\n")
                            accepted.flush()
                            rows.append(row)
                            hashes.add(digest)
                            print(json.dumps({"accepted": len(rows), "unit": row["unit"], "title": record.get("title")}), flush=True)
                        except (OSError, ValueError, subprocess.TimeoutExpired, Image.DecompressionBombError) as error:
                            rejected.write(json.dumps({"media_id": media["id"], "url": url, "error": str(error)}) + "\n")
                            rejected.flush()
                if len(rows) >= args.limit:
                    break
    summary = {"schema": 1, "images": len(rows), "requested_limit": args.limit,
               "limit_reached": len(rows) >= args.limit, "inspected_this_run": inspected,
               "train": sum(row["split"] == "train" for row in rows),
               "validation": sum(row["split"] == "validation" for row in rows),
               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "pretrained_models_used": False, "purpose": "pipeline pilot, not production corpus"}
    atomic(out / "summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary), flush=True)
    if not rows:
        raise SystemExit("No usable images acquired")


if __name__ == "__main__":
    main()
