"""Stream WIT metadata into a bounded-memory bilingual candidate index.

This produces candidates, never a training-ready corpus: individual image
licenses, image downloads, image-level grouping and caption review remain gates.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit
from text_image_runtime import digest
from checkpoint_store import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    csv.field_size_limit(4 * 1024 * 1024)
    db = sqlite3.connect(out / "candidates.sqlite")
    # This database is a rebuildable index of the checksummed source. WAL avoids
    # repeatedly syncing large rollback journals during random-key insertion.
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-65536")
    db.execute("CREATE TABLE pairs (identity TEXT PRIMARY KEY, image_id TEXT, language TEXT, record TEXT)")
    db.execute("CREATE INDEX images ON pairs(image_id)")
    seen, selected = 0, 0
    sources = []
    for path in args.metadata:
        sources.append({"path": str(path.resolve()), "sha256": digest(path)})
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            required = {"language", "image_url", "page_url", "original_height", "original_width"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError("unrecognized WIT metadata columns")
            for row in reader:
                seen += 1
                if seen % 100000 == 0:
                    print(f"PROGRESS scanned={seen} selected_before_dedup={selected}", flush=True)
                language = row["language"]
                if language not in ("en", "zh"):
                    continue
                url = row["image_url"]
                parsed = urlsplit(url)
                # Exclude language-wiki fair-use images. Commons is still only
                # a candidate source, not evidence of an individual license.
                if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org" or not parsed.path.startswith("/wikipedia/commons/"):
                    continue
                try:
                    height, width = int(row["original_height"]), int(row["original_width"])
                except ValueError:
                    continue
                if min(height, width) < 256 or max(height, width) / min(height, width) > 3:
                    continue
                field = "caption_reference_description" if row.get("caption_reference_description", "").strip() else "caption_alt_text_description"
                caption = row.get(field, "").strip()
                if not (4 if language == "zh" else 12) <= len(caption) <= 4096:
                    continue
                if row.get("mime_type") not in ("image/jpeg", "image/png", "image/webp"):
                    continue
                image_id = hashlib.sha256(url.encode()).hexdigest()
                identity = hashlib.sha256((url + "\0" + language + "\0" + caption).encode()).hexdigest()
                record = {"id": identity, "image_id": image_id, "image_url": url,
                    "page_url": row["page_url"], "language": language, "caption": caption,
                    "caption_field": field, "width": width, "height": height,
                    "source": "WIT", "source_metadata": path.name,
                    "license_status": "pending_individual_image_verification", "training_admitted": False}
                db.execute("INSERT OR IGNORE INTO pairs VALUES (?, ?, ?, ?)",
                    (identity, image_id, language, json.dumps(record, ensure_ascii=False)))
                selected += 1
                if selected % 1000 == 0:
                    db.commit()
        db.commit()
        print(f"INDEXED {path.name} scanned={seen} selected_before_dedup={selected}", flush=True)
    destination = out / "candidates.jsonl.gz"
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for (record,) in db.execute("SELECT record FROM pairs ORDER BY identity"):
                    text.write(record + "\n")
    languages = dict(db.execute("SELECT language, COUNT(*) FROM pairs GROUP BY language"))
    images = db.execute("SELECT COUNT(DISTINCT image_id) FROM pairs").fetchone()[0]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    summary = {"schema": 1, "status": "candidate_metadata_only", "scanned_rows": seen,
        "candidate_pairs_by_language": languages, "candidate_unique_image_urls": images,
        "downloaded_images": 0, "training_admitted_images": 0,
        "sources": sources, "candidates_sha256": digest(destination),
        "required_next_gates": ["individual image license and attribution", "image download and decode",
            "caption quality and diversity", "exact/perceptual duplicate grouping", "immutable train/evaluation shards"]}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
