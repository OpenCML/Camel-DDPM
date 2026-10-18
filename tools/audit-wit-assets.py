"""Verify transported pixels and license declarations on the server.

This audit does not create a sealed training split or assert legal clearance.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from PIL import Image
from text_image_runtime import digest


def declared_license(metadata):
    def value(key):
        return str(metadata.get(key, {}).get("value", "")).strip()
    name, url = value("LicenseShortName"), value("LicenseUrl")
    parsed = urlsplit(url)
    versions = r"(?:1\.0|2\.0|2\.5|3\.0|4\.0)"
    match = re.fullmatch(r"CC (BY(?:-SA)?) (" + versions + ")", name)
    if match:
        expected = "/licenses/" + match[1].lower() + "/" + match[2]
        valid = parsed.scheme == "https" and parsed.hostname in ("creativecommons.org", "www.creativecommons.org") and parsed.path.rstrip("/") == expected and bool(value("Artist"))
    elif name == "CC0":
        valid = parsed.scheme == "https" and parsed.hostname == "creativecommons.org" and parsed.path.rstrip("/") == "/publicdomain/zero/1.0"
    elif name == "Public domain":
        # Verify Commons' explicit declaration is internally consistent and
        # retain the complete metadata for downstream provenance review.
        valid = value("License") == "pd" and value("Copyrighted").lower() == "false"
    else:
        valid = False
    return name, url, valid and not value("Restrictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    Image.MAX_IMAGE_PIXELS = 40_000_000
    counts, languages, licenses = Counter(), Counter(), Counter()
    manifest = out / "audit.jsonl"
    with manifest.open("x", encoding="utf-8") as stream:
        for folder in sorted(args.source.iterdir()):
            if not folder.is_dir() or not re.fullmatch(r"[a-f0-9]{64}", folder.name) or not (folder / "ready.json").exists():
                continue
            counts["transported_images"] += 1
            row = {"image_id": folder.name, "training_admitted": False}
            try:
                ready = json.loads((folder / "ready.json").read_text(encoding="utf-8"))
                if ready["candidate"]["image_id"] != folder.name or set(ready["sha256"]) != {"imageinfo.json", "image.bin"}:
                    raise ValueError("invalid transport inventory")
                for name, checksum in ready["sha256"].items():
                    if digest(folder / name) != checksum:
                        raise ValueError("transported file checksum mismatch")
                api = json.loads((folder / "imageinfo.json").read_text(encoding="utf-8"))
                pages = list(api["query"]["pages"].values())
                if len(pages) != 1:
                    raise ValueError("ambiguous source page")
                info = pages[0]["imageinfo"][0]
                with Image.open(folder / "image.bin") as image:
                    image.verify()
                with Image.open(folder / "image.bin") as image:
                    image.load()
                    width, height = image.size
                    if min(width, height) < 256 or max(width, height) / min(width, height) > 3:
                        raise ValueError("decoded dimensions outside policy")
                    gray = image.convert("RGB").convert("L").resize((9, 8))
                    pixels = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
                counts["decoded_images"] += 1
                name, url, allowed = declared_license(info.get("extmetadata", {}))
                licenses[name] += 1
                status = "license_declaration_verified" if allowed else "license_review_required"
                counts[status] += 1
                for pair in ready["candidate"]["captions"]:
                    languages[pair["language"]] += 1
                dhash = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))
                row.update(status=status, width=width, height=height, license=name, license_url=url,
                    file_sha256=ready["sha256"], candidate=ready["candidate"],
                    source_page=info.get("descriptionurl"), download_url=ready["download_url"],
                    derivative=ready["derivative"], extmetadata=info.get("extmetadata", {}), dhash64=format(dhash, "016x"))
            except (OSError, ValueError, KeyError, IndexError) as error:
                counts["rejected_or_invalid"] += 1
                row.update(status="rejected_or_invalid", error=str(error))
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"schema": 1, "status": "pixel_and_provenance_audit_only", **counts,
        "caption_pairs_by_language": dict(languages), "licenses": dict(licenses),
        "training_admitted_images": 0, "manifest_sha256": digest(manifest),
        "remaining_gates": ["caption quality and coverage", "cross-source duplicate grouping", "immutable training/evaluation shards"]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
