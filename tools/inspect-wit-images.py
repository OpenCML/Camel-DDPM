"""Server-only codec contact sheet for visual review of transported images."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.audit.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["status"] != "rejected_or_invalid"][:16]
    if not rows:
        raise ValueError("no decoded images to inspect")
    sheet = Image.new("RGB", (4 * 192, ((len(rows) + 3) // 4) * 216), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        with Image.open(args.source / row["image_id"] / "image.bin") as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((188, 188))
            x, y = (index % 4) * 192, (index // 4) * 216
            sheet.paste(image, (x + (192 - image.width) // 2, y + (188 - image.height) // 2))
            languages = ",".join(sorted({pair["language"] for pair in row["candidate"]["captions"]}))
            draw.text((x + 4, y + 193), f"{index}: {languages} {row['image_id'][:8]}", fill="black")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)


if __name__ == "__main__":
    main()
