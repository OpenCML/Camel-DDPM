"""Codec-only contact sheets of native Camel outputs; no model computation."""
import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    images = [Image.open(path).convert("RGB") for path in args.inputs]
    canvas = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.resize((canvas.width * 4, canvas.height * 4), Image.Resampling.NEAREST).save(args.out)


if __name__ == "__main__":
    main()
