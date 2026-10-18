"""Run native Camel RGB/text integration checks; no model execution in Python."""
import argparse
import json
from pathlib import Path

from text_image_runtime import run_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/openaccess-pilot"))
    parser.add_argument("--out", type=Path, default=Path("runs/text-image-checks"))
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    row = next(row for row in rows if row["split"] == "train")
    image = dataset / row["rgb64"]
    caption = image.with_suffix(".caption.txt")
    if not image.is_file() or not caption.is_file():
        raise SystemExit("Incomplete pilot artifacts; run prepare_openaccess.py again")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "fixture.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checks = [
        ("attention_checks", {}),
        ("text_data_checks", {"image": image, "caption": caption}),
        ("text_image_checks", {"image": image, "caption": caption, "out": out}),
    ]
    for name, arguments in checks:
        run_native("tests/" + name + ".cml", arguments, out / (name + ".log"), args.gpu, timeout=180)
    (out / "status.json").write_text(json.dumps({"checks": "passed", "trained_steps": 10,
        "training_fixture": row["sha256"], "production_ready": False}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
