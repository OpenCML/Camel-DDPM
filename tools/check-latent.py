"""Run complete-model numerical checks on the selected server GPU."""
import argparse
import json
from pathlib import Path
from text_image_runtime import PROJECT, run_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mixed_log = out / "mixed.log"
    run_native("tests/mixed_checks.cml", {"run_gpu": "true"}, mixed_log, args.gpu)
    plans = [json.loads(line) for line in mixed_log.read_text().splitlines() if line.startswith("{\"schema\"")]
    implementations = [[stage["implementation"] for stage in plan["stages"]] for plan in plans]
    assert implementations == [["cublas_bf16"], ["standard"], ["standard"]], implementations
    full = out / "full"
    full.mkdir()
    run_native("tests/full_model_checks.cml", {"out": full}, out / "full.log", args.gpu)
    print("LATENT NUMERICAL CHECKS PASS")


if __name__ == "__main__":
    main()
