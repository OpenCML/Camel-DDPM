"""Consume the ten verified metadata files as the local transport publishes them."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from checkpoint_store import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=43200)
    args = parser.parse_args()
    root = args.root.resolve()
    deadline = time.monotonic() + args.wait_seconds
    progress = root / "index-progress.json"
    for index in range(10):
        source = root / "metadata" / f"wit_v1.train.all-{index:05d}-of-00010.tsv.gz"
        out = root / f"index-shard{index}-v1"
        reported = None
        while not (out / "summary.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("metadata transfer/index wait deadline exceeded; partial index folders are preserved")
            status = "waiting_for_active_index" if out.exists() else "waiting_for_metadata"
            if source.exists() and not out.exists():
                atomic_json(progress, {"shard": index, "completed_shards": index, "status": "indexing", "training_admitted_images": 0})
                command = [sys.executable, str(Path(__file__).with_name("index-wit.py")), "--metadata", str(source), "--out", str(out)]
                with (root / f"index-shard{index}.log").open("x") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                break
            if status != reported:
                atomic_json(progress, {"shard": index, "completed_shards": index, "status": status, "training_admitted_images": 0})
                print(f"SHARD {index} {status}", flush=True)
                reported = status
            time.sleep(10)
        print(f"SHARD {index} complete", flush=True)
    atomic_json(progress, {"completed_shards": 10, "status": "metadata_indexes_complete", "training_admitted_images": 0})


if __name__ == "__main__":
    main()
