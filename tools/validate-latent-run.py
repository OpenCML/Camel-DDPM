"""Held-out evaluations and independently reloaded prompt/replay samples."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import latent
from text_image_runtime import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    common = dict(run=args.run, gpu=args.gpu, weights="model")
    for initial in (True, False):
        latent.evaluate(SimpleNamespace(**common, out=out / ("initial" if initial else "trained"), count=16, initial=initial))
    prompts = ("A bronze sculpture of a bear", "A landscape painting with trees", "一只站在树枝上的红色小鸟", "A bronze sculpture of a bear")
    for index, prompt in enumerate(prompts):
        latent.sample(SimpleNamespace(**common, out=out / ("sample-" + str(index)), prompt=prompt, seed=12345, steps=30, guidance=2.0))
    first = json.loads((out / "sample-0/sha256.json").read_text())
    replay = json.loads((out / "sample-3/sha256.json").read_text())
    assert first == replay, "independent reload with fixed prompt/seed diverged"
    # Inspection only: NumPy is not used for model execution or optimization.
    import numpy as np
    arrays = [np.load(out / ("sample-" + str(i)) / "sample.npy", allow_pickle=False) for i in range(3)]
    for index, array in enumerate(arrays):
        assert array.shape == (1, 3, 64, 64) and np.isfinite(array).all()
        z = np.load(out / ("sample-" + str(index)) / "latent.npy", allow_pickle=False)
        assert z.shape == (1, 64, 4) and np.isfinite(z).all()
    report = {"fixed_seed_replay_identical": True,
        "english_prompt_swap_mean_abs": float(np.abs(arrays[0] - arrays[1]).mean()),
        "chinese_prompt_swap_mean_abs": float(np.abs(arrays[0] - arrays[2]).mean()),
        "note": "Sensitivity does not establish semantic alignment or image quality.",
        "run_config_sha256": digest(args.run / "config.json")}
    (out / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
