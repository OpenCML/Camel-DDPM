"""Server-only training, recovery, evaluation and sampling for native latent diffusion.

All learned computation is Camel. This launcher manages files and processes.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
from checkpoint_store import atomic_json, read, validate
from model_schema import shapes, count
from text_image_runtime import PROJECT, digest, run_native


def fingerprints():
    files = list((PROJECT / "src").glob("*.cml"))
    files += [PROJECT / ".camel/bin/camel", Path(__file__).resolve(), PROJECT / "tools/model_schema.py",
              PROJECT / "tools/checkpoint_store.py", PROJECT / "tools/text_image_runtime.py"]
    files += [p for p in (PROJECT / ".camel/libs").glob("*.so*") if p.is_file()]
    files += list((PROJECT / ".camel/stdlib").glob("*.cmo"))
    files += list((PROJECT / ".camel/stdlib").glob("*.cml"))
    return {str(p.relative_to(PROJECT)): digest(p) for p in sorted(files)}


def kind(stage):
    return "AE" if stage == "vae" else "LD"


def indexes(dataset, run):
    summary = json.loads((dataset / "summary.json").read_text())
    if not summary.get("sealed"):
        raise ValueError("train only an immutable corpus created with seal-corpus.py")
    manifest = dataset / "manifest.jsonl"
    if digest(manifest) != summary["manifest_sha256"]:
        raise ValueError("sealed manifest was modified")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    groups = {"train": set(), "validation": set()}
    for row in rows:
        groups[row["split"]].add(row["split_group"])
        if digest(dataset / row["rgb64"]) != row["rgb64_sha256"]:
            raise ValueError("training pixels were modified")
        if (dataset / row["rgb64"]).with_suffix(".caption.txt").read_text(encoding="utf-8") != row["caption"]:
            raise ValueError("training caption was modified")
    if groups["train"] & groups["validation"]:
        raise ValueError("image group crosses the evaluation boundary")
    for split in groups:
        ids = [row["sha256"] for row in rows if row["split"] == split]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("empty or duplicate split")
        text = "\n".join(ids) + "\n"
        path = run / (split + "_ids.txt")
        if path.exists() and path.read_text() != text:
            raise ValueError("data order changed")
        if not path.exists(): path.write_text(text)
    return summary


def autoencoder_reference(path):
    path = path.resolve()
    config = json.loads((path / "config.json").read_text())
    if config["schema"] != 3 or config["stage"] != "vae" or config["fingerprints"] != fingerprints():
        raise ValueError("autoencoder has incompatible source/runtime identity")
    checkpoint, step = read(path, shapes("AE"))
    return {"run": str(path), "checkpoint": str(checkpoint), "step": step,
        "manifest_sha256": digest(checkpoint / "manifest.json"), "config_sha256": digest(path / "config.json")}


def data_path(path):
    storage = Path('/mnt/data3/zjwei/Camel-DDPM-data').resolve(strict=True)
    target = path.resolve()
    if not target.is_relative_to(storage):
        raise ValueError('training data and outputs must be on /mnt/data3: ' + str(target))
    return target


def training_config(args, summary, ae=None, migration=None):
    gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    config = {'schema': 3, 'architecture': 'latent64-transformer-v1', 'stage': args.stage,
        'kind': kind(args.stage), 'parameters': count(kind(args.stage)), 'dataset': str(args.dataset.resolve()),
        'dataset_summary': summary, 'seed': args.seed, 'rate': args.rate, 'batch': args.batch,
        'timesteps': args.timesteps, 'autoencoder': ae, 'gpu_uuid': gpu_uuid,
        'precision': 'BF16 GEMM / FP32 boundaries, master weights, reductions and Adam',
        'rng': 'seed+global-step+1', 'gradient_clip_l2': 1.0, 'ema_decay': 0.999,
        'checkpoint_format': 'native-tree-v1', 'fingerprints': fingerprints(), 'production_ready': False}
    if migration is not None:
        config['migration'] = migration
    return config


def train(args):
    run = data_path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'writer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = data_path(args.dataset)
        summary = indexes(dataset, run)
        ae = autoencoder_reference(args.autoencoder) if args.stage == 'diffusion' else None
        path = run / 'config.json'
        existing = json.loads(path.read_text()) if path.exists() else None
        config = training_config(args, summary, ae, existing.get('migration') if existing and args.resume else None)
        if args.resume:
            if existing != config:
                raise ValueError('resume source/runtime/data/autoencoder/hyperparameter identity changed')
            published = list((run / 'checkpoints').glob('step-*'))
            previous, start = read(run, shapes(config['kind'])) if published else (None, 0)
        else:
            if existing is not None:
                raise ValueError('run exists; use --resume or a fresh output')
            atomic_json(path, config)
            previous, start = None, 0
        if args.steps <= start:
            raise ValueError('target must exceed the saved global step')
        (run / 'checkpoints').mkdir(exist_ok=True)
        parameters = {'data': dataset, 'index': run / 'train_ids.txt', 'out': run,
            'resume': previous if previous else '', 'start': start, 'end': args.steps,
            'checkpoint_every': args.checkpoint_every, 'batch': args.batch, 'initial_seed': args.seed,
            'rate': args.rate, 'dataset_sha': summary['manifest_sha256'], 'config_sha': digest(path)}
        if ae:
            parameters.update(ae_path=ae['checkpoint'], timesteps=args.timesteps)
        log = run / ('train-' + str(start) + '-' + str(args.steps) + '-' + str(time.time_ns()) + '.log')
        execution = run_native('src/train_vae.cml' if args.stage == 'vae' else 'src/train_latent.cml',
            parameters, log, args.gpu, timeout=args.timeout or None)
        checkpoint, step = read(run, shapes(config['kind']))
        if step != args.steps:
            raise ValueError('Camel did not publish the target training step')
        atomic_json(log.with_suffix('.timing.json'), {**execution, 'updates': step - start,
            'native_processes': 1, 'includes_initialization_compile_and_checkpoint_write': True})
        print('Completed native training through ' + str(checkpoint), flush=True)

def load_run(path):
    path = path.resolve()
    config = json.loads((path / "config.json").read_text())
    if config["schema"] != 3:
        raise ValueError("legacy run requires explicit migration or its archived source/runtime")
    if config["fingerprints"] != fingerprints():
        raise ValueError("source/runtime identity changed")
    checkpoint, step = read(path, shapes(config["kind"]))
    return path, config, checkpoint, step


def evaluate(args):
    run, config, checkpoint, step = load_run(args.run)
    out = data_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    ae = config["autoencoder"]
    if ae:
        if digest(Path(ae["checkpoint"]) / "manifest.json") != ae["manifest_sha256"]:
            raise ValueError("autoencoder checkpoint changed")
        read(Path(ae["run"]), shapes("AE"))
    atomic_json(out / "evaluation.json", {"run": str(run), "step": step, "weights": args.weights,
        "initialization_baseline": args.initial, "validation_ids_sha256": digest(run / "validation_ids.txt")})
    run_native("src/evaluate_latent.cml", {"stage": config["stage"], "model": "" if args.initial else str(checkpoint), "weights": args.weights,
        "ae_path": ae["checkpoint"] if ae else "", "data": config["dataset"],
        "index": run / "validation_ids.txt", "out": out, "count": args.count,
        "initial_seed": config["seed"], "timesteps": config["timesteps"]}, out / "evaluate.log", args.gpu, timeout=600)


def sample(args):
    run, config, checkpoint, step = load_run(args.run)
    if config["stage"] != "diffusion": raise ValueError("sampling requires a trained diffusion checkpoint")
    ae = config["autoencoder"]
    reference = Path(ae["checkpoint"])
    if digest(reference / "manifest.json") != ae["manifest_sha256"]:
        raise ValueError("autoencoder checkpoint changed")
    if digest(Path(ae["run"]) / "config.json") != ae["config_sha256"]:
        raise ValueError("autoencoder configuration changed")
    validate(reference, shapes("AE"))
    out = data_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "sample.json", {"run": str(run), "step": step, "weights": args.weights,
        "prompt": args.prompt, "seed": args.seed, "sampler": "DDIM eta=0", "steps": args.steps,
        "guidance": args.guidance, "config_sha256": digest(run / "config.json"), "production_ready": False})
    run_native("src/sample_latent.cml", {"model": str(checkpoint), "weights": args.weights, "ae_path": str(reference),
        "out": out, "prompt": args.prompt, "sample_seed": args.seed, "steps": args.steps,
        "timesteps": config["timesteps"], "guidance": args.guidance}, out / "sample.log", args.gpu, timeout=600)
    atomic_json(out / "sha256.json", {name: digest(out / name) for name in ("latent.npy", "sample.npy", "sample.ppm")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--stage", choices=("vae", "diffusion"), required=True)
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--autoencoder", type=Path)
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--checkpoint-every", type=int, default=100)
    training.add_argument("--batch", type=int, default=2)
    training.add_argument("--seed", type=int, default=20261018)
    training.add_argument("--rate", type=float, default=0.0001)
    training.add_argument("--timesteps", type=int, default=100)
    training.add_argument("--timeout", type=int, default=0, help="Total process deadline in seconds; 0 disables it")
    training.add_argument("--resume", action="store_true")
    training.set_defaults(action=train)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--count", type=int, default=16)
    evaluation.add_argument("--initial", action="store_true")
    evaluation.add_argument("--weights", choices=("model", "ema"), default="model")
    evaluation.set_defaults(action=evaluate)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--run", type=Path, required=True)
    sampling.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--prompt", required=True)
    sampling.add_argument("--seed", type=int, default=12345)
    sampling.add_argument("--steps", type=int, default=30)
    sampling.add_argument("--guidance", type=float, default=2.0)
    sampling.add_argument("--weights", choices=("model", "ema"), default="model")
    sampling.set_defaults(action=sample)
    for command in (training, evaluation, sampling): command.add_argument("--gpu", default="2")
    args = parser.parse_args()
    if args.command == "train":
        if min(args.steps, args.batch, args.checkpoint_every) < 1 or args.timeout < 0 or args.timesteps < 2 or args.seed < 0 or not math.isfinite(args.rate) or args.rate <= 0:
            parser.error("invalid training sizes, seed or rate")
        if args.stage == "diffusion" and not args.autoencoder: parser.error("diffusion needs --autoencoder")
    if args.command == "evaluate" and args.count < 1: parser.error("count must be positive")
    if args.command == "sample" and (args.steps < 2 or args.seed < 0 or not math.isfinite(args.guidance) or args.guidance < 0): parser.error("invalid sampler settings")
    if args.gpu not in ('1', '2', '6', '7'): parser.error('GPU has not been authorized')
    args.action(args)


if __name__ == "__main__":
    main()
