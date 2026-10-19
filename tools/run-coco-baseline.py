"""Server-only COCO import, native VAE training and held-out comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import latent
from checkpoint_store import atomic_json
from text_image_runtime import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    storage = Path('/mnt/data3/zjwei/Camel-DDPM-data').resolve(strict=True)
    dataset, out = args.dataset.resolve(), args.out.resolve()
    if not all(path.is_relative_to(storage) for path in (dataset, out)):
        raise ValueError('dataset and run must be on the server data disk')
    if args.gpu not in ('1', '2', '6', '7'):
        raise ValueError('GPU has not been authorized for this experiment')
    if args.steps <= 0 or args.batch <= 0:
        raise ValueError('steps and batch must be positive')
    out.mkdir(parents=True, exist_ok=False)
    status = {'stage': 'preparing_data', 'production_ready': False,
              'dataset': str(dataset), 'steps': args.steps, 'batch': args.batch, 'gpu': args.gpu}
    def update(stage, **details):
        status.update(stage=stage, **details)
        atomic_json(out / 'status.json', status)
        print(json.dumps(status), flush=True)
    try:
        update('preparing_data')
        if not (dataset / 'summary.json').exists():
            subprocess.run([sys.executable, str(PROJECT / 'tools/prepare-coco.py'),
                            '--archives', str(storage / 'coco2017/archives'),
                            '--out', str(dataset)], check=True)
        update('training_vae')
        latent.train(SimpleNamespace(stage='vae', dataset=dataset, out=out / 'vae',
                     gpu=args.gpu, steps=args.steps, batch=args.batch, seed=20261019,
                     rate=0.0001, timesteps=100, autoencoder=None, checkpoint_every=100,
                     segment_timeout=3600, resume=False))
        update('evaluating_vae')
        for initial in (True, False):
            latent.evaluate(SimpleNamespace(run=out / 'vae', out=out / ('initial' if initial else 'trained'),
                            gpu=args.gpu, weights='model', count=128, initial=initial))
        before = json.loads((out / 'initial/metrics.json').read_text())
        after = json.loads((out / 'trained/metrics.json').read_text())
        update('complete', initial=before, trained=after,
               reconstruction_mse_improved=after['mse'] < before['mse'],
               note='Small RGB64 VAE baseline; visual review still required before diffusion training.')
    except BaseException as error:
        update('failed', error=repr(error))
        raise


if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    main()
