"""Explicitly import an archived schema-2 VAE into native Camel checkpoints.

No fingerprint exception is added to training or inference. The source remains immutable.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import latent
import legacy_checkpoint_store
from checkpoint_store import atomic_json, read, validate, TREES
from model_schema import shapes
from text_image_runtime import PROJECT, digest, run_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--legacy-project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', choices=('1', '2', '6', '7'), default='2')
    args = parser.parse_args()
    source, archived = args.source.resolve(), args.legacy_project.resolve()
    config_path = source / 'config.json'
    old = json.loads(config_path.read_text())
    if old['schema'] != 2 or old['stage'] != 'vae':
        raise ValueError('this importer accepts schema-2 VAE runs only')
    for name, expected in old['fingerprints'].items():
        candidate = (archived / name).resolve(strict=True)
        if not candidate.is_relative_to(archived) or digest(candidate) != expected:
            raise ValueError('archived source/runtime differs: ' + name)
    previous, step = legacy_checkpoint_store.read(source, shapes('AE'))
    out = latent.data_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dataset = latent.data_path(Path(old['dataset']))
    summary = latent.indexes(dataset, out)
    for split in ('train', 'validation'):
        if digest(source / (split + '_ids.txt')) != digest(out / (split + '_ids.txt')):
            raise ValueError('migration changed the admitted data order')
    if summary != old['dataset_summary']:
        raise ValueError('migration data version differs')
    migration = {'source_run': str(source), 'source_project': str(archived), 'source_step': step,
        'source_config_sha256': digest(config_path), 'source_state_sha256': digest(previous / 'state.json'),
        'importer_sha256': digest(Path(__file__)), 'source_fingerprints': old['fingerprints']}
    values = SimpleNamespace(stage='vae', dataset=dataset, seed=old['seed'], rate=old['rate'],
        batch=old['batch'], timesteps=old['timesteps'], gpu=args.gpu)
    config = latent.training_config(values, summary, migration=migration)
    atomic_json(out / 'config.json', config)
    run_native('src/migrate_latent.cml', {'model_kind': 'AE', 'legacy': previous, 'out': out,
        'step': step, 'initial_seed': old['seed'], 'dataset_sha': summary['manifest_sha256'],
        'config_sha': digest(out / 'config.json')}, out / 'migration.log', args.gpu)
    checkpoint, migrated_step = read(out, shapes('AE'))
    hashes = validate(checkpoint, shapes('AE'))
    equality = {tree + '.' + field: digest(previous / tree / (field + '.npy')) == hashes[tree + '.' + field + '.npy']
        for tree in TREES for field in shapes('AE')}
    if migrated_step != step or not all(equality.values()):
        raise ValueError('migration changed an NPY payload or training step')
    atomic_json(out / 'migration.json', {'source': migration, 'native_manifest_sha256': digest(checkpoint / 'manifest.json'),
        'step': step, 'all_tensor_files_byte_identical': True, 'tensor_files': len(equality),
        'production_ready': False})
    print('MIGRATION PASS step=' + str(step) + ' byte_identical_tensors=' + str(len(equality)))


if __name__ == '__main__':
    main()
