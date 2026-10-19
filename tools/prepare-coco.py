"""Validate official COCO archives and seal RGB64 training data on the server.

Image decoding is a host data task; model computation remains native Camel.
Official validation images take precedence when exact duplicates cross splits.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from PIL import Image, ImageOps
from checkpoint_store import atomic_json
from text_image_runtime import digest


def prepare(args):
    storage = Path('/mnt/data3/zjwei/Camel-DDPM-data').resolve(strict=True)
    archives = args.archives.resolve(strict=True)
    out = args.out.resolve()
    if not archives.is_relative_to(storage) or not out.is_relative_to(storage):
        raise ValueError('COCO input and output must reside on the server data disk')
    if os.stat(storage).st_dev != os.stat('/mnt/data3').st_dev:
        raise ValueError('data storage is not on the expected mounted filesystem')
    if shutil.disk_usage(storage).free < 10 * 1024**3:
        raise ValueError('need at least 10 GiB free on the data disk')
    pending = out.with_name(out.name + '.pending')
    if out.exists() or pending.exists():
        raise ValueError('output or pending corpus already exists; use a fresh destination')
    pending.mkdir(parents=True)
    (pending / 'rgb64').mkdir()
    (pending / 'metadata').mkdir()
    sources = {}
    for name in ('annotations_trainval2017.zip', 'val2017.zip', 'train2017.zip'):
        provenance = json.loads((archives / (name + '.json')).read_text())
        checksum = digest(archives / name)
        if checksum != provenance['sha256']:
            raise ValueError('archive SHA-256 mismatch: ' + name)
        sources[name] = {'sha256': checksum, 'bytes': (archives / name).stat().st_size,
                         'source': provenance['source']}
        print('Verified archive SHA-256: ' + name, flush=True)
    counts, exclusions = Counter(), Counter()
    seen_raw, seen_rgb = set(), set()
    started = time.monotonic()
    with zipfile.ZipFile(archives / 'annotations_trainval2017.zip') as annotations, \
            (pending / 'manifest.jsonl').open('x', encoding='utf-8') as manifest, \
            (pending / 'exclusions.jsonl').open('x', encoding='utf-8') as rejected:
        bad = annotations.testzip()
        if bad:
            raise ValueError('annotation ZIP CRC failure: ' + bad)
        for official, split in (('val2017', 'validation'), ('train2017', 'train')):
            metadata = json.loads(annotations.read('annotations/captions_' + official + '.json'))
            licenses = {row['id']: row for row in metadata['licenses']}
            captions = defaultdict(list)
            for row in sorted(metadata['annotations'], key=lambda row: row['id']):
                text = ' '.join(row['caption'].split())
                if text:
                    captions[row['image_id']].append(text)
            atomic_json(pending / 'metadata' / (official + '.json'),
                        {'info': metadata['info'], 'licenses': metadata['licenses']})
            images = sorted(metadata['images'], key=lambda row: row['id'])
            if args.limit_per_split:
                images = images[:args.limit_per_split]
            with zipfile.ZipFile(archives / (official + '.zip')) as archive:
                # Consume members in physical ZIP order to avoid random disk seeks.
                images.sort(key=lambda row: archive.getinfo(official + '/' + row['file_name']).header_offset)
                for position, image in enumerate(images, 1):
                    member = official + '/' + image['file_name']
                    try:
                        if not captions[image['id']]:
                            raise ValueError('no nonempty caption')
                        # ZipFile.read checks the CRC of every consumed JPEG member.
                        original = archive.read(member)
                        checksum = hashlib.sha256(original).hexdigest()
                        with Image.open(io.BytesIO(original)) as decoded:
                            decoded.load()
                            if decoded.size != (image['width'], image['height']):
                                raise ValueError('annotation dimensions do not match decoded image')
                            rgb = ImageOps.fit(ImageOps.exif_transpose(decoded).convert('RGB'),
                                               (64, 64), method=Image.Resampling.LANCZOS).tobytes()
                        rgb_hash = hashlib.sha256(rgb).hexdigest()
                        reason = ('duplicate_original' if checksum in seen_raw else
                                  'duplicate_rgb64' if rgb_hash in seen_rgb else None)
                        if reason:
                            exclusions[reason] += 1
                            rejected.write(json.dumps({'member': member, 'reason': reason}) + '\n')
                            continue
                        license_info = licenses[image['license']]
                        text = '\n'.join(captions[image['id']])
                        relative = 'rgb64/' + checksum + '.rgb'
                        (pending / relative).write_bytes(rgb)
                        (pending / relative).with_suffix('.caption.txt').write_text(text, encoding='utf-8')
                        row = {'source': 'COCO2017', 'record_id': image['id'], 'sha256': checksum,
                               'image': str(archives / (official + '.zip')) + '::' + member,
                               'rgb64': relative, 'rgb64_sha256': rgb_hash, 'language': 'en',
                               'caption': text, 'captions': captions[image['id']],
                               'image_license': license_info,
                               'coco_url': image.get('coco_url'), 'flickr_url': image.get('flickr_url'),
                               'split': split, 'split_group': rgb_hash}
                        manifest.write(json.dumps(row, ensure_ascii=False) + '\n')
                        seen_raw.add(checksum)
                        seen_rgb.add(rgb_hash)
                        counts[split] += 1
                        counts['captions'] += len(captions[image['id']])
                    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                        # A corrupt/missing member must not silently shrink a sealed corpus.
                        raise RuntimeError('COCO import failed at ' + member) from error
                    if position % 1000 == 0 or position == len(images):
                        manifest.flush()
                        atomic_json(pending / 'progress.json', {'split': split, 'processed': position,
                                    'expected': len(images), 'admitted': dict(counts),
                                    'excluded': dict(exclusions), 'seconds': time.monotonic() - started})
                        print(f'{official} {position}/{len(images)} admitted={dict(counts)}', flush=True)
    if not counts['train'] or not counts['validation']:
        raise ValueError('both splits must be nonempty')
    summary = {'schema': 2, 'sealed': True, 'images': counts['train'] + counts['validation'],
               'train': counts['train'], 'validation': counts['validation'], 'captions': counts['captions'],
               'groups': len(seen_rgb), 'languages': ['en'], 'sources': sources,
               'manifest_sha256': digest(pending / 'manifest.jsonl'), 'excluded': dict(exclusions),
               'grouping': 'official splits; exact JPEG and RGB64 duplicates removed, validation first',
               'near_duplicate_check': False, 'limit_per_split': args.limit_per_split,
               'preprocessing': 'EXIF transpose, RGB, center crop, Lanczos 64x64',
               'image_zip_crc': 'all annotated image members verified while reading'}
    atomic_json(pending / 'summary.json', summary)
    pending.rename(out)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit-per-split', type=int, default=0)
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error('limit must be nonnegative')
    prepare(args)


if __name__ == '__main__':
    main()
