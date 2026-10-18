"""Optional PNG conversion only. The Camel workflow itself has no Python dependency."""
import argparse
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
args = parser.parse_args()
for pgm in args.run.glob('*.pgm'):
    with Image.open(pgm) as im:
        im.resize((im.width * 3, im.height * 3), Image.Resampling.NEAREST).save(pgm.with_suffix('.png'))
