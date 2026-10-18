#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p data
for name in train-images-idx3-ubyte t10k-images-idx3-ubyte; do
    if [[ ! -f "data/$name.gz" ]]; then
        curl --fail --location --retry 3 --output "data/$name.download" "https://storage.googleapis.com/cvdf-datasets/mnist/$name.gz"
        mv -- "data/$name.download" "data/$name.gz"
    fi
done
printf '%s\n' 'f68b3c2dcbeaaa9fbdd348bbdeb94873  data/train-images-idx3-ubyte.gz' '9fb629c4189551a2d022fa330f9573f3  data/t10k-images-idx3-ubyte.gz' | md5sum --check
gzip -dc data/train-images-idx3-ubyte.gz > data/train-images-idx3-ubyte
gzip -dc data/t10k-images-idx3-ubyte.gz > data/t10k-images-idx3-ubyte
