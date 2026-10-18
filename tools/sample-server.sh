#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
: "${OUT:?Set OUT to a completed run directory}"
get_config() { awk -F= -v key="$1" '$1 == key {print substr($0, index($0,"=")+1); exit}' "$OUT/config.txt"; }
MODEL=$(get_config model)
TIMESTEPS=$(get_config timesteps)
GPU=${GPU:-$(get_config gpu)}
SAMPLES=${SAMPLES:-64}
SAMPLE_SEED=${SAMPLE_SEED:-$(get_config sample_seed)}
case "$MODEL" in epsilon|patch-x0) ;; *) echo 'Unknown checkpoint model' >&2; exit 2 ;; esac
for number in "$TIMESTEPS" "$SAMPLES"; do
    if [[ ! "$number" =~ ^[1-9][0-9]*$ ]]; then echo 'Invalid sampling sizes' >&2; exit 2; fi
done
if [[ ! "$GPU" =~ ^[0-9]+$ || ! "$SAMPLE_SEED" =~ ^-?[0-9]+$ ]]; then echo 'Invalid GPU/seed' >&2; exit 2; fi
export CAMEL_HOME="$PROJECT_ROOT/.camel"
export CAMEL_STD_LIB="$CAMEL_HOME/stdlib"
export CAMEL_PACKAGES="$PROJECT_ROOT/src"
export CAMEL_TILES_DEVICE="$GPU"
export DDPM_OUT="$OUT"
export LD_LIBRARY_PATH="$CAMEL_HOME/libs:${CUDA_PATH:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
"$CAMEL_HOME/bin/camel" --arg "timesteps=$TIMESTEPS" --arg "samples=$SAMPLES" --arg "sample_seed=$SAMPLE_SEED" --arg threads=1 --arg "model_kind=$MODEL" src/sample_gpu.cml std::macro std::fvm 2>&1 | tee "$OUT/resample.log"
sha256sum "$OUT/samples.npy" | tee "$OUT/resample.sha256"
