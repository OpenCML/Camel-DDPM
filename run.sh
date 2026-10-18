#!/usr/bin/env bash
# Explicit GPU execution; suitable for a remote nohup session.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
export CAMEL_HOME="$PROJECT_ROOT/.camel"
export CAMEL_STD_LIB="$CAMEL_HOME/stdlib"
export CAMEL_PACKAGES="$PROJECT_ROOT/src"
export LD_LIBRARY_PATH="$CAMEL_HOME/libs:${CUDA_PATH:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
if [[ -f "$CAMEL_HOME/libs/libstdc++.so.6" && -f "$CAMEL_HOME/libs/libgcc_s.so.1" ]]; then
    export LD_PRELOAD="$CAMEL_HOME/libs/libstdc++.so.6:$CAMEL_HOME/libs/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export CAMEL_TILES_DEVICE=${GPU:-2}
MODEL=${MODEL:-patch-x0}
STEPS=${STEPS:-5000}
WIDTH=${WIDTH:-64}
BATCH=${BATCH:-32}
SAMPLES=${SAMPLES:-64}
TIMESTEPS=${TIMESTEPS:-100}
RATE=${RATE:-0.001}
DELAY_MS=${DELAY_MS:-10}
SAMPLE_SEED=${SAMPLE_SEED:-12345}
case "$MODEL" in epsilon|patch-x0) ;; *) echo 'Unknown model' >&2; exit 2 ;; esac
if [[ "$MODEL" == patch-x0 && "$WIDTH" != 64 ]]; then echo 'patch-x0 width must be 64' >&2; exit 2; fi
for number in "$STEPS" "$WIDTH" "$BATCH" "$SAMPLES" "$TIMESTEPS"; do
    if [[ ! "$number" =~ ^[1-9][0-9]*$ ]]; then echo 'Sizes must be positive integers' >&2; exit 2; fi
done
if [[ "$TIMESTEPS" -lt 2 || "$BATCH" -gt 60000 || ! "$DELAY_MS" =~ ^[0-9]+$ ]]; then echo 'Invalid schedule, batch or delay' >&2; exit 2; fi
if [[ ! "$RATE" =~ ^[0-9]+([.][0-9]+)?$ || "$RATE" == 0 || "$RATE" == 0.0 ]]; then echo 'RATE must be a positive decimal' >&2; exit 2; fi
export DDPM_OUT=${OUT:-"$PROJECT_ROOT/runs/$(date +%Y%m%d-%H%M%S)"}
if [[ -e "$DDPM_OUT/config.txt" ]]; then echo 'Run exists; choose a fresh OUT' >&2; exit 2; fi
mkdir -p "$DDPM_OUT"
printf 'model=%s\nsteps=%s\nwidth=%s\nbatch=%s\ntimesteps=%s\nrate=%s\ngpu=%s\ndelay_ms=%s\nsample_seed=%s\nsamples=%s\n' "$MODEL" "$STEPS" "$WIDTH" "$BATCH" "$TIMESTEPS" "$RATE" "$CAMEL_TILES_DEVICE" "$DELAY_MS" "$SAMPLE_SEED" "$SAMPLES" > "$DDPM_OUT/config.txt"
"$CAMEL_HOME/bin/camel" --version >> "$DDPM_OUT/config.txt"
nvidia-smi -i "$CAMEL_TILES_DEVICE" --query-gpu=index,uuid,name,memory.total,memory.free --format=csv > "$DDPM_OUT/gpu.csv"
printf 'RUN start=%s host=%s pid=%s\n' "$(date -Iseconds)" "$(hostname)" "$$" | tee "$DDPM_OUT/status.txt"
trap 'code=$?; printf "RUN exit=%s end=%s\n" "$code" "$(date -Iseconds)" >> "$DDPM_OUT/status.txt"' EXIT
start=$SECONDS
"$CAMEL_HOME/bin/camel" --arg "steps=$STEPS" --arg "batch=$BATCH" --arg "width=$WIDTH" --arg "timesteps=$TIMESTEPS" --arg report=100 --arg threads=1 --arg "rate=$RATE" --arg "delay_ms=$DELAY_MS" --arg "model_kind=$MODEL" src/train_gpu.cml std::macro std::fvm 2>&1 | tee "$DDPM_OUT/train.log"
printf 'training_seconds=%s\n' "$((SECONDS-start))" > "$DDPM_OUT/timing.txt"
"$CAMEL_HOME/bin/camel" --arg "timesteps=$TIMESTEPS" --arg "samples=$SAMPLES" --arg "sample_seed=$SAMPLE_SEED" --arg threads=1 --arg "model_kind=$MODEL" src/sample_gpu.cml std::macro std::fvm 2>&1 | tee "$DDPM_OUT/sample.log"
sha256sum "$DDPM_OUT/samples.npy" | tee "$DDPM_OUT/samples.sha256"
