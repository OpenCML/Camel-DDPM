# Remote GPU training

The Windows machine is used only for editing, SSH and retrieving results.
The current server deployment is `88zjwei:/home/zjwei/projects/Camel-DDPM`.
The server has several RTX 3090 devices; select an idle physical device before
starting a run. Do not terminate unrelated jobs or modify other experiment trees.

## Project-local Linux runtime

The Windows `.camel/` installation cannot run on Linux. The server deployment
builds a separate runtime from the recorded Camel source revision and the scoped
runtime patch. Build products stay in `runtime-build/Camel/`, then its complete
`out/latest/` distribution is installed into the remote project's `.camel/`.

For the prepared server toolchain:

```bash
cd /home/zjwei/projects/Camel-DDPM
CAMEL_TOOLCHAIN_ROOT=/home/zjwei/camel-eval/conda/envs/clang \
CAMEL_BUILD_TOOLS=/home/zjwei/camel-eval/venv-build/bin \
CUDA_PATH=/usr/local/cuda \
nice -n 10 bash tools/build-server.sh > runs/build-server.log 2>&1
```

The build helper requires an extracted compatible source tree, applied patches,
and installed npm dependencies. It invokes `conan install` and the standard
`npm run build`, uses four compilation workers by default, and disables Python
embedding because the model/data path is native Camel. It copies the toolchain's
required C++ runtime libraries into `.camel/libs/`. It refuses to overwrite an
existing project installation.

The Linux launchers explicitly preload the bundled C++/GCC runtime libraries
when present. The conda compiler embeds an absolute `DT_RPATH` that can take
precedence over `LD_LIBRARY_PATH`; preloading prevents it from selecting the
build environment's C++ runtime instead of the installed copies. CUDA driver and
NVRTC remain system dependencies. No installed executable is patched in place.

## Checks and short run

```bash
cd /home/zjwei/projects/Camel-DDPM
bash tools/prepare.sh
export CAMEL_HOME="$PWD/.camel"
export CAMEL_STD_LIB="$CAMEL_HOME/stdlib"
export CAMEL_PACKAGES="$PWD/src"
export LD_LIBRARY_PATH="$CAMEL_HOME/libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CAMEL_TILES_DEVICE=2
export DDPM_OUT="$PWD/runs/checks"
mkdir -p "$DDPM_OUT"
.camel/bin/camel tests/checks.cml std::macro std::fvm
.camel/bin/camel tests/gpu_checks.cml std::macro std::fvm
.camel/bin/camel tests/patch_checks.cml std::macro std::fvm
GPU=2 MODEL=patch-x0 WIDTH=64 STEPS=3 BATCH=4 SAMPLES=8 OUT="$PWD/runs/patch-smoke" bash run.sh
```

Use `CAMEL_TILES_DEVICE`/`GPU` to select a physical CUDA driver ordinal. Do not
assume that another framework's `CUDA_VISIBLE_DEVICES` remapping selects Tiles'
device. Each run records the physical index and GPU UUID.

## Training and monitoring

After numerical and short-run checks pass:

```bash
nohup env GPU=2 MODEL=patch-x0 WIDTH=64 STEPS=5000 BATCH=32 \
  OUT="$PWD/runs/patch-5000" \
  nice -n 10 bash run.sh > runs/patch-5000-launch.log 2>&1 < /dev/null &
echo $! > runs/patch-5000.pid
tail -f runs/patch-5000/train.log
```

The recorded PID identifies only this run. `status.txt` records launch and exit
status; `metrics.json` and checkpoint files appear after successful training;
the final sample checksum appears after sampling. A launcher PID alone is not
evidence of success. Inspect logs and generated samples before reporting results.

Resample from a completed checkpoint in a fresh process (no training):

```bash
OUT="$PWD/runs/patch-5000" bash tools/sample-server.sh
```

The helper reads the checkpoint's model and diffusion-step configuration without
evaluating it as shell code. `SAMPLES`, `SAMPLE_SEED` and `GPU` can be overridden.
It replaces that run's sample/trajectory files and writes a separate resample log
and checksum, so use the original seed to check deterministic reproduction.

The epsilon baseline remains available using `MODEL=epsilon WIDTH=512`. The new
`patch-x0` model uses a fixed width of 64 and predicts clean pixels with an x0-MSE
objective; its loss values are not directly comparable to epsilon-MSE.

No training command in this document launches a local Windows training process.
# Server data storage

All datasets and training artifacts on `88zjwei` live under `/mnt/data3/zjwei/Camel-DDPM-data/`:

- `wit/`: WIT metadata, indices and images.
- `coco2017/archives/`: official COCO image and annotation archives.
- `project-data/`: earlier MNIST and sealed experimental corpora.
- `project-runs/`: logs, model checkpoints and run artifacts.

The code and runtime installation remain at `/home/zjwei/projects/Camel-DDPM`. Its `data` and `runs` entries are symlinks to the corresponding directories above; their contents occupy `/mnt/data3`, not the home filesystem. Preserve those links when deploying or extracting source archives. The earlier data and checkpoints were migrated on 2026-10-19.

