#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT=${CAMEL_SOURCE_ROOT:-"$PROJECT_ROOT/runtime-build/Camel"}
INSTALL_ROOT=${CAMEL_INSTALL_ROOT:-"$PROJECT_ROOT/.camel"}
if [[ -e "$INSTALL_ROOT" ]]; then
    echo 'Refusing to replace an existing runtime; choose a fresh CAMEL_INSTALL_ROOT.' >&2
    exit 1
fi
: "${CAMEL_TOOLCHAIN_ROOT:?Set CAMEL_TOOLCHAIN_ROOT to the Clang conda environment}"
: "${CAMEL_BUILD_TOOLS:?Set CAMEL_BUILD_TOOLS to the bin directory containing cmake, ninja and conan}"
export CONDA_PREFIX="$CAMEL_TOOLCHAIN_ROOT"
export PATH="$CONDA_PREFIX/bin:$CAMEL_BUILD_TOOLS:$PATH"
for activation in "$CONDA_PREFIX"/etc/conda/activate.d/activate-*.sh; do
    source "$activation"
done
mkdir -p "$SOURCE_ROOT/.build-tools"
ln -sfn "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-clang" "$SOURCE_ROOT/.build-tools/clang"
ln -sfn "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-clang++" "$SOURCE_ROOT/.build-tools/clang++"
export PATH="$SOURCE_ROOT/.build-tools:${CUDA_PATH:-/usr/local/cuda}/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${CUDA_PATH:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}
cd "$SOURCE_ROOT"
printf '{"CAMEL_ENABLE_PYTHON":"OFF"}\n' > .camel-opts.json
conan install . --output-folder=build --build=missing --profile=clang23
npm run build
cp -a out/latest "$INSTALL_ROOT"
cp -L "$CONDA_PREFIX/lib/libstdc++.so.6" "$INSTALL_ROOT/libs/libstdc++.so.6"
cp -L "$CONDA_PREFIX/lib/libgcc_s.so.1" "$INSTALL_ROOT/libs/libgcc_s.so.1"
"$INSTALL_ROOT/bin/camel" --version
sha256sum "$INSTALL_ROOT/bin/camel" > "$INSTALL_ROOT/installation.sha256"
