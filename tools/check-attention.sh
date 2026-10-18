#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export CAMEL_HOME="$PROJECT_ROOT/.camel"
export CAMEL_STD_LIB="$CAMEL_HOME/stdlib"
export CAMEL_PACKAGES="$PROJECT_ROOT/src"
export CAMEL_TILES_DEVICE=${GPU:-2}
export LD_LIBRARY_PATH="$CAMEL_HOME/libs:${CUDA_PATH:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
if [[ -f "$CAMEL_HOME/libs/libstdc++.so.6" && -f "$CAMEL_HOME/libs/libgcc_s.so.1" ]]; then
    export LD_PRELOAD="$CAMEL_HOME/libs/libstdc++.so.6:$CAMEL_HOME/libs/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi
mkdir -p runs/checks
for check in attention_checks multihead_checks; do
    "$CAMEL_HOME/bin/camel" "tests/$check.cml" std::macro std::fvm 2>&1 | tee "runs/checks/$check.log"
done
