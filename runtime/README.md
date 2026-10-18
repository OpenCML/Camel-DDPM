# Required Camel library fixes

The validated project-local installation was built from Camel `1555bf5b`
(development version 0.2.14) with the fixes in
[`camel-ddpm-primitives.patch`](camel-ddpm-primitives.patch).
That patch deliberately excludes unrelated changes in the source checkout.

It changes these library files in a compatible Camel source checkout, and adds
their regression cases, test plan and technical note:

- `modules/math/module.cpp`: f32/f64 scalar sqrt VJP, registered by math.
- `modules/fs/operators.cpp`: length-aware binary output and write/close error detection.
- `modules/tiles/src/compiler.cpp`: specialize tensor numel/shape using concrete
  invocation shapes; lower inverse permutations through the core layout instance;
  compose batched matrix pullbacks, including the shared-RHS FMA chain.
- `modules/tiles/src/matmul_kernel.cpp` and `schedule.cpp`: rank-2 and higher
  matrix products, core-defined batch broadcasting and independent batched CUDA
  tiles. Vector products remain unsupported; no precision permissions are added.
- `schedule.cpp`, `codegen.cpp`, `cuda.cpp`, new `mixed_gemm.cpp`/header and
  CMake targets: optional cuBLAS BF16 GEMM, selected only with explicit
  reassociate/change_precision/finite_math grants. FP32 boundaries and state,
  private stream-bound handle, no atomics/reduced-precision reduction, Ampere+.
  Strict and incompletely authorized products retain the original FP32 path.

Apply and build using Camel's normal build workflow, then install the collected
distribution into this project:

```powershell
# From the compatible Camel source checkout:
git apply --check D:\path\to\Camel-DDPM\runtime\camel-ddpm-primitives.patch
git apply D:\path\to\Camel-DDPM\runtime\camel-ddpm-primitives.patch
$env:CMAKE_BUILD_PARALLEL_LEVEL = '1'
npm run build

# From Camel-DDPM:
.\tools\install-runtime.ps1 -Distribution D:\path\to\Camel\out\latest
.\test.ps1
```

Read and obey the source checkout's AGENTS.md before modifying it. If these fixes
have already landed upstream, do not reapply the patch; install a distribution
with the fixes and run the numerical tests.

The patch includes regression cases and the test plan
`test/plans/feat/modules/ddpm_primitives.plan.toml` (21 NVM/FVM/JIT checks).
They cover scalar sqrt gradients, an
embedded-NUL/high-byte file round trip and CUDA source specialization of the
mean/layout gradients with changing input shapes and negative permutation axes. The numerical square-root rule is
tested in the differentiable positive domain. Batch cases also cover changed
batch sizes, shared projections, broadcast dimensions and partial tiles. Run
`camel --arg run_gpu=true test/cases/modules/tiles/batch_matmul.cml std::macro std::fvm`
to compare actual CUDA forward/gradient values and directional finite differences.

The batched runtime was built and verified on `88zjwei`; the workstation runtime
has not been rebuilt. The remote project keeps its previous runtime under
`.camel-pre-batched-92df61c/`. Build into a fresh directory with
`CAMEL_INSTALL_ROOT` when validating an update before activation.

The current BF16-enabled runtime was additionally checked against all 58 Tiles
regressions and the 21 DDPM primitive cases. The prior batched snapshot remains
at `.camel-pre-mixed-8e38813/`; no workstation build was run. Full-model
FP32-reference comparisons and server results are in
[`docs/results/server-latent-validation.md`](../docs/results/server-latent-validation.md).

No patched binaries or source checkout are needed after installation: `.camel/`
is a complete independent runtime snapshot. The installation manifest records
where that snapshot came from; it is provenance, not an execution dependency.

The optional `camel-device-trees.patch` records Camel commit `1b724238`:
`Device<Parameters>` normalizes struct/tuple fields, and `upload`/`download`
transfer complete parameter trees. It applies on top of the integrated DDPM
primitives and the October 18 training design. Skip it when that commit is
already present upstream. The server built this into independent
`.camel-device-tree-v1` / `.camel-device-tree-v2` installations; the default
`.camel` runtime was preserved. The new example requires this capability:
[`device-tree-step.cml`](../examples/device-tree-step.cml).
Actual model updates and checkpoint replay results are documented in
[`device-tree-training.md`](../docs/results/device-tree-training.md).
