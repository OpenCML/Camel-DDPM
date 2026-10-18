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
  invocation shapes; lower inverse permutations through the core layout instance.

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
`test/plans/feat/modules/ddpm_primitives.plan.toml` (15 NVM/FVM/JIT checks).
They cover scalar sqrt gradients, an
embedded-NUL/high-byte file round trip and CUDA source specialization of the
mean/layout gradients with changing input shapes and negative permutation axes. The numerical square-root rule is
tested in the differentiable positive domain.

No patched binaries or source checkout are needed after installation: `.camel/`
is a complete independent runtime snapshot. The installation manifest records
where that snapshot came from; it is provenance, not an execution dependency.
