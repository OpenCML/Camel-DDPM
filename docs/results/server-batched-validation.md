# Batched RGB/text CUDA validation — 2026-10-18

The `rgb64-joint-v2` integration network now uses four attention heads and independently sampled batch items. The model still has 54,464 randomly initialized parameters with shared projections; it is not a production DiT. All model computation, gradients, Adam, EMA and sampling remain Camel-native. Tests/builds ran on `88zjwei`, GPU2 (RTX3090, UUID `GPU-c10aaf4f-7c2c-3fa5-f47d-bb26db738a49`); nothing was trained or built on the workstation.

## Backend change

Tiles previously rejected every matmul with rank other than 2. It now uses the Tensor core's existing index maps for rank ≥2 inputs, including batch broadcasting. The tiled kernel preserves the declared ascending FP32 FMA chain. The shared rank-2 projection pullback flattens batch/row into one GEMM, matching the CPU algorithm without introducing an additional batch-sum rounding stage. No FP16, TF32, reassociation or CPU fallback is enabled.

CUDA checks cover rank3/rank4 products, changed batch size, singleton/unequal-rank broadcasting, shared left/right projections, non-tile-aligned dimensions, an independent integer-valued product, empty contraction and empty output batch. In the seven random forward/backward cases, CPU/GPU forward and parameter-gradient L1 differences were **zero**. Maximum directional finite-difference absolute error was `3.502790641505271e-4` (epsilon `1e-3`); finite-difference cancellation is not a GPU error estimate. The three-backend source-only plan adds no CUDA context requirement and passes **18/18** NVM/FVM/JIT cases.

Rank-1 vector products are still unsupported by Tiles. Batch and row tiles share CUDA grid Y and fail explicitly beyond its supported limit. This is a correctness extension of the FP32 kernel, not a claim of production throughput or multi-GPU support.

## Attention and integration

`attention.cml` provides last-axis stable softmax and a graph-visible VJP, scaled attention over `[batch, heads, tokens, head_width]`, and independently parameterized Q/K/V/output projections. Explicit layout attributes support different query/context lengths. The RGB fixture also splits its text/image mixing into four heads.

| Check | Result |
|---|---|
| Multi-head cross-attention, batch2, query length3 / context length5 | Gradient L1 difference `2.2035328584024683e-7`; directional error `6.469014806498308e-6` |
| Same attention with batch3 | Gradient L1 difference `2.5080407795030624e-7`; directional error `4.252542566973716e-6` |
| Distinct batch values with uniform attention logits | Isolation error `0` |
| Batched RGB patch/inverse mapping | Error `0` |
| Full RGB model CPU/GPU gradient L1 difference | `1.174548390088276e-6` |
| Text embedding gradient L1 | `0.09167366665401744`, nonzero |
| Fixed two-item numerical fitting, 10 steps | MSE `0.7749525785887353 → 0.06588361506788348` |
| Weight reload / changed text | Reload L1 error `0`; changed text alters output |
| Real-data batch4: continuous20 vs10 + restart +10 | All28 model/Adam-m/Adam-v/EMA files byte-identical |
| Mid-segment interruption after step11, resume checkpoint10 to20 | Same28 files byte-identical; corrupt files and invalid pointers rejected |
| Independent 100-step DDPM/CFG=2 sampling after both runs | Same NPY SHA-256 |
| Existing MNIST numerical, GPU Adam, permutation and patch checks | All pass |
| Existing Tiles backend regression plan | 58/58 pass |

The fixed fitting fixture uses one real image and its negation with distinct text inputs to exercise batch arithmetic; it is not an image-caption quality benchmark. The batch4 training runs use the eight acquired museum image/caption pairs. Random times are shared within each batch, while data selection, caption dropout and Gaussian noise vary by item. No held-out data was added in this change.

Sampling prompt: `A red bird on a branch`; EMA weights; seed12345. NPY SHA-256:

`33db28c009639a07c8e6f67e1f74ac0e03852c3e54e018b87b6e4ce2979ab966`

The image remains nearly uniform gray. Neither short-run loss nor successful deterministic recovery establishes usable generation quality.

Server artifacts are under `runs/text-image-batched-checks-002/`, `runs/rgb-batched-continuous/`, `runs/rgb-batched-resumed/`, `runs/rgb-batched-sample-001/`, `runs/rgb-batched-sample-002/`, and `runs/tiles-regression.log`. The old runtime is preserved as `.camel-pre-batched-92df61c/`; the active project `.camel/` contains the tested batched backend.

```bash
python3 tools/check-text-image.py --out runs/new-batched-checks
python3 tools/text-image.py train --out runs/new-batch4 --steps 10 --batch 4
python3 tools/text-image.py train --out runs/new-batch4 --steps 20 --batch 4 --resume
python3 tools/text-image.py sample --run runs/new-batch4 --out runs/new-batch4-sample \
  --prompt 'A red bird on a branch' --guidance 2.0
```
