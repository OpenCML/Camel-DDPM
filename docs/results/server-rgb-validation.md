# Native RGB/text integration — 2026-10-18

Historical `rgb64-joint-v1` results at source commit `92df61c`. The current model is batched `v2`; see [its validation record](server-batched-validation.md). Reproducing this older run requires its original source/runtime fingerprints.

This validates the execution and recovery pipeline, **not general generation quality**. All model weights start randomly; Python performs transport, image codecs, file validation and process orchestration only. Forward computation, byte token encoding, attention, autodiff, Adam, EMA and DDPM sampling are Camel programs running through explicit Tiles CUDA calls.

- Host: `88zjwei` / `tangshan88`; project: `/home/zjwei/projects/Camel-DDPM`.
- Device: RTX 3090, physical GPU 2, UUID `GPU-c10aaf4f-7c2c-3fa5-f47d-bb26db738a49`.
- Runtime: project `.camel`, with the scoped primitives patch under `runtime/`.
- Model: `rgb64-joint-v1`, RGB64, patch8, width64, 54,464 parameters. It is a small single-head integration fixture, not a full production DiT.
- Corpus: eight CC0-marked Smithsonian image/text records, acquired on the server. Caption source is museum metadata. The hash split happened to assign all eight to training; **there is no held-out quality evaluation**.
- Manifest SHA-256: `917a2632a3b06a7be1a3fb74470f0cec58ebfb537983e37b4d60fe9d3d88179e`.
- A second preparation run verified cached image/RGB/caption files and preserved the manifest digest.

| Check | Observed result |
|---|---|
| Attention CPU/GPU parameter-gradient L1 difference | `2.7008354663848877e-7` |
| Attention directional finite-difference absolute error | `5.758798215538263e-5` |
| Softmax with large positive/negative and tied finite logits | Expected probabilities; L1 error `0` |
| UTF-8 bytes for `A中`, BOS/EOS/PAD | Pass |
| RGB patchify/inverse round trip | Error `0` |
| Complete RGB network CPU/GPU gradient L1 difference | `1.0390952526629499e-6` |
| Text embedding gradient L1 | `0.11520784563685993`, nonzero |
| Fixed single-image/noise fitting, 10 steps | x0-MSE `0.7749511152236904 → 0.0552092822438511` |
| Reloaded RGB forward output | L1 error `0` |
| Changing prompt with identical noisy image/time | Forward L1 change `19.553584456443787`; sensitivity alone is not semantic alignment |
| Continuous 20 updates vs 10 + fresh process + 10 | All 28 model/Adam-m/Adam-v/EMA NPY files byte-identical |
| Active interruption after step 11, recover checkpoint10, continue to20 | Same 28 files byte-identical to continuous run |
| Corrupted checkpoint tensor / invalid latest pointer | Both rejected |
| Two independent 100-step DDPM sampling processes, CFG=2 | Identical NPY SHA-256 |

Sampling uses EMA, seed12345, prompt `A red bird on a branch`; SHA-256 is `e7df0e4b01d448324119493e718dac6860fd53f2a513276a7eaa93f6676fd78c`. The resulting 64×64 image is nearly uniform gray, without a recognizable bird. Twenty updates on eight museum images cannot establish prompt following, diversity or usable image quality. Do not promote this checkpoint as a product model.

Server artifacts:

- `runs/text-image-checks-003/`: attention, byte/RGB, gradient and reload logs.
- `runs/rgb-continuous/`: 20-step continuous training and complete checkpoint.
- `runs/rgb-resumed/`: checkpoint10 and resumed checkpoint20.
- `runs/rgb-sample-001/`, `runs/rgb-sample-002/`: independent DDPM/CFG outputs and checksums.

Recheck recovery, including a real process interruption in an isolated temporary copy:

```bash
python3 tests/recovery_checks.py runs/rgb-continuous runs/rgb-resumed --exercise-interruption
```

The interruption test controls only the processes and temporary output tree it creates. It does not stop other server jobs. It verifies process failure, not arbitrary power loss or network-filesystem durability.
