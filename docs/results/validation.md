# Recorded validation — 2026-10-18

The committed image is the unfiltered 64-sample grid from the GPU baseline.
It is the Camel-produced PGM converted to PNG for browser display, with nearest
neighbor enlargement. No generated image or enhancement model is involved.

Configuration: width 512, batch 32, 5,000 Adam updates at 0.001, 100 cosine
diffusion steps, sample seed 12345, training/tensor RNG seed 20261018, 10 ms
pause per update, one host tensor thread. Device: NVIDIA GeForce RTX 4060 Ti
(sm_89). CUDA is required; the graph APIs do not fall back to CPU.

- Training including startup/validation: 158.234 seconds (single measurement).
- Fixed validation MSE: 0.5219627431965799 → 0.3710841253440509.
- Fixed-validation ratio: 0.7109398710556903.
- Final samples: 64 × 784 f32 values, equivalent to 64 × 28 × 28 grayscale images.
- Repeated sampling in a fresh process with the same checkpoint and seed produced
  the identical NPY SHA-256:
  `f62f3407cd8069904977178926dc56372aa7bd6a065d5e08b34e412a2231ce51`.

The finite-difference and CPU/GPU checks use an independent small model, not the
baseline validation images. `test.ps1` reported:

```text
CHECK gradient finite_difference=0.0122474454317123 autodiff=0.012250000167114194 abs_error=2.5547354018939927e-06
CHECK Adam before=1.44761966854154 after=1.440850106978663 reload=1.440850106978663
CHECK diffusion forward_error=0.0 terminal_error=0.0 terminal_abar=2.4285722793500615e-07
CHECKS PASS
GPU CHECK gradient_l1=1.3098736673100575e-07 Adam_l1=9.79952702095943e-07 changed_input_l1=1.3754377681252095e-07
GPU CHECKS PASS
```

Acceptance: the numerical/training/checkpoint/sampling pipeline passed. Visual
sample quality did not pass: strokes appear, but substantial noise remains.
Neither FID nor a classifier-based digit-recognition score has been measured.
The 28.9% decrease is relative to this model's initialization (which already has
an analytic noise skip), not relative to a zero predictor or another architecture.
