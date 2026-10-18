# Native device parameter trees

The 2026-10-18 Camel training infrastructure design is being implemented in stages. The first delivered part normalizes `Device<Parameters>` into the same struct/tuple type returned by `run_device`, and supports `upload`/`download` on the complete tree.

`examples/device-tree-step.cml` exercises the actual 1,116,864-parameter VAE and 1,906,180-parameter text-conditioned diffusion model on one real batch from the sealed `openaccess-256-v1` corpus. Both models initialize from scratch. Forward, autodiff, Adam and latent noising execute through native Camel/Tiles; the encoded latent remains on the GPU. Initialization and noise generation are still on the host in this stage.

Two independent runs on server `88zjwei`, physical GPU 2, produced:

| Observation | Result |
|---|---:|
| VAE loss | 0.3690241159925844 |
| Diffusion noise MSE | 1.025955108172196 |
| VAE squared parameter change | 0.00011043555311198371 |
| Diffusion squared parameter change | 0.00018747812900134675 |
| Model and Adam moment `.npy` leaves | 390 |
| Finite f32 values in every leaf | Yes |
| Identical checkpoint bytes between runs | Yes |

Logs and checkpoints: `/home/zjwei/projects/Camel-DDPM/runs/device-tree-training-001` and `device-tree-training-002`. The independent runtimes are `.camel-device-tree-v1` and `.camel-device-tree-v2`; the original `.camel` installation and earlier experiment identities remain intact.

The original observations above used project revision `a8f04f1`. The current example also uses generic `tree` operations and `Bytes`, so reproduce it with `.camel-tree-bytes-v2` (or the updated default installation). Set `CAMEL_HOME` to that installation, `CAMEL_STD_LIB` to its `stdlib`, `CAMEL_PACKAGES` to the project `src`, and preload that runtime's `libstdc++.so.6` and `libgcc_s.so.1`. Select physical GPU 2 through `CUDA_VISIBLE_DEVICES=2` with `CAMEL_TILES_DEVICE=0`; keep host tensor threads at one. Invoke:

```bash
"$CAMEL_HOME/bin/camel" \
  --arg data=data/openaccess-256-v1 \
  --arg index=runs/vae-256-mixed-v1/train_ids.txt \
  --arg out=runs/device-tree-training-fresh \
  examples/device-tree-step.cml std::macro std::fvm
```

Use a fresh output directory. These are integration observations, not evidence of production image quality. Generic `tree` operations now replace generated parameter operations. The migrated example produced the same 390 checkpoint files byte for byte. Native tree checkpoint publication, tensor/Bytes conversion, keyed random numbers and a complete device-resident step remain outstanding.

WIT acquisition continues separately. The latest audited snapshot (`audit-003`) decoded 576 transported images, found 509 consistent license declarations and set 67 aside for review. Caption quality, duplicate grouping and sealed train/evaluation shards are still required; none of these raw WIT assets has been admitted to training yet.
