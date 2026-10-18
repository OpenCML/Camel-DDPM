# General text-to-image, trained from scratch

## Binding requirements

- General Chinese/English text-to-image; MNIST is a regression suite only.
- Initialize **every learned component randomly**: tokenizer statistics (if any), text encoder, image autoencoder, denoiser and any training discriminator. Do not load CLIP, T5, VAE, diffusion, captioning or perceptual-loss weights. Do not train in PyTorch behind a Camel entry point.
- Run downloads, builds, numerical checks and training on `88zjwei`. The workstation is for editing and transport only. Use the project-local `.camel` distribution, explicit CUDA, one host tensor thread, and an unoccupied GPU. Other server users' allocations are not available capacity.
- Label correctness fixtures and early checkpoints honestly. No current checkpoint is production ready.

## Architecture direction

Use a jointly trained byte-level text Transformer and a conditional diffusion Transformer. A graph-visible attention implementation fits Camel's existing matrix primitives and avoids making GPU convolution a dependency of the first RGB integration test. UTF-8 bytes cover both languages without importing a pretrained tokenizer. The 64-byte fixture is **not** the eventual context budget: it truncates Chinese much earlier than English.

The scalable target is latent diffusion, with an autoencoder trained first on the selected image corpus. Train its encoder, decoder and any discriminator from random initialization. Pretrained VGG/LPIPS perceptual losses would violate this project's training constraint. A direct pixel model can validate conditional denoising before that autoencoder is ready; it does not require an image encoder at all.

The checked-in `text_image.cml` is such an integration fixture: RGB64, 8×8 pixel patches, width 64, four heads, independent batch elements, shared projections, bidirectional text mixing, image self-attention and text cross-attention. All seven parameter leaves are new random tensors and receive gradients. It is not a full DiT implementation and has neither sufficient capacity nor the training evidence to claim general generation.

Reference architecture research: [Diffusion Transformers](https://arxiv.org/abs/2212.09748). That paper uses latent patches and reports class-conditional ImageNet results; it does not by itself validate our text-conditioned architecture, corpus or training budget.

## Data acquisition

The implemented source is [Smithsonian Open Access](https://github.com/Smithsonian/OpenAccess), fetched from its published S3 metadata indexes. `tools/prepare_openaccess.py` selects records whose metadata **and individual image media** explicitly say `CC0`. A metadata license alone never admits an image. The pilot preserves JPEGs as retrieved from the published delivery service, metadata shards, record URLs, media URLs, credit lines, license declarations, retrieval times and SHA-256 digests. These JPEGs may be derivatives of the archival files. It exports a 64×64 center-cropped RGB view for the Camel loader. Captions combine existing titles/topics/medium fields and are labeled `museum-metadata`, not dense visual descriptions.

The pilot is a transport/codec fixture with museum-specific coverage. It is not an adequate corpus for general text-to-image. Its English metadata cannot establish Chinese instruction following. Exact image hashes determine train/validation assignment; perceptual duplicates, object-level duplicates and related images across sources still require grouping before a real evaluation split.

For broader bilingual coverage, [WIT](https://github.com/google-research-datasets/wit) publishes 37.6M image-text examples, 11.5M unique images and 108 languages, with [downloadable metadata shards](https://github.com/google-research-datasets/wit/blob/main/DATA.md). Image availability, individual licenses/attribution, caption suitability, language distribution and overlap need auditing before admitting any WIT images. A successful metadata probe is not an image corpus. The server's Wikimedia API request timed out during initial probing, so WIT is currently a candidate, not an acquired training source.

Run on the server:

```bash
cd /home/zjwei/projects/Camel-DDPM
# Use an environment with Pillow. Python handles network/codec work only.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/zjwei/camel-eval/venv-bench/bin/python tools/prepare_openaccess.py \
  --units saam --limit 8
python3 tools/check-text-image.py --gpu 2 --out runs/text-image-checks-001
# Fresh random initialization; this short run verifies the training pipeline.
python3 tools/text-image.py train --out runs/rgb-example --steps 10
python3 tools/text-image.py train --out runs/rgb-example --steps 20 --resume
python3 tools/text-image.py sample --run runs/rgb-example --out runs/rgb-sample \
  --prompt 'A red bird on a branch' --guidance 2.0
```

The downloader resumes by verifying existing images, RGB views and captions. Each fetch has size and time bounds, errors are recorded, and the summary reports whether the requested limit was reached. A single preparation process must own a dataset directory; concurrent writers are unsupported.

The RGB trainer defaults to two independently selected images per update (`--batch` is configurable), with independent Gaussian noise, a shared random diffusion time for the batch, joint text embedding updates, 10% per-item caption dropout, native Camel Adam and EMA. Every checkpoint contains all four tensor trees. The host launcher validates shape, size, dtype, finiteness and SHA-256, flushes files, then publishes the directory and latest pointer atomically. A process lock prevents concurrent writers. Per-step reseeding reconstructs random draws from the recorded seed and global step, rather than relying on an unrecorded generator state. Source/runtime fingerprints, dataset order/hash and hyperparameters, including batch size, must match on resume. Only previously published complete checkpoints are eligible; partial directories are ignored.

This Linux fixture restarts Camel between checkpoint segments. That makes process recovery easy to test but adds compilation/launch overhead and is not the final large-scale training scheduler. Mid-segment interruption loses updates after the last published checkpoint. The filesystem must support the POSIX rename/fsync semantics used by the launcher; power-loss behavior on every server filesystem has not been tested.

The batched model is versioned `rgb64-joint-v2`. Source/runtime identity checks deliberately reject continuation of old `v1` runs; reproduce those with their original source/runtime. See [batched validation results](results/server-batched-validation.md) and the [historical single-head results](results/server-rgb-validation.md). These are pipeline checks with **no useful text-to-image quality claim**.

## Required capability gates

| Gate | Acceptance evidence | Current state |
|---|---|---|
| Native attention | Stable large finite logits, CPU/GPU gradients, independent directional finite differences | Batched multi-head self/cross-attention and RMSNorm pass; batch isolation error zero |
| Native RGB/text integration | Real image decode, UTF-8 byte ids, GPU forward/backward, joint text gradients, weight reload | Passes on real RGB data |
| Dataset readiness | Diverse train/held-out corpus, language counts, caption inspection, license/provenance ledger, grouped duplicates, immutable shard hashes | Bounded museum pilot only |
| Scalable kernels | Batched multi-head attention, embedding gather/scatter VJP, efficient matmul, documented precision contracts | Rank ≥2 batched matmul/pullbacks and multi-head attention implemented; gather/scatter and high-throughput kernels still missing |
| Memory and throughput | Mixed precision with explicit numeric permissions, activation recomputation, accumulation, measured peak memory/step time | Missing; no scaling claim |
| Recoverable training | Atomic versioned model/Adam/EMA/step/RNG/data-cursor checkpoint; interrupted vs uninterrupted trajectory comparison | RGB fixture passes process restart and mid-segment interruption; MNIST entry point remains model-only |
| Full learned stack | Random-initialized autoencoder/text encoder/denoiser trained and validated at target resolution | Not trained |
| Conditional sampling | Independent checkpoint load, prompt encoder, classifier-free training/guidance, DDPM and faster sampler checks | RGB DDPM/CFG and repeated sampling pass; faster samplers missing |
| Evaluation | Fixed Chinese/English prompts, prompt swap tests, held-out denoising, diversity, memorization checks, human review | Not established; prompt sensitivity alone is insufficient |
| Serving | Versioned config/tokenization/weights, deterministic seed handling, bounded requests, health checks, concurrency and latency/load tests | Missing |

## Scaling decision

Do not start a large training allocation by merely increasing MNIST steps. First measure the complete RGB stack, including checkpoint round trips and activation memory. For FP32 training, parameters, gradients, Adam moments and EMA alone occupy about **20 bytes per parameter**, before activations, compilation workspace and the live old/new tensors in functional updates. A 500M-parameter model therefore starts at roughly 10GB of persistent tensors, not a 10GB total working set. This is a storage lower bound, not a feasibility or speed estimate.

Choose model size, batch size, target corpus and training duration from server measurements after the missing kernels are implemented. Multi-GPU is not currently supported by this project; no NCCL/distributed speedup is assumed. Keep intermediate 64/128px tests separate from the eventual 256/512px quality gates. Publish the source revision, runtime patch, initialization seed, corpus manifest, objective, schedule, precision settings, validation results and hardware for each promoted checkpoint.
