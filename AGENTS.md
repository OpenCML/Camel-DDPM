# Experiment rules

- Use Windows PowerShell and the project-local `.camel/bin/camel.exe`.
- Keep model math, differentiation, optimizers, IDX decoding and sampling in Camel.
- Default to explicit CUDA execution. Never silently fall back to CPU for training.
- Keep CPU tensor threads at one unless the user explicitly requests more.
- Use short verification runs first; do not start another long training run without a concrete validation need.
- Distinguish pipeline correctness, validation improvement and visual sample quality in reports.
- Keep datasets, runtime installations and run checkpoints out of Git.
- Record necessary Camel library changes as scoped patches under runtime/; do not bundle unrelated checkout changes.

## Server storage

- On `88zjwei`, store all datasets, downloaded archives, image caches, indices, training logs and checkpoints under `/mnt/data3/zjwei/Camel-DDPM-data/`, never as real files under `/home`.
- Project `data` and `runs` are symlinks to `project-data` and `project-runs` under that storage root. Preserve these links during deployments; do not replace them with directories.
- WIT lives under that root's `wit/`; COCO lives under `coco2017/`. Resolve destination paths and verify their filesystem before starting a large transfer or training run.
