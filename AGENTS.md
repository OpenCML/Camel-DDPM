# Experiment rules

- Use Windows PowerShell and the project-local `.camel/bin/camel.exe`.
- Keep model math, differentiation, optimizers, IDX decoding and sampling in Camel.
- Default to explicit CUDA execution. Never silently fall back to CPU for training.
- Keep CPU tensor threads at one unless the user explicitly requests more.
- Use short verification runs first; do not start another long training run without a concrete validation need.
- Distinguish pipeline correctness, validation improvement and visual sample quality in reports.
- Keep datasets, runtime installations and run checkpoints out of Git.
- Record necessary Camel library changes as scoped patches under runtime/; do not bundle unrelated checkout changes.
