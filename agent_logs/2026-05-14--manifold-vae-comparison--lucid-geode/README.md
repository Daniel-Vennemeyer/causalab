# 2026-05-14--manifold-vae-comparison--lucid-geode

This session investigates how the current behavior-manifold learning code compares against a Behavior-Aligned Manifold VAE architecture, with the expected deliverable being a concrete experiment plan that tests VAE/atlas/metric/intervention variants against the existing spline-style manifold approach.

## Layout

- `plan/` — research objective, task-spec drafts, approval-checkpoint logs
- `run/` — resolved-config snapshot (`--cfg job` output), `run.log`, slurm logs
- `result/` — `REPORT.md` (single consolidated interpretation written by `/interpret-experiment`), `figures/` for embedded plots/tables
- `code/` — session-local Python + Hydra (via `/setup-methods`, `/setup-analyses`, `/run-experiment`)
- `artifacts/` — raw experiment outputs at `{task}/{model}/{analysis}/...`
- `issues.md` — top-level issue log spanning all phases (managed by `/document-issues`)
