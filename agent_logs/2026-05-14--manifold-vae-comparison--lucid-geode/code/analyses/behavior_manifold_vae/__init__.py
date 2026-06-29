"""Session-local analysis `behavior_manifold_vae`.

Trains one behavior-aligned VAE arm on cached subspace features and writes
comparison-ready manifold-geometry metrics. The entry point is
``analyses.behavior_manifold_vae.main.main``; this package intentionally does
NOT re-export it, so ``import analyses.behavior_manifold_vae.main`` resolves to
the submodule (matching the shipped ``causalab.analyses.*`` convention and the
runner's ``_name_`` dispatch). See set_up_analysis.md alongside this file for
the spec; see ARCHITECTURE.md §3 for the layering rules this module respects.
"""
