"""Session-local analysis `compare_manifold_architectures`.

Aggregates current-code (spline) and VAE manifold arms into one comparison
table plus report-ready artifacts. The entry point is
``analyses.compare_manifold_architectures.main.main``; this package
intentionally does NOT re-export it, so
``import analyses.compare_manifold_architectures.main`` resolves to the
submodule (matching the shipped ``causalab.analyses.*`` convention and the
runner's ``_name_`` dispatch). See set_up_analysis.md alongside this file for
the spec; see ARCHITECTURE.md §3 for the layering rules this module respects.
"""
