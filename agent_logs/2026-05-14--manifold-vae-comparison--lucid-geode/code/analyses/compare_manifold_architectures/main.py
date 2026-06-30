"""compare_manifold_architectures: aggregate spline + VAE manifold arms.

Pure artifact aggregation — no model, no task model, CPU-only. Globs the
``behavior_manifold_vae`` arms (one row each from their ``comparison_ready.json``)
and the current-code spline baseline (``activation_manifold`` + ``path_steering``
metrics), maps both onto a common metric schema, and writes a summary table, an
ablation matrix, metric deltas, and figures.

This is a session-local *analysis* (research-question wrapper) — see ARCHITECTURE.md §3.
Layering rules respected by this module:
  - depends only on causalab.io.* primitives for I/O; reads peer-analysis output
    artifacts from disk (it does not import peer-analysis orchestration code)
  - no hyperparameter defaults inline — display/format knobs come from
    ``cfg.compare_manifold_architectures.<knob>``
  - ``cfg.experiment_root`` is the single source of truth for output paths
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
from typing import Any

from omegaconf import DictConfig, OmegaConf

from causalab.io.artifacts import save_experiment_metadata, save_json_results
from causalab.io.plots.figure_format import path_with_figure_format

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "compare_manifold_architectures"

# Metric columns shared across arms (names aligned with behavior_manifold_vae).
_METRIC_KEYS = [
    "reconstruction",
    "kl",
    "behavior_distance",
    "isometry_pearson_r",
    "geodesic_naturalness",
    "patch_consistency",
]
# Arm descriptor columns.
_DESCRIPTOR_KEYS = [
    "arm_id",
    "architecture",
    "arm_label",
    "task",
    "topology",
    "n_charts",
    "metric",
    "loss_set",
    "layer",
    "token_position",
    "seed",
]

# The six planned ablations: (label, predicate over a row -> bool membership).
_ABLATIONS = [
    (
        "flat-vs-manifold-geometry",
        "spline / flat_vae vs metric_vae / atlas_vae",
        lambda r: r.get("architecture"),
    ),
    (
        "single-chart-vs-atlas",
        "n_charts == 1 vs n_charts > 1",
        lambda r: "atlas" if (r.get("n_charts") or 1) > 1 else "single-chart",
    ),
    (
        "recon-only-vs-behavior-aligned",
        "loss_set",
        lambda r: r.get("loss_set"),
    ),
    (
        "decoder-vs-behavior-pullback-metric",
        "metric",
        lambda r: r.get("metric"),
    ),
    (
        "single-site-vs-trajectory",
        "activation surface (token_position)",
        lambda r: r.get("token_position"),
    ),
    (
        "unstructured-vs-known-topology",
        "topology",
        lambda r: r.get("topology"),
    ),
]


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _arm_id_from_path(root: str, comparison_path: str) -> str:
    """Stable arm id = path of the arm dir relative to the vae root."""
    arm_dir = os.path.dirname(comparison_path)
    rel = os.path.relpath(arm_dir, os.path.join(root, "behavior_manifold_vae"))
    return rel.replace(os.sep, "/")


def _collect_vae_arms(root: str) -> list[dict[str, Any]]:
    """One row per behavior_manifold_vae arm (from comparison_ready.json)."""
    rows: list[dict[str, Any]] = []
    vae_root = os.path.join(root, "behavior_manifold_vae")
    if not os.path.isdir(vae_root):
        return rows
    for dirpath, _dirnames, filenames in os.walk(vae_root):
        if "comparison_ready.json" in filenames:
            cpath = os.path.join(dirpath, "comparison_ready.json")
            data = _read_json(cpath)
            if data is None:
                continue
            row: dict[str, Any] = {"arm_id": _arm_id_from_path(root, cpath)}
            row.update(data)
            rows.append(row)
    return rows


def _first_metrics_json(
    base: str, prefixes: tuple[str, ...], prefer_mode: str = "geometric"
) -> dict[str, Any] | None:
    """Find a ``metrics.json`` anywhere under ``base`` whose path contains a
    directory component starting with one of ``prefixes``.

    path_steering nests criteria deeply, e.g.
    ``path_steering/<ss>/L<layer>_<pos>/<spline_sub>/<tv>/criteria/isometry/geometric/metrics.json``
    so we glob recursively and match on any path component (not just direct
    subdirs of ``base``). When multiple match, prefer the ``prefer_mode`` path
    (the manifold-geodesic result is the representative spline number)."""
    if not os.path.isdir(base):
        return None
    matched = [
        cand
        for cand in glob.glob(os.path.join(base, "**", "metrics.json"), recursive=True)
        if any(part.startswith(prefixes) for part in cand.split(os.sep))
    ]
    if not matched:
        return None
    preferred = [c for c in matched if prefer_mode in c.split(os.sep)]
    pick = preferred[0] if preferred else sorted(matched)[0]
    return _read_json(pick)


def _collect_spline_baseline(root: str) -> dict[str, Any] | None:
    """Build a single spline-baseline row from current-code artifacts.

    Defensive: every source is optional; missing -> the value stays None.
    """
    found_any = False
    row: dict[str, Any] = {
        "arm_id": "spline_baseline",
        "architecture": "spline",
        "task": None,
        "topology": None,
        "n_charts": 1,
        "metric": "spline",
        "loss_set": "recon_only",
        "layer": None,
        "token_position": None,
        "seed": None,
    }
    for k in _METRIC_KEYS:
        row[k] = None

    # reconstruction (KL) from activation_manifold metadata.json files.
    am_root = os.path.join(root, "activation_manifold")
    if os.path.isdir(am_root):
        for meta_path in glob.glob(
            os.path.join(am_root, "**", "metadata.json"), recursive=True
        ):
            meta = _read_json(meta_path)
            if meta and "reconstruction_kl" in meta:
                row["reconstruction"] = meta["reconstruction_kl"]
                row.setdefault("task", meta.get("task"))
                if row.get("task") is None:
                    row["task"] = meta.get("task")
                if row.get("layer") is None:
                    row["layer"] = meta.get("layer")
                if row.get("token_position") is None:
                    row["token_position"] = meta.get("token_position")
                if row.get("seed") is None:
                    row["seed"] = meta.get("seed")
                found_any = True
                break

    # path_steering: isometry pearson_r + coherence + distance_from_behavior_manifold
    ps_root = os.path.join(root, "path_steering")
    iso = _first_metrics_json(ps_root, ("isometry",))
    if iso is not None:
        if "pearson_r" in iso:
            row["isometry_pearson_r"] = iso["pearson_r"]
        found_any = True
    coh = _first_metrics_json(ps_root, ("coherence",))
    if coh is not None:
        # carry coherence as an extra column under a stable key.
        row["coherence"] = coh.get("coherence", coh.get("mean", None))
        found_any = True
    dist = _first_metrics_json(ps_root, ("distance_from_behavior_manifold",))
    if dist is not None:
        row["distance_from_behavior_manifold"] = dist.get(
            "distance_from_behavior_manifold", dist.get("mean", None)
        )
        found_any = True

    # path_steering results_summary.csv (optional, defensive): pull any of the
    # mapped metric columns if present and not already filled.
    summary_csv = os.path.join(ps_root, "results_summary.csv")
    if os.path.isfile(summary_csv):
        try:
            with open(summary_csv, newline="") as f:
                reader = csv.DictReader(f)
                first = next(reader, None)
            if first:
                found_any = True
                for src, dst in (
                    ("pearson_r", "isometry_pearson_r"),
                    ("isometry_pearson_r", "isometry_pearson_r"),
                    ("coherence", "coherence"),
                    (
                        "distance_from_behavior_manifold",
                        "distance_from_behavior_manifold",
                    ),
                ):
                    if src in first and row.get(dst) in (None,):
                        try:
                            row[dst] = float(first[src])
                        except (TypeError, ValueError):
                            pass
        except (OSError, csv.Error) as exc:
            logger.warning("Could not read %s: %s", summary_csv, exc)

    return row if found_any else None


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _write_summary_by_arm(rows: list[dict[str, Any]], out_dir: str) -> str:
    """Aggregate per-arm/seed rows into one row per arm (group over seed) with
    mean/std/n across seeds for each metric. This is the error-bar view: an arm
    is the descriptor tuple minus seed."""
    group_keys = [
        "architecture", "arm_label", "task", "topology", "n_charts",
        "metric", "loss_set", "layer", "token_position",
    ]
    agg_metrics: list[str] = []
    for m in list(_METRIC_KEYS) + ["coherence", "distance_from_behavior_manifold"]:
        if m not in agg_metrics:
            agg_metrics.append(m)

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in group_keys)
        groups.setdefault(key, []).append(r)

    out_rows: list[dict[str, Any]] = []
    for key, grp in groups.items():
        row: dict[str, Any] = dict(zip(group_keys, key))
        row["n_seeds"] = len(grp)
        for m in agg_metrics:
            vals = [v for v in (_to_float(r.get(m)) for r in grp) if v is not None]
            if vals:
                mean = sum(vals) / len(vals)
                std = (
                    (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                    if len(vals) > 1
                    else 0.0
                )
                row[f"{m}_mean"] = round(mean, 6)
                row[f"{m}_std"] = round(std, 6)
            else:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
        out_rows.append(row)

    columns = group_keys + ["n_seeds"]
    for m in agg_metrics:
        columns += [f"{m}_mean", f"{m}_std"]
    path = os.path.join(out_dir, "summary_by_arm.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in out_rows:
            writer.writerow({c: r.get(c, "") for c in columns})
    return path


def _write_summary_csv(rows: list[dict[str, Any]], out_dir: str) -> str:
    # Column union: descriptors first, then metric keys, then any extras.
    extra_cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in _DESCRIPTOR_KEYS and k not in _METRIC_KEYS and k not in extra_cols:
                extra_cols.append(k)
    columns = _DESCRIPTOR_KEYS + _METRIC_KEYS + sorted(extra_cols)
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})
    return path


def _write_ablation_matrix(rows: list[dict[str, Any]], out_dir: str) -> str:
    lines = ["# Ablation matrix", ""]
    lines.append("| Ablation | Dimension | Arms present | Key metric (recon / iso_r) |")
    lines.append("|---|---|---|---|")
    for label, dim, key_fn in _ABLATIONS:
        groups: dict[Any, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(key_fn(r), []).append(r)
        present = [str(g) for g in groups if g is not None]
        if not present:
            arms_cell = "not yet run"
            metric_cell = "—"
        else:
            arms_cell = ", ".join(
                f"{g} (n={len(groups[g])})" for g in groups if g is not None
            )
            deltas = []
            for g in groups:
                if g is None:
                    continue
                recons = [_to_float(r.get("reconstruction")) for r in groups[g]]
                isos = [_to_float(r.get("isometry_pearson_r")) for r in groups[g]]
                recons = [x for x in recons if x is not None]
                isos = [x for x in isos if x is not None]
                rec_m = f"{sum(recons) / len(recons):.4f}" if recons else "—"
                iso_m = f"{sum(isos) / len(isos):.4f}" if isos else "—"
                deltas.append(f"{g}: {rec_m} / {iso_m}")
            metric_cell = "; ".join(deltas) if deltas else "—"
        lines.append(f"| {label} | {dim} | {arms_cell} | {metric_cell} |")
    lines.append("")
    path = os.path.join(out_dir, "ablation_matrix.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _write_metric_deltas(
    vae_rows: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    out_dir: str,
) -> str:
    deltas: dict[str, dict[str, float]] = {}
    if baseline is not None:
        for r in vae_rows:
            arm_id = r.get("arm_id", "unknown")
            arm_deltas: dict[str, float] = {}
            for k in _METRIC_KEYS:
                v_vae = _to_float(r.get(k))
                v_base = _to_float(baseline.get(k))
                if v_vae is not None and v_base is not None:
                    arm_deltas[k] = v_vae - v_base
            if arm_deltas:
                deltas[arm_id] = arm_deltas
    return save_json_results(deltas, out_dir, "metric_deltas.json")


def _make_figures(
    rows: list[dict[str, Any]], fig_dir: str, figure_format: str
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    written: list[str] = []

    # Stable color per architecture.
    archs = sorted({str(r.get("architecture")) for r in rows})
    cmap = plt.get_cmap("tab10")
    arch_color = {a: cmap(i % 10) for i, a in enumerate(archs)}

    # (a) reconstruction vs behavior_distance scatter, colored by architecture.
    pts = [
        (
            _to_float(r.get("reconstruction")),
            _to_float(r.get("behavior_distance")),
            str(r.get("architecture")),
        )
        for r in rows
    ]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if len(pts) >= 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        for a in archs:
            xs = [p[0] for p in pts if p[2] == a]
            ys = [p[1] for p in pts if p[2] == a]
            if xs:
                ax.scatter(xs, ys, label=a, color=arch_color[a])
        ax.set_xlabel("reconstruction")
        ax.set_ylabel("behavior_distance")
        ax.set_title("Reconstruction vs behavior distance")
        ax.legend()
        path = path_with_figure_format(
            os.path.join(fig_dir, "recon_vs_behavior.png"), figure_format
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    else:
        logger.info("Skipping recon_vs_behavior figure: <1 finite point")

    # (b) isometry_pearson_r bar by arm.
    bars = [
        (str(r.get("arm_id")), _to_float(r.get("isometry_pearson_r")), str(r.get("architecture")))
        for r in rows
    ]
    bars = [b for b in bars if b[1] is not None]
    if len(bars) >= 1:
        fig, ax = plt.subplots(figsize=(max(6, len(bars) * 1.2), 5))
        ax.bar(
            range(len(bars)),
            [b[1] for b in bars],
            color=[arch_color[b[2]] for b in bars],
        )
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("isometry_pearson_r")
        ax.set_title("Isometry (Pearson r) by arm")
        path = path_with_figure_format(
            os.path.join(fig_dir, "isometry_by_arm.png"), figure_format
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    else:
        logger.info("Skipping isometry_by_arm figure: <1 finite point")

    # (c) geodesic_naturalness bar by arm.
    nat = [
        (str(r.get("arm_id")), _to_float(r.get("geodesic_naturalness")), str(r.get("architecture")))
        for r in rows
    ]
    nat = [b for b in nat if b[1] is not None]
    if len(nat) >= 1:
        fig, ax = plt.subplots(figsize=(max(6, len(nat) * 1.2), 5))
        ax.bar(
            range(len(nat)),
            [b[1] for b in nat],
            color=[arch_color[b[2]] for b in nat],
        )
        ax.set_xticks(range(len(nat)))
        ax.set_xticklabels([b[0] for b in nat], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("geodesic_naturalness")
        ax.set_title("Geodesic naturalness by arm (lower = better)")
        path = path_with_figure_format(
            os.path.join(fig_dir, "geodesic_naturalness_by_arm.png"), figure_format
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    else:
        logger.info("Skipping geodesic_naturalness_by_arm figure: <1 finite point")

    return written


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the compare_manifold_architectures aggregation."""
    analysis = cfg[ANALYSIS_NAME]
    root = cfg.experiment_root
    out_dir = analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)
    figure_format = analysis.visualization.figure_format

    vae_rows = _collect_vae_arms(root)
    baseline = _collect_spline_baseline(root)

    all_rows: list[dict[str, Any]] = []
    if baseline is not None:
        all_rows.append(baseline)
    all_rows.extend(vae_rows)

    if not all_rows:
        # No arms at all: write a header-only summary + a missing-inputs note.
        columns = _DESCRIPTOR_KEYS + _METRIC_KEYS
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
        missing = {
            "behavior_manifold_vae": os.path.isdir(
                os.path.join(root, "behavior_manifold_vae")
            ),
            "activation_manifold": os.path.isdir(
                os.path.join(root, "activation_manifold")
            ),
            "path_steering": os.path.isdir(os.path.join(root, "path_steering")),
            "note": (
                "No spline or VAE manifold arms found under experiment_root. "
                "Run behavior_manifold_vae and/or the current-code spline "
                "analyses first."
            ),
        }
        save_json_results(missing, out_dir, "missing_inputs.json")
        save_experiment_metadata(
            OmegaConf.to_container(cfg, resolve=True),
            out_dir,
            "experiment_metadata.json",
        )
        logger.warning("compare_manifold_architectures: no arms found.")
        return {"summary_rows": 0, "out_dir": out_dir}

    _write_summary_csv(all_rows, out_dir)
    _write_summary_by_arm(all_rows, out_dir)
    _write_ablation_matrix(all_rows, out_dir)
    _write_metric_deltas(vae_rows, baseline, out_dir)
    _make_figures(all_rows, os.path.join(out_dir, "figures"), figure_format)

    save_experiment_metadata(
        OmegaConf.to_container(cfg, resolve=True), out_dir, "experiment_metadata.json"
    )

    logger.info(
        "compare_manifold_architectures complete: %d arms -> %s",
        len(all_rows),
        out_dir,
    )
    return {"summary_rows": len(all_rows), "out_dir": out_dir}
