"""Corrected Metric B: transport vs linear on the TOPIC-CONTROLLED praise ribbon.

The geometry probe showed praise is strongly curved (arc/chord ~3 within-topic) but topic
explains ~3.7x more activation variance than praise level. The original Metric B measured
off-manifold distance to the WHOLE response cloud, whose spread is dominated by topic -> both
the transport and linear paths sat ~equally far from a fat topic blob (ratio 1.08). Wrong
reference set.

Here we remove topic (subtract each prompt's mean, overlaying all prompts' ribbons), so the
cloud IS the thin curved praise ribbon. We then compare three interpolations from low->high
praise and their off-ribbon distance:
  - LINEAR chord (c0 -> cK straight): cuts across the curve.
  - PWL through centroids (piecewise-linear c0-c1-...-cK): the 'known curve' upper baseline.
  - TRANSPORT field (segmented, re-anchored): learned field; its value over PWL is smoothness
    and, crucially, that it generalizes to steering from arbitrary points (not just centroids).
On a curved ribbon the chord should be far more off-manifold than PWL/transport. We report both
the RAW (topic-uncontrolled, reproduces the artifact) and TOPIC-CONTROLLED numbers side by side.

Run: python -m analyses.sycophancy_gate_a.praise_transport_within --responses <out>/responses.json --out <out>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyses.sycophancy_gate_a.gate_a import _load
from analyses.sycophancy_gate_a.praise_geometry_probe import _collect
from analyses.sycophancy_gate_a.transport_praise import _min_dist_to_real, _segmented_path


def _pwl_path(cents, K, sub):
    pts = []
    for k in range(K - 1):
        for s in range(sub):
            f = s / sub
            pts.append(cents[k] * (1 - f) + cents[k + 1] * f)
    pts.append(cents[K - 1])
    return np.asarray(pts)


def _paths_and_offmanifold(Xp, levels, K, field, sub):
    cents = np.stack([Xp[levels == k].mean(0) for k in range(K)])
    chord = np.stack([cents[0] * (1 - f) + cents[K - 1] * f
                      for f in np.linspace(0, 1, (K - 1) * sub + 1)])
    pwl = _pwl_path(cents, K, sub)
    tra = _segmented_path(field, cents, K, sub)
    return {"linear_chord": float(_min_dist_to_real(chord, Xp).mean()),
            "pwl_through_centroids": float(_min_dist_to_real(pwl, Xp).mean()),
            "transport_field": float(_min_dist_to_real(tra, Xp).mean())}


def _fit_field(Xp, levels, K, epochs, w_density):
    import torch

    from methods.behavioral_transport.transport import train_transport

    field, _ = train_transport(
        torch.tensor(Xp, dtype=torch.float32), torch.tensor(levels).long(), np.arange(K),
        periodic=False, W=K, hidden_dims=[128], epochs=epochs, lr=1e-3,
        neighbor_radius=1, w_density=w_density, seed=0, device="cpu")
    return field


def _pca(X, k):
    m = X.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(X - m, full_matrices=False)
    return (X - m) @ vt[:k].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 28])
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--sub_steps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--w_density", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = _load(args.model)
    data = json.load(open(args.responses))
    A, P, K = _collect(model, tok, data, args.layers)

    out = {"concept": "praise", "P": P, "K": K, "pca": args.pca, "layers": {}}
    print("=" * 84)
    print("CORRECTED METRIC B — off-manifold distance of low->high praise interpolations")
    print(f"{'layer':>5} {'setting':>16} {'linear_chord':>13} {'pwl_centroids':>14} "
          f"{'transport':>10} {'lin/transport':>14}")
    for L in args.layers:
        X = A[L]                                   # (P, K, D)
        levels = np.tile(np.arange(K), P)
        raw = X.reshape(P * K, -1)
        wc = (X - X.mean(1, keepdims=True)).reshape(P * K, -1)   # topic-controlled

        res = {}
        for name, flat in [("raw", raw), ("topic_controlled", wc)]:
            Xp = _pca(flat, args.pca)
            field = _fit_field(Xp, levels, K, args.epochs, args.w_density)
            m = _paths_and_offmanifold(Xp, levels, K, field, args.sub_steps)
            m["linear_over_transport"] = m["linear_chord"] / (m["transport_field"] + 1e-9)
            res[name] = m
            print(f"{L:>5} {name:>16} {m['linear_chord']:>13.3f} {m['pwl_through_centroids']:>14.3f} "
                  f"{m['transport_field']:>10.3f} {m['linear_over_transport']:>14.2f}")
        out["layers"][str(L)] = res
    print("=" * 84)
    print("read: topic_controlled lin/transport >> 1 => following the curved ribbon beats the")
    print("      straight chord (transport advantage), hidden before by topic variance.")
    print("=" * 84)

    with open(os.path.join(args.out, "praise_transport_within.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {args.out}/praise_transport_within.json")


if __name__ == "__main__":
    main()
