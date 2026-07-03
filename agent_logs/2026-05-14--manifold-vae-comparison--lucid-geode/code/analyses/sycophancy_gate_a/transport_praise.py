"""Gate T + E1 (praise) — transport steering on a social behavior.

Both gates passed: praise is a consistent 1-D judgment axis (Gate A, cycles 0.059) AND a
well-ordered activation direction (Gate B, isometry 0.685 / PC1-vs-level 0.864). This trains
the decoder-free transport field on the praise axis (in the validated PCA-64 subspace) and
evaluates two ways:

  METRIC B (on-manifold interpolation — where transport genuinely differs from linear):
    interpolate low->high praise along the FIELD vs a STRAIGHT chord and measure how far each
    intermediate point sits from the real-response activation cloud. Praise has only 6 sparse
    levels, so the field is integrated SEGMENT-BY-SEGMENT, re-anchored at each real level
    centroid (free integration across sparse levels diverges). Transport should hug the curved
    manifold; the chord cuts across it -> transport more on-manifold. This reproduces the
    paper's core geodesic-vs-linear claim for a social behavior.

  METRIC A (behavioral dose-response — is the axis a causal generation knob?):
    ActAdd the praise steering direction (diff-of-means; with re-anchoring transport's net
    displacement equals this) at increasing strengths at layer 28, generate, and use the
    target model's own praise judge to test whether steered responses are MORE praising than
    baseline and whether praise rises MONOTONICALLY with strength, while a distinct-token
    ratio guards against degeneration.

Run: python -m analyses.sycophancy_gate_a.transport_praise --responses <out>/responses.json --out <out>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyses.sycophancy_gate_a.gate_a import CONCEPTS, _chat_generate, _judge, _load
from analyses.sycophancy_gate_a.gate_b import _response_activation


def _collect(model, tok, data, layer):
    acts, levels = [], []
    for item in data:
        for k, resp in enumerate(item["responses"]):
            acts.append(_response_activation(model, tok, item["prompt"], resp, layer))
            levels.append(k)
    return np.asarray(acts), np.asarray(levels)


def _min_dist_to_real(points: np.ndarray, real: np.ndarray) -> np.ndarray:
    d = np.sqrt(((points[:, None, :] - real[None, :, :]) ** 2).sum(-1))
    return d.min(1)


def _segmented_path(field, cents_pca, K, sub):
    """Integrate one level-interval at a time, re-anchoring at each real centroid. Free
    integration across sparse levels diverges (untrained field regions -> exploding Jacobian);
    re-anchoring bounds drift to a single interval."""
    import torch

    from methods.behavioral_transport.transport import integrate_path

    pts = []
    for k in range(K - 1):
        seg = integrate_path(field, torch.tensor(cents_pca[k], dtype=torch.float32),
                             float(k), float(k + 1), n_steps=sub + 1).numpy()
        pts.append(seg[:-1])                 # drop last: it is the k+1 re-anchor
    pts.append(cents_pca[K - 1][None])
    return np.concatenate(pts, 0)


def _distinct_ratio(text: str) -> float:
    toks = text.split()
    return len(set(toks)) / max(1, len(toks))


class _ActAdd:
    """Context manager: add a fixed delta (D,) to a layer's residual output during generate."""

    def __init__(self, model, layer_idx, delta):
        import torch

        self.layer = model.model.layers[layer_idx]
        self.delta = torch.tensor(delta, dtype=torch.float32, device=model.device)
        self.h = None

    def __enter__(self):
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                return (out[0] + self.delta.to(out[0].dtype),) + tuple(out[1:])
            return out + self.delta.to(out.dtype)

        self.h = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self.h.remove()


def _praise_winrate(model, tok, prompt, steered, baseline, judge_tmpl, seed):
    """Fraction of 2 order-swapped judgments where `steered` is judged MORE praising."""
    wins = 0.0
    if _judge(model, tok, prompt, steered, baseline, judge_tmpl, seed=seed) == "A":
        wins += 1
    if _judge(model, tok, prompt, baseline, steered, judge_tmpl, seed=seed + 1) == "B":
        wins += 1
    return wins / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--w_density", type=float, default=3.0)
    ap.add_argument("--sub_steps", type=int, default=4, help="integration steps per level interval")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0],
                    help="steering strengths (x ||cent_high - cent_low||) for the dose-response")
    ap.add_argument("--n_eval_prompts", type=int, default=6)
    ap.add_argument("--skip_generation", action="store_true", help="Metric B only (no generation)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from methods.behavioral_transport.transport import train_transport

    data = json.load(open(args.responses))
    model, tok = _load(args.model)

    # ---- activations at layer L, mean over response tokens
    acts, levels = _collect(model, tok, data, args.layer)
    K = int(levels.max()) + 1

    # ---- PCA-64 (validated subspace; keeps the field well-scaled & stable)
    m = acts.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(acts - m, full_matrices=False)
    V = vt[: args.pca]                                     # (k, D)
    Xp = (acts - m) @ V.T                                  # (N, k)
    cents_pca = np.stack([Xp[levels == k].mean(0) for k in range(K)])

    field, info = train_transport(
        torch.tensor(Xp, dtype=torch.float32), torch.tensor(levels).long(), np.arange(K),
        periodic=False, W=K, hidden_dims=[128], epochs=args.epochs, lr=1e-3,
        neighbor_radius=1, w_density=args.w_density, seed=0, device="cpu",
    )

    # ---- METRIC B: on-manifold interpolation, transport (segmented field) vs linear chord
    t_path = _segmented_path(field, cents_pca, K, args.sub_steps)        # (M, k)
    M = t_path.shape[0]
    fr = np.linspace(0.0, 1.0, M)
    l_path = cents_pca[0][None] + fr[:, None] * (cents_pca[K - 1] - cents_pca[0])[None]
    t_off = _min_dist_to_real(t_path, Xp).mean()
    l_off = _min_dist_to_real(l_path, Xp).mean()
    metric_b = {
        "space": f"pca{args.pca}",
        "transport_mean_offmanifold": float(t_off),
        "linear_mean_offmanifold": float(l_off),
        "transport_more_onmanifold": bool(t_off < l_off),
        "offmanifold_ratio_transport_over_linear": float(t_off / (l_off + 1e-9)),
    }
    print("=" * 60)
    print("METRIC B [praise] — on-manifold interpolation (transport vs linear chord)")
    print(f"  mean off-manifold dist:  transport {t_off:.3f}   linear {l_off:.3f}"
          f"   -> transport {'MORE' if t_off < l_off else 'LESS'} on-manifold"
          f"  (ratio {metric_b['offmanifold_ratio_transport_over_linear']:.2f})")
    print("=" * 60)

    result = {"concept": "praise", "layer": args.layer, "n_levels": K,
              "transport_train": {k: (float(v) if isinstance(v, (int, float)) else v)
                                  for k, v in info.items()},
              "metric_b": metric_b}

    # ---- METRIC A: behavioral dose-response along the praise axis (full-space diff-of-means)
    if not args.skip_generation:
        cents_full = np.stack([acts[levels == k].mean(0) for k in range(K)])
        dir_full = cents_full[K - 1] - cents_full[0]
        base_norm = float(np.linalg.norm(dir_full))
        unit = dir_full / (base_norm + 1e-9)
        judge_tmpl = CONCEPTS["praise"]["judge"]
        prompts = [d["prompt"] for d in data][: args.n_eval_prompts]

        # baseline generations once
        base_gen = {}
        for i, p in enumerate(prompts):
            base_gen[i] = _chat_generate(model, tok, [{"role": "user", "content": p}],
                                         max_new_tokens=200, temperature=0.0, seed=i)

        dose = []
        for a in args.alphas:
            delta = unit * (a * base_norm)
            wins, distinct = [], []
            for i, p in enumerate(prompts):
                msgs = [{"role": "user", "content": p}]
                if a == 0.0:
                    gen = base_gen[i]
                else:
                    with _ActAdd(model, args.layer - 1, delta):
                        gen = _chat_generate(model, tok, msgs, max_new_tokens=200,
                                             temperature=0.0, seed=i)
                wins.append(_praise_winrate(model, tok, p, gen, base_gen[i], judge_tmpl, seed=1000 + i))
                distinct.append(_distinct_ratio(gen))
            dose.append({"alpha": a, "praise_winrate_vs_base": float(np.mean(wins)),
                         "distinct_token_ratio": float(np.mean(distinct))})
            print(f"  alpha={a:>4}: praise vs base {np.mean(wins):.2f}   distinct-tok {np.mean(distinct):.2f}")

        wr = [d["praise_winrate_vs_base"] for d in dose]
        monotone = all(wr[i] <= wr[i + 1] + 1e-9 for i in range(len(wr) - 1))
        result["metric_a"] = {"n_prompts": len(prompts), "base_norm": base_norm,
                              "dose_response": dose, "monotone_nondecreasing": bool(monotone)}
        print(f"  monotone praise increase with strength: {monotone}")
        print("=" * 60)

    with open(os.path.join(args.out, "transport_praise.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}/transport_praise.json")


if __name__ == "__main__":
    main()
