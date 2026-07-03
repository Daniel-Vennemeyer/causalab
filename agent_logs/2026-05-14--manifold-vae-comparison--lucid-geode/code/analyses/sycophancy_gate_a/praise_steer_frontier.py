"""Confound #4, done right — praise-vs-naturalness FRONTIER (kills the overshoot + circular-metric
confounds in praise_curved_steer).

praise_curved_steer scaled BOTH directions to alpha*||c_high-c_low|| with alpha=2-3, which
overshoots the intermediate-target direction's natural magnitude (||c_mid-c_low||~0.5x) by 4-6x
-> it flew off-manifold/degenerate through no fault of the geometry. And it judged naturalness by
activation distance (circular). Here we:
  - sweep magnitude FINELY (incl. small, on-target values) for each direction, and
  - judge naturalness by TEXT PERPLEXITY (mean NLL of the generated response under the UNSTEERED
    model) -- lower = more natural text, independent of activation geometry.
Then compare the praise-vs-perplexity frontier: at a given achieved praise level, which direction
produces more natural text? Frontier comparison is immune to any single-magnitude (overshoot) bias.

Directions (raw level centroids; the c_k - c_lo difference cancels the shared topic offset):
  LINEAR = unit(c_high - c_lo)   |   MID = unit(c_mid - c_lo)   |   TANGENT = unit(c_lo+1 - c_lo)
(MID/TANGENT are the naive 'aim at an intermediate/local on-ribbon state' heuristics; still single
vectors, so this tests aim-target, NOT a true curved trajectory -- that needs trajectory patching.)

Run: python -m analyses.sycophancy_gate_a.praise_steer_frontier --responses <out>/responses.json --out <out>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyses.sycophancy_gate_a.gate_a import CONCEPTS, _chat_generate, _judge, _load
from analyses.sycophancy_gate_a.praise_geometry_probe import _collect
from analyses.sycophancy_gate_a.transport_praise import _ActAdd, _distinct_ratio, _praise_winrate


def _nll(model, tok, prompt, response):
    """Mean negative log-likelihood of the RESPONSE tokens under the (unsteered) model."""
    import torch

    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    ids = tok.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(model.device)
    pref = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   tokenize=True, return_tensors="pt")
    start = int(pref.shape[1])
    labels = ids.clone()
    labels[:, :start] = -100
    with torch.no_grad():
        out = model(ids, labels=labels)
    return float(out.loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--mid_level", type=int, default=3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--dirs", nargs="+", default=["linear", "mid", "tangent"])
    ap.add_argument("--n_eval_prompts", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = _load(args.model)
    data = json.load(open(args.responses))
    L = args.layer
    A, P, K = _collect(model, tok, data, [L])
    c = A[L].mean(0)                                     # (K, D) level centroids
    chord_norm = float(np.linalg.norm(c[K - 1] - c[0]))
    dirs = {
        "linear": (c[K - 1] - c[0]),
        "mid": (c[args.mid_level] - c[0]),
        "tangent": (c[1] - c[0]),
    }
    units = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in dirs.items()}
    angles = {k: float(np.degrees(np.arccos(np.clip(units[k] @ units["linear"], -1, 1))))
              for k in dirs}

    judge_tmpl = CONCEPTS["praise"]["judge"]
    prompts = [d["prompt"] for d in data][: args.n_eval_prompts]
    base_gen = {i: _chat_generate(model, tok, [{"role": "user", "content": p}],
                                  max_new_tokens=args.max_new_tokens, temperature=0.0, seed=i)
                for i, p in enumerate(prompts)}
    base_nll = float(np.mean([_nll(model, tok, prompts[i], base_gen[i]) for i in range(len(prompts))]))

    print("=" * 82)
    print(f"PRAISE STEER FRONTIER @ layer {L}   (angle vs linear: "
          f"mid={angles['mid']:.0f}deg tangent={angles['tangent']:.0f}deg)")
    print(f"baseline response NLL (naturalness) = {base_nll:.3f}   (delta norm unit = ||c_hi-c_lo||)")
    print(f"{'dir':>8} {'alpha':>6} {'praise_wr':>10} {'nll':>8} {'d_nll':>7} {'distinct':>9}")
    grid = []
    for name in args.dirs:
        u = units[name]
        for a in args.alphas:
            delta = u * (a * chord_norm)
            wr, nll, dt = [], [], []
            for i, p in enumerate(prompts):
                with _ActAdd(model, L - 1, delta):
                    gen = _chat_generate(model, tok, [{"role": "user", "content": p}],
                                         max_new_tokens=args.max_new_tokens, temperature=0.0, seed=i)
                wr.append(_praise_winrate(model, tok, p, gen, base_gen[i], judge_tmpl, seed=1000 + i))
                nll.append(_nll(model, tok, p, gen))
                dt.append(_distinct_ratio(gen))
            row = {"direction": name, "alpha": a, "praise_winrate_vs_base": float(np.mean(wr)),
                   "nll": float(np.mean(nll)), "delta_nll": float(np.mean(nll)) - base_nll,
                   "distinct_token_ratio": float(np.mean(dt))}
            grid.append(row)
            print(f"{name:>8} {a:>6} {row['praise_winrate_vs_base']:>10.2f} {row['nll']:>8.3f} "
                  f"{row['delta_nll']:>+7.3f} {row['distinct_token_ratio']:>9.2f}")
    print("=" * 82)
    # frontier read: for each direction, best naturalness (min d_nll) among rows that reach praise>=0.75
    print("frontier @ praise_wr >= 0.75 (naturalness cost = delta_nll; lower is better):")
    for name in args.dirs:
        hits = [r for r in grid if r["direction"] == name and r["praise_winrate_vs_base"] >= 0.75]
        if hits:
            best = min(hits, key=lambda r: r["delta_nll"])
            print(f"  {name:>8}: reaches praise {best['praise_winrate_vs_base']:.2f} at alpha {best['alpha']}"
                  f" with delta_nll {best['delta_nll']:+.3f}, distinct {best['distinct_token_ratio']:.2f}")
        else:
            print(f"  {name:>8}: never reaches praise_wr 0.75 in the swept range")
    print("=" * 82)

    with open(os.path.join(args.out, "praise_steer_frontier.json"), "w") as f:
        json.dump({"concept": "praise", "layer": L, "angles_vs_linear": angles,
                   "baseline_nll": base_nll, "grid": grid}, f, indent=2)
    print(f"-> {args.out}/praise_steer_frontier.json")


if __name__ == "__main__":
    main()
