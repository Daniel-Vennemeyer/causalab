"""Confound #4 — does praise's curvature matter BEHAVIORALLY, or is linear 'good enough'?

Binary win-rate hit 0.96 with a single linear direction, but win-rate can't see transport's
actual claim: steering that stays ON the real-response manifold -> more natural/coherent output.
Curvature bites most at INTERMEDIATE targets, where the straight full-praise chord diverges from
the curved ribbon. So we compare, at matched delta norm at the write-layer, two single-vector
ActAdd directions to raise praise toward a MODERATE level:

  - LINEAR : unit(c_high - c_low)      -- the naive full-praise chord
  - CURVED : unit(c_mid  - c_low)      -- toward the REAL intermediate praise state (on-ribbon)

They differ exactly by praise's curvature (angle between the two vectors). For each we generate
and measure: (1) praise win-rate vs baseline [control -- both should raise praise]; (2) ON-MANIFOLD
distance of the steered generation's own activation to the real topic-controlled praise ribbon
[transport's claim -- CURVED should be lower]; (3) distinct-token ratio [coherence].

Prediction: comparable praise, but CURVED generations sit closer to the real praise manifold and
stay more coherent -- the naturalness advantage win-rate structurally misses.

Run: python -m analyses.sycophancy_gate_a.praise_curved_steer --responses <out>/responses.json --out <out>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyses.sycophancy_gate_a.gate_a import CONCEPTS, _chat_generate, _judge, _load
from analyses.sycophancy_gate_a.gate_b import _response_activation
from analyses.sycophancy_gate_a.praise_geometry_probe import _collect
from analyses.sycophancy_gate_a.transport_praise import _ActAdd, _distinct_ratio, _praise_winrate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=10, help="write layer (steer here)")
    ap.add_argument("--mid_level", type=int, default=3, help="intermediate praise target level")
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 3.0],
                    help="delta norm = alpha * ||c_high - c_low||")
    ap.add_argument("--n_eval_prompts", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = _load(args.model)
    data = json.load(open(args.responses))
    L = args.layer
    A, P, K = _collect(model, tok, data, [L])
    X = A[L]                                             # (P, K, D)
    c = X.mean(0)                                        # (K, D) level centroids
    m_p = X.mean(1)                                      # (P, D) per-prompt baseline
    wc = (X - m_p[:, None, :]).reshape(P * K, -1)        # topic-controlled ribbon (reference cloud)

    lo, mid, hi = 0, args.mid_level, K - 1
    chord = c[hi] - c[lo]
    chord_norm = float(np.linalg.norm(chord))
    v_lin = chord / (chord_norm + 1e-9)
    v_cur = (c[mid] - c[lo]) / (np.linalg.norm(c[mid] - c[lo]) + 1e-9)
    angle = float(np.degrees(np.arccos(np.clip(v_lin @ v_cur, -1, 1))))

    judge_tmpl = CONCEPTS["praise"]["judge"]
    prompts = [d["prompt"] for d in data][: args.n_eval_prompts]

    def onman(gen, p_idx):
        a = _response_activation(model, tok, prompts[p_idx], gen, L)
        return float(np.min(np.linalg.norm((a - m_p[p_idx])[None, :] - wc, axis=1)))

    base_gen, base_onman = {}, []
    for i, p in enumerate(prompts):
        base_gen[i] = _chat_generate(model, tok, [{"role": "user", "content": p}],
                                     max_new_tokens=args.max_new_tokens, temperature=0.0, seed=i)
        base_onman.append(onman(base_gen[i], i))
    base_onman = float(np.mean(base_onman))

    print("=" * 84)
    print(f"CURVED vs LINEAR praise steering @ layer {L}  (curvature angle(v_lin,v_cur) = {angle:.1f} deg)")
    print(f"baseline generation on-manifold dist = {base_onman:.3f}")
    print(f"{'dir':>8} {'alpha':>6} {'praise_wr':>10} {'onmanifold':>11} {'distinct':>9}")
    grid = []
    for name, v in [("linear", v_lin), ("curved", v_cur)]:
        for a in args.alphas:
            delta = v * (a * chord_norm)
            wr, om, dt = [], [], []
            for i, p in enumerate(prompts):
                with _ActAdd(model, L - 1, delta):
                    gen = _chat_generate(model, tok, [{"role": "user", "content": p}],
                                         max_new_tokens=args.max_new_tokens, temperature=0.0, seed=i)
                wr.append(_praise_winrate(model, tok, p, gen, base_gen[i], judge_tmpl, seed=1000 + i))
                om.append(onman(gen, i))
                dt.append(_distinct_ratio(gen))
            row = {"direction": name, "alpha": a, "praise_winrate_vs_base": float(np.mean(wr)),
                   "onmanifold_dist": float(np.mean(om)), "distinct_token_ratio": float(np.mean(dt))}
            grid.append(row)
            print(f"{name:>8} {a:>6} {row['praise_winrate_vs_base']:>10.2f} "
                  f"{row['onmanifold_dist']:>11.3f} {row['distinct_token_ratio']:>9.2f}")
    print("=" * 84)
    # summarize: at matched alpha, curved should be >= praise, <= onmanifold, >= distinct
    for a in args.alphas:
        lin = next(r for r in grid if r["direction"] == "linear" and r["alpha"] == a)
        cur = next(r for r in grid if r["direction"] == "curved" and r["alpha"] == a)
        print(f"alpha={a}: praise lin {lin['praise_winrate_vs_base']:.2f} / cur {cur['praise_winrate_vs_base']:.2f}"
              f" | onmanifold lin {lin['onmanifold_dist']:.2f} / cur {cur['onmanifold_dist']:.2f}"
              f" ({'CURVED more on-manifold' if cur['onmanifold_dist'] < lin['onmanifold_dist'] else 'linear more on-manifold'})"
              f" | distinct lin {lin['distinct_token_ratio']:.2f} / cur {cur['distinct_token_ratio']:.2f}")
    print("=" * 84)

    with open(os.path.join(args.out, "praise_curved_steer.json"), "w") as f:
        json.dump({"concept": "praise", "layer": L, "mid_level": mid, "curvature_angle_deg": angle,
                   "baseline_onmanifold": base_onman, "grid": grid}, f, indent=2)
    print(f"-> {args.out}/praise_curved_steer.json")


if __name__ == "__main__":
    main()
