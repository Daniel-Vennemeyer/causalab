"""Diagnostic: is the praise axis a causal generation knob at the RIGHT write-layer?

transport_praise Metric A found zero behavioral effect steering at layer 28 — but layer 28
was chosen to READ praise (Gate B isometry), and it is the output of layer 28/32 with little
downstream computation left + huge residual norms. Steering (writing) usually needs a MIDDLE
layer. This sweeps write-layer x strength: at each layer, take the level5-level0 diff-of-means
direction (mean over response tokens), ActAdd it during generation, and judge praise vs
baseline. If some mid layer moves praise -> the pipeline is fine, wrong write-layer. If no
layer moves it -> praise is readable but not ActAdd-steerable (deeper negative).

Run: python -m analyses.sycophancy_gate_a.praise_steer_layers --responses <out>/responses.json --out <out>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyses.sycophancy_gate_a.gate_a import CONCEPTS, _chat_generate, _judge, _load
from analyses.sycophancy_gate_a.transport_praise import _ActAdd, _distinct_ratio, _praise_winrate


def _collect_multilayer(model, tok, data, layers):
    """Return {layer: (N, D) response-mean activations}, levels (N,), mean hidden norm/layer."""
    import torch

    per = {L: [] for L in layers}
    levels = []
    norms = {L: [] for L in layers}
    for item in data:
        for k, resp in enumerate(item["responses"]):
            conv = [{"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": resp}]
            ids_full = tok.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(model.device)
            ids_pref = tok.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}], add_generation_prompt=True,
                tokenize=True, return_tensors="pt")
            start = int(ids_pref.shape[1])
            with torch.no_grad():
                out = model(ids_full, output_hidden_states=True)
            for L in layers:
                hs = out.hidden_states[L][0]
                seg = hs[start:] if hs.shape[0] > start else hs[-1:]
                v = seg.float().mean(0).cpu().numpy()
                per[L].append(v)
                norms[L].append(float(np.linalg.norm(v)))
            levels.append(k)
    return ({L: np.asarray(per[L]) for L in layers}, np.asarray(levels),
            {L: float(np.mean(norms[L])) for L in layers})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 14, 18, 22])
    ap.add_argument("--alphas", type=float, nargs="+", default=[3.0, 6.0])
    ap.add_argument("--n_eval_prompts", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = _load(args.model)
    data = json.load(open(args.responses))
    acts, levels, hidnorm = _collect_multilayer(model, tok, data, args.layers)
    K = int(levels.max()) + 1
    judge_tmpl = CONCEPTS["praise"]["judge"]
    prompts = [d["prompt"] for d in data][: args.n_eval_prompts]

    base_gen = {i: _chat_generate(model, tok, [{"role": "user", "content": p}],
                                  max_new_tokens=args.max_new_tokens, temperature=0.0, seed=i)
                for i, p in enumerate(prompts)}

    grid = []
    print("=" * 66)
    print("PRAISE STEERING — write-layer x strength (win-rate = MORE praising than base)")
    print(f"{'layer':>6} {'|dir|/|h|':>9} {'alpha':>6} {'praise_wr':>10} {'distinct':>9}")
    for L in args.layers:
        d = acts[L][levels == K - 1].mean(0) - acts[L][levels == 0].mean(0)
        base_norm = float(np.linalg.norm(d))
        unit = d / (base_norm + 1e-9)
        rel = base_norm / (hidnorm[L] + 1e-9)
        for a in args.alphas:
            delta = unit * (a * base_norm)
            wins, distinct = [], []
            for i, p in enumerate(prompts):
                with _ActAdd(model, L - 1, delta):
                    gen = _chat_generate(model, tok, [{"role": "user", "content": p}],
                                         max_new_tokens=args.max_new_tokens, temperature=0.0, seed=i)
                wins.append(_praise_winrate(model, tok, p, gen, base_gen[i], judge_tmpl, seed=1000 + i))
                distinct.append(_distinct_ratio(gen))
            wr, dt = float(np.mean(wins)), float(np.mean(distinct))
            grid.append({"layer": L, "alpha": a, "rel_dir_norm": rel,
                         "praise_winrate_vs_base": wr, "distinct_token_ratio": dt})
            print(f"{L:>6} {rel:>9.3f} {a:>6} {wr:>10.2f} {dt:>9.2f}")
    print("=" * 66)
    best = max(grid, key=lambda g: g["praise_winrate_vs_base"])
    print(f"BEST: layer {best['layer']} alpha {best['alpha']} -> praise win-rate {best['praise_winrate_vs_base']:.2f}")
    print("=" * 66)

    with open(os.path.join(args.out, "praise_steer_layers.json"), "w") as f:
        json.dump({"concept": "praise", "n_levels": K, "n_prompts": len(prompts),
                   "grid": grid, "best": best}, f, indent=2)
    print(f"-> {args.out}/praise_steer_layers.json")


if __name__ == "__main__":
    main()
