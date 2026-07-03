"""Sycophancy Gate B (praise) — isometry: is the praise axis a STEERABLE activation
direction, or can the model only JUDGE it?

Gate A showed praise is a consistent 1-D order in the model's judgments. Gate B tests
whether that axis lives in activation space: run the model over each (user, response),
take the mean activation over the RESPONSE tokens (no system prompt -> the praise signal
must come from the response content, not the label), bin by praise-intensity LEVEL
(Gate A's tau=0.72 confirms level tracks the self-ranking), and correlate activation
centroid distances D_X with praise-level distances D_Y. r>~0.5 => represented =>
proceed to 1-D transport; r~0 (like the label-free `activation` arm) => stop.

Run: python -m analyses.sycophancy_gate_a.gate_b --responses <gate_a_out>/responses.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np

from analyses.sycophancy_gate_a.gate_a import _load


def _response_activation(model, tok, user: str, response: str, layer: int) -> np.ndarray:
    """Mean hidden state (layer) over the assistant-RESPONSE tokens of (user, response)."""
    import torch

    conv = [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
    ids_full = tok.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(model.device)
    ids_pref = tok.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True,
        tokenize=True, return_tensors="pt",
    )
    start = int(ids_pref.shape[1])
    with torch.no_grad():
        out = model(ids_full, output_hidden_states=True)
    hs = out.hidden_states[layer][0]                    # (T, H)
    seg = hs[start:] if hs.shape[0] > start else hs[-1:]
    return seg.float().mean(0).detach().cpu().numpy()


def _pca(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[: min(k, vt.shape[0])].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="responses.json from a Gate A run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--pca", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = json.load(open(args.responses))
    model, tok = _load(args.model)

    acts, levels = [], []
    for item in data:
        for k, resp in enumerate(item["responses"]):
            acts.append(_response_activation(model, tok, item["prompt"], resp, args.layer))
            levels.append(k)
    X = np.asarray(acts)
    levels = np.asarray(levels)
    Xp = _pca(X, args.pca)                              # (N, pca)
    K = int(levels.max()) + 1
    cents = np.stack([Xp[levels == k].mean(0) for k in range(K)])  # (K, pca) per-level centroids

    pairs = list(combinations(range(K), 2))
    dx = np.array([np.linalg.norm(cents[i] - cents[j]) for i, j in pairs])   # activation dist
    dy = np.array([abs(i - j) for i, j in pairs], dtype=float)               # praise-level dist
    r = float(np.corrcoef(dx, dy)[0, 1])
    # also: does the 1st PC of the centroids track the praise level? (monotone axis)
    c1 = cents[:, 0] if cents.shape[1] else np.zeros(K)
    tau_pc1 = float(np.corrcoef(c1, np.arange(K))[0, 1])

    passed = r >= 0.5
    verdict = {
        "concept": "praise",
        "n_responses": int(len(levels)),
        "n_levels": K,
        "layer": args.layer,
        "isometry_pearson_r": r,
        "pc1_vs_level_corr": tau_pc1,
        "passed": bool(passed),
        "verdict": (
            "praise IS a steerable activation direction -> proceed to 1-D transport"
            if passed
            else "praise axis NOT clearly represented in activations -> stop / rethink"
        ),
    }
    print("=" * 60)
    print(f"GATE B [praise]: {verdict['verdict']}")
    print(f"  isometry r (D_X activation vs D_Y praise level): {r:.3f}  (>= 0.5?)")
    print(f"  PC1-vs-level corr (is there a monotone axis):    {tau_pc1:.3f}")
    print(f"  -> {args.out}/gate_b.json")
    print("=" * 60)
    with open(os.path.join(args.out, "gate_b.json"), "w") as f:
        json.dump(verdict, f, indent=2)


if __name__ == "__main__":
    main()
