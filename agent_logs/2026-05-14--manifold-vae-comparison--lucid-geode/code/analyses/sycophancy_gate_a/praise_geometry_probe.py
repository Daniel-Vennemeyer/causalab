"""Diagnostic: is the praise manifold REALLY flat, or did topic variance / the read-layer /
coarse sampling hide its curvature?

Metric B (transport~linear, ratio 1.08) measured praise geometry ACROSS 16 topics at layer 28.
The off-manifold distances were huge (7+), meaning within-level (topic) scatter swamps the
between-level (praise) signal. This probe removes topic and sweeps layers, field-free:

  For each layer L, with one response per (prompt p, level k):
   - VARIANCE DECOMPOSITION (two-way, balanced P x K): what fraction of activation variance is
     explained by praise LEVEL vs by TOPIC (prompt) vs residual. If topic >> level -> Metric B's
     flatness is a topic-variance artifact, not real flatness.
   - CURVATURE of the praise trajectory, GLOBAL (level centroids) and WITHIN-PROMPT (topic-matched,
     averaged over prompts): arc/chord ratio (1=collinear/flat, >1=curved; arc/chord is
     translation-invariant so the per-topic offset cancels) and the PCA variance spectrum of the
     level centroids (PC1 fraction ~1 => flat).

If within-prompt arc/chord >> 1 (esp. at mid layers) while global is ~1 -> the flat result was
an artifact of measuring across topics at the wrong layer, and transport SHOULD help with a
topic-controlled setup. If arc/chord ~1 everywhere, even within-prompt -> praise is genuinely
~linear and the flat result is real.

Run: python -m analyses.sycophancy_gate_a.praise_geometry_probe --responses <out>/responses.json --out <out>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyses.sycophancy_gate_a.gate_a import _load


def _collect(model, tok, data, layers):
    """(prompt, level)-indexed activations. Returns A{L: (P,K,D)} (one response per cell)."""
    import torch

    P, K = len(data), len(data[0]["responses"])
    A = {L: np.zeros((P, K, 0)) for L in layers}
    buf = {L: [] for L in layers}
    for item in data:
        for resp in item["responses"]:
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
                buf[L].append(seg.float().mean(0).cpu().numpy())
    D = len(buf[layers[0]][0])
    return {L: np.asarray(buf[L]).reshape(P, K, D) for L in layers}, P, K


def _arc_chord(traj):
    """traj (K, D) ordered by level -> arc length / chord length (>=1; 1 = collinear)."""
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    arc = float(steps.sum())
    chord = float(np.linalg.norm(traj[-1] - traj[0]) + 1e-9)
    return arc / chord


def _pca_frac(X):
    """X (K, D) -> variance fraction of PC1 among the (<=K-1) nonzero components."""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    return float(ev[0] / (ev.sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 10, 16, 22, 28])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = _load(args.model)
    data = json.load(open(args.responses))
    A, P, K = _collect(model, tok, data, args.layers)

    rows = []
    print("=" * 82)
    print("PRAISE GEOMETRY PROBE — is the ribbon curved once topic is removed?")
    print(f"{'layer':>5} {'var%_level':>10} {'var%_topic':>10} {'arc/chord_global':>16} "
          f"{'arc/chord_within':>16} {'PC1%_global':>11}")
    for L in args.layers:
        X = A[L]                                   # (P, K, D)
        flat = X.reshape(P * K, -1)
        grand = flat.mean(0)
        c_k = X.mean(0)                            # (K, D) level centroids (topic-averaged)
        m_p = X.mean(1)                            # (P, D) prompt centroids (level-averaged)
        ss_total = float(((flat - grand) ** 2).sum())
        ss_level = float(P * ((c_k - grand) ** 2).sum())
        ss_topic = float(K * ((m_p - grand) ** 2).sum())
        var_level = ss_level / (ss_total + 1e-9)
        var_topic = ss_topic / (ss_total + 1e-9)

        ac_global = _arc_chord(c_k)
        ac_within = float(np.mean([_arc_chord(X[p]) for p in range(P)]))
        pc1_global = _pca_frac(c_k)
        # topic-controlled curvature: subtract each prompt's mean, then average per level
        Xwc = X - m_p[:, None, :]
        pc1_within = float(np.mean([_pca_frac(Xwc[p]) for p in range(P)]))

        rows.append({"layer": L, "var_frac_level": var_level, "var_frac_topic": var_topic,
                     "arc_chord_global": ac_global, "arc_chord_within_prompt": ac_within,
                     "pc1_frac_global": pc1_global, "pc1_frac_within_prompt": pc1_within})
        print(f"{L:>5} {var_level:>10.3f} {var_topic:>10.3f} {ac_global:>16.3f} "
              f"{ac_within:>16.3f} {pc1_global:>11.3f}")
    print("=" * 82)
    print("read: var%_topic >> var%_level => topic swamps praise (Metric B artifact).")
    print("      arc/chord_within >> 1 (esp. mid layers) => praise IS curved when topic-controlled.")
    print("      arc/chord ~1 everywhere => praise genuinely ~linear (flat result real).")
    print("=" * 82)

    with open(os.path.join(args.out, "praise_geometry_probe.json"), "w") as f:
        json.dump({"concept": "praise", "P": P, "K": K, "layers": args.layers, "rows": rows}, f, indent=2)
    print(f"-> {args.out}/praise_geometry_probe.json")


if __name__ == "__main__":
    main()
