"""Geometric Flow Grounding (Yu et al. 2026) applied to praise steering.

Implements GFG's two core operators on the praise ribbon and asks whether its central
assumption -- that the control-relevant dynamics are TANGENT to the learned data manifold --
holds for behavioral steering:

  1) STATE MANIFOLD via a learned decoder G: Z->X (GFG's state stream), trained on the
     TOPIC-CONTROLLED praise ribbon (GFG's dual-stream logic: topic = static state offset,
     removed by subtracting each prompt's mean; praise = the dynamics we model).
  2) NEURAL TANGENT PROJECTION: J_G(z0) (via autograd) spans the tangent space T = Range(J_G).
     - NTP steering direction = JVP(G, z0, dz_praise) = J_G(z0) @ (enc(c_hi)-enc(c_lo)) -- GFG's
       grounded on-manifold praise velocity at the low-praise state.
     - Projection residual (GFG's verification metric): for each candidate steer direction u,
       tangent_frac = ||P_T u||^2 / ||u||^2 (1 = fully on-manifold, 0 = fully off-manifold).

The decisive question: GFG predicts effective dynamics are tangent (high tangent_frac). Our
confound-#4 result says the effective praise CONTROL axis (linear diff-of-extremes) is ~93° from
the local ribbon tangent -> it should have LOW tangent_frac, while the on-manifold directions
(tangent/NTP) should have HIGH tangent_frac but NOT control praise. If so, GFG's tangency
assumption is violated for behavioral control (representation manifold != control manifold).

Run: python -m analyses.sycophancy_gate_a.gfg_praise --responses <out>/responses.json --out <out>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyses.sycophancy_gate_a.gate_a import CONCEPTS, _chat_generate, _judge, _load
from analyses.sycophancy_gate_a.praise_geometry_probe import _collect
from analyses.sycophancy_gate_a.transport_praise import _ActAdd, _distinct_ratio, _praise_winrate


def _train_ae(wc, d_z, epochs, seed, device):
    """GFG state stream (minimal): autoencoder on the topic-controlled ribbon. Returns dec, enc."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    D = wc.shape[1]
    enc = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, d_z)).to(device)
    dec = nn.Sequential(nn.Linear(d_z, 256), nn.GELU(), nn.Linear(256, D)).to(device)
    X = torch.tensor(wc, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    for _ in range(epochs):
        z = enc(X)
        loss = ((dec(z) - X) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval(); dec.eval()
    return dec, enc, float(loss.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--mid_level", type=int, default=3)
    ap.add_argument("--d_z", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--ae_epochs", type=int, default=800)
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--n_eval_prompts", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--skip_generation", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from torch.autograd.functional import jacobian

    model, tok = _load(args.model)
    data = json.load(open(args.responses))
    L = args.layer
    A, P, K = _collect(model, tok, data, [L])
    X = A[L]                                             # (P, K, D)
    wc = (X - X.mean(1, keepdims=True)).reshape(P * K, -1)   # topic-controlled ribbon
    levels = np.tile(np.arange(K), P)
    cw = np.stack([wc[levels == k].mean(0) for k in range(K)])   # (K, D) ribbon centroids
    lo, mid, hi = 0, args.mid_level, K - 1

    dirs = {"linear": cw[hi] - cw[lo], "mid": cw[mid] - cw[lo], "tangent": cw[1] - cw[lo]}
    chord_norm = float(np.linalg.norm(dirs["linear"]))

    # ---- GFG: train state decoder(s), build tangent projection, NTP direction, residuals
    dev = "cpu"
    per_dz = {}
    ntp_dir = None
    for d_z in args.d_z:
        dec, enc, rec = _train_ae(wc, d_z, args.ae_epochs, seed=0, device=dev)
        with torch.no_grad():
            z_lo = enc(torch.tensor(cw[lo], dtype=torch.float32, device=dev)[None])[0]
            z_hi = enc(torch.tensor(cw[hi], dtype=torch.float32, device=dev)[None])[0]
        dz_praise = (z_hi - z_lo)
        J = jacobian(lambda z: dec(z[None])[0], z_lo).detach().cpu().numpy()   # (D, d_z)
        # tangent projection P_T = J (J^T J)^-1 J^T; tangent_frac(u) = u^T P u / u^T u
        JTJ_inv = np.linalg.pinv(J.T @ J)
        def tfrac(u):
            Pu = J @ (JTJ_inv @ (J.T @ u))
            return float((u @ Pu) / (u @ u + 1e-12))
        # NTP grounded praise velocity = JVP(G, z_lo, dz_praise) = J @ dz_praise
        ntp = J @ dz_praise.detach().cpu().numpy()
        if d_z == args.d_z[-1]:
            ntp_dir = ntp / (np.linalg.norm(ntp) + 1e-9)
        per_dz[d_z] = {"ae_recon": rec,
                       "tangent_frac": {k: tfrac(v) for k, v in dirs.items()},
                       "ntp_tangent_frac": tfrac(ntp),
                       "ntp_angle_vs_linear": float(np.degrees(np.arccos(np.clip(
                           (ntp / np.linalg.norm(ntp)) @ (dirs["linear"] / np.linalg.norm(dirs["linear"])), -1, 1))))}
    dirs["ntp"] = ntp_dir * chord_norm    # give NTP the same norm scale as the chord unit

    units = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in dirs.items()}
    angles = {k: float(np.degrees(np.arccos(np.clip(units[k] @ units["linear"], -1, 1)))) for k in units}

    print("=" * 78)
    print("GFG on praise — is the effective control direction TANGENT to the learned manifold?")
    print(f"angles vs linear:  " + "  ".join(f"{k}={angles[k]:.0f}deg" for k in units))
    for d_z in args.d_z:
        pd = per_dz[d_z]
        tf = pd["tangent_frac"]
        print(f"  d_z={d_z} (AE recon {pd['ae_recon']:.2f}): tangent_frac  "
              f"linear={tf['linear']:.2f} mid={tf['mid']:.2f} tangent={tf['tangent']:.2f}"
              f"  | NTP frac={pd['ntp_tangent_frac']:.2f} NTP-angle-vs-linear={pd['ntp_angle_vs_linear']:.0f}deg")
    print("  (GFG predicts EFFECTIVE dynamics have HIGH tangent_frac; praise control = linear)")
    print("=" * 78)

    result = {"concept": "praise", "layer": L, "angles_vs_linear": angles, "gfg_by_dz": per_dz}

    # ---- confirm behaviorally: does the GFG/NTP direction actually steer praise?
    if not args.skip_generation:
        judge_tmpl = CONCEPTS["praise"]["judge"]
        prompts = [d["prompt"] for d in data][: args.n_eval_prompts]
        base_gen = {i: _chat_generate(model, tok, [{"role": "user", "content": p}],
                                      max_new_tokens=args.max_new_tokens, temperature=0.0, seed=i)
                    for i, p in enumerate(prompts)}
        print(f"{'dir':>8} {'alpha':>6} {'praise_wr':>10} {'distinct':>9}")
        grid = []
        for name in ["linear", "ntp", "mid", "tangent"]:
            u = units[name]
            for a in args.alphas:
                delta = u * (a * chord_norm)
                wr, dt = [], []
                for i, p in enumerate(prompts):
                    with _ActAdd(model, L - 1, delta):
                        gen = _chat_generate(model, tok, [{"role": "user", "content": p}],
                                             max_new_tokens=args.max_new_tokens, temperature=0.0, seed=i)
                    wr.append(_praise_winrate(model, tok, p, gen, base_gen[i], judge_tmpl, seed=1000 + i))
                    dt.append(_distinct_ratio(gen))
                row = {"direction": name, "alpha": a, "praise_winrate_vs_base": float(np.mean(wr)),
                       "distinct_token_ratio": float(np.mean(dt))}
                grid.append(row)
                print(f"{name:>8} {a:>6} {row['praise_winrate_vs_base']:>10.2f} {row['distinct_token_ratio']:>9.2f}")
        print("=" * 78)
        result["generation"] = grid

    with open(os.path.join(args.out, "gfg_praise.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}/gfg_praise.json")


if __name__ == "__main__":
    main()
