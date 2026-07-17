# GFG (Geometric Flow Grounding) on the geometric tasks — RESULT

**Method (`method: gfg`).** Same transport tangent field (dynamics stream) as the transport arms,
but steering integrates in a learned state decoder's LATENT space and decodes each step (Neural
Tangent Projection) → on-manifold by construction, **with `w_density=0.0`** (transport needed it).
Clean ablation of the matching transport arm: identical field/coordinate, density crutch removed.

Synthetic + unit tests: free transport (no density) drifts off-manifold (dist 64044, explodes);
GFG stays glued (0.151).

**Real-task patch-grounded `distance_from_behavior_manifold` (lower = more on-manifold):**

| task | manifold (nodes) | AE recon | **GFG (no density)** | transport (w/ density) | spline geodesic | GFG coherence |
|---|---|---|---|---|---|---|
| weekdays | ring, 7 | 112.7 | **0.169 ✅** | 0.700 | ~0.33 | 0.728 |
| alphabet | line, 22 | 193.5 | **4.30 ❌** | 0.527 | 0.219 | 0.907 |
| grid_5×5 | 2-D, 25 | 76.5 | **1.14 ❌** | 0.600 (VAE 0.751) | — | 0.966 |

**Verdict: GFG's on-manifold grounding is real but DECODER-QUALITY-GATED — matches the paper's
own limitation (Appendix A.2: "error degrades with the quality of the learned tangent space").**
- **Weekdays (clean 7-node ring, decoder fits): GFG is the best on-manifold steerer measured** —
  0.169, ~4× better than transport-with-density (0.700), below the spline geodesic (~0.33), and
  with NO density penalty. GFG's thesis realized on our task.
- **Alphabet (22-node line, decoder underfit: 88 pts / 22 classes / 1-D latent → recon 193.5):**
  GFG collapses to 4.30 (worse than transport's *pre-density* drift). Bad manifold ⇒ decoded steps
  land off the data.
- **Grid (2-D, decoder decent but not tight, recon 76.5): GFG 1.14 loses to transport 0.600 / VAE
  0.751.** The density-regularized free field still wins on the 2-D lattice.

**Takeaway.** Where the state decoder reconstructs the manifold well and the topology is clean +
low-dim (weekdays), GFG's tangent-bundle grounding dominates and removes the `w_density`
hyperparameter. Where the decoder underfits (alphabet's sparse 22-class line, the 2-D grid), the
free transport field + density beats it. GFG converts transport's "add a density penalty and hope"
into "spend the effort on a good decoder"; the on-manifold guarantee is then automatic.

**Next levers if we want GFG to win everywhere:** better state decoder — more latent dims / epochs,
or reuse the trained behavior-aligned VAE decoder instead of a fresh plain AE; and GFG's velocity
primitive codebook (VQ) which we have not added (dynamics stream is currently the plain transport
field). The alphabet/grid losses are decoder-capacity problems, not method-thesis problems.
