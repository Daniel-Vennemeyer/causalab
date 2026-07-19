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

## UPDATE — VAE decoder (`gfg_decoder: vae`) REFUTES the decoder-capacity hypothesis
Wired the reconstruction-only behavior-aligned VAE (`flat_vae`, `w_recon=1`/`w_kl=0.1`, all else 0)
as the state manifold (`VAEStateDecoder`, `arm_label=gfg_vae`). Reconstruction improved 4–7× on
every task — yet on-manifold steering got WORSE on every task:

| task | recon AE→VAE | **dist AE-GFG → VAE-GFG** | coherence VAE | transport | spline |
|---|---|---|---|---|---|
| weekdays (ring) | 112 → 25 | 0.169 → **1.41** ❌ | 0.697 | 0.700 | ~0.33 |
| alphabet (line) | 193 → 41 | 4.30 → **4.38** ❌ | 0.899 | 0.527 | 0.219 |
| grid (2-D) | 76 → 17 | 1.14 → **1.68** ❌ | 0.968 | 0.600 | — |

**Conclusion: GFG steering quality is NOT bottlenecked by reconstruction fidelity.** Cleanest
evidence = alphabet (a line, no topology confound): recon 193→41 but distance unchanged (4.30→4.38).
GFG needs a decoder whose LATENT AXIS is aligned with the behavioral steering direction; recon-only
training aligns the latent with variance, not behavior, and the KL/standardization that buy better
recon scramble the behavioral ordering → pulling the transport velocity back through that latent
traverses the manifold poorly. The plain AE's WORSE recon came with a latent that traced the
behavioral coordinate more faithfully (hence its weekdays win 0.169). **Two confounds/levers
identified:** (1) topology — weekdays is a ring but I used a 1-D UNSTRUCTURED latent, which must fold
(likely the weekdays blow-up); a `topology=s1` latent is the correct test. (2) behavioral alignment
— the state decoder must be trained so its latent parametrizes the behavioral axis (e.g. with the
isometry/behavior loss), departing from GFG's pure recon-only `L_topo`.

**Net finding across both GFG runs:** GFG's on-manifold grounding *works* (weekdays AE 0.169 beats
everything) but is **fragile** — it depends on the decoder's latent being behaviorally aligned AND
topology-correct, neither of which reconstruction quality ensures. Where those align (weekdays AE),
GFG dominates; otherwise the density-regularized free transport field (or the spline) is more robust.
The transport field's `w_density` is a blunt but reliable on-manifold force; GFG trades it for a
decoder-alignment requirement that is easy to get wrong.
