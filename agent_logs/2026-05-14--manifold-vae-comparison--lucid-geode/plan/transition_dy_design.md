# (C) Transition-derived `d_y` — design sketch

**Goal:** derive the relational target `d_y(i,j)` from the model's *behavior*, with
the topology (ring/period/ordering) as an **output**, not an input. This is the
honest version of "ground geometry in behavioral consequences" — and the only
label-free signal that *can* recover the weekday ring, because the ring is in the
input→output **mechanism**, not in the static per-class output distributions
(which are near-equidistant one-hots — see REPORT.md "Why Euclidean d_y had to fail").

## Why static behavior can't work (recap)
`d_y` from per-class output distributions ≈ uniform (every weekday pair ≈ √2 apart):
no adjacency signal. The cyclic adjacency only appears when you look at how behavior
**transitions** under input changes. So we must measure transitions.

## Three concrete sources of a transition `d_y` (no topology injected)

### C1 — Counterfactual transition matrix (uses existing CF data; cheap, no extra runs)
The task already generates counterfactuals that resample the input variable. Build a
class×class transition tensor `T[i,j]` = P(model outputs class j | input that maps to
class i is perturbed by the CF generator). Adjacency emerges from the `+N` arithmetic
already in the data: a unit change in `number` moves `result` by one step, so
neighboring classes co-occur in CF (base→cf) pairs far more than distant ones.
`d_y(i,j)` = a distance derived from `T` (e.g. `1 - T_sym` or graph/commute distance on
`T`). The ring falls out of `T` without ever stating "period 7" or the ordering.
- **Inputs:** baseline run's per-example outputs + the CF pairing (both already on disk).
- **Output:** a (W,W) `d_y` matrix → feed to the isometry loss as a precomputed target.
- **Risk:** if the CF generator resamples `all` variables (not single-step), `T` is
  diffuse → weak adjacency. Mitigation: generate single-variable (`number`) ±1 CFs
  specifically for this (a small, targeted dataset), or weight by |Δnumber|=1 pairs.

### C2 — Activation-interpolation behavior (uses the patch machinery)
Steer/interpolate activations from class i toward class j and read the model's output
*along the path*. `d_y(i,j)` = path length in behavior space, or "does the trajectory
pass through intermediate classes?" Adjacent classes have short, clean transitions;
distant ones pass through intermediates. This is exactly what `path_steering` already
measures — reuse `collect_grid_distributions`. The ordering emerges from which classes
lie between which.
- **Inputs:** model + the existing patch/featurizer path (we built it for `patch_eval`).
- **Cost:** GPU (8B forwards), like patch_eval. Most causally grounded.

### C3 — Causal steering distance (intervention effort)
`d_y(i,j)` = magnitude of the activation edit needed to convert class-i behavior into
class-j behavior (e.g. norm of the steering vector / number of steps to flip the
output). Derived purely from the model's response to interventions.

## Implementation hook (already in place)
The isometry loss accepts a relational target via `geometry_distance` /
`geometry_coords`. Add `geometry_distance="precomputed"` that takes a per-example
(or per-class) **precomputed `d_y` matrix** built by C1/C2/C3 in the analysis layer,
and matches the latent pairwise geometry to it. No new method-level topology knowledge —
the analysis computes `d_y` from behavior and passes the matrix.

## Decision rule (after the (B) activation arm runs)
- (B) recovers the ring → structure is in the activations; label-free unsupervised
  manifold learning suffices, C is optional.
- (B) fails → run **C1 first** (cheap, data-only). If the ring emerges from the CF
  transition matrix, we've discovered topology from behavior without injecting it —
  the real result. Escalate to C2 (causal) only if C1 is too diffuse.

## What would count as "solved"
The pipeline outputs the ring (period, ordering) given only activations + behavioral
observations, on a task whose topology was **not** encoded in any loss — and ideally
generalizes to a task where we genuinely don't know the topology (the eventual
sycophancy/refusal target).
