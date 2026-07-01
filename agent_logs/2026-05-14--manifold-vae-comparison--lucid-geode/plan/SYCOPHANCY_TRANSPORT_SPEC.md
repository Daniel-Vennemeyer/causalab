# Spec: transport steering on a social behavior (sycophantic praise) via self-ranking

## Objective
Test whether the **transport** steering method (a decoder-free tangent field + density
force, validated on weekdays/months/alphabet) extends to a **social behavior** —
sycophantic praise — using the **target model's own 1-D self-ranking** as the
behavioral coordinate. Structured as a **gated de-risking sequence**: each gate can
kill the experiment cheaply before the next investment.

## Why this design (from the session)
- On-manifold steering needs a global manifold structure **or** an explicit density
  force. A social trait has no global parametric manifold → transport + `w_density`
  is the viable route.
- The coordinate should be the model's **own operationalization** (self-ranking), not
  an external/"general" notion — a geometry the model doesn't represent isn't
  steerable (our label-free `activation` arm: isometry ≈ 0).
- 1-D self-ranking defines the coordinate; the multi-dim "signature" is only for
  cross-effects. Keep it 1-D for the coordinate; measure covariates at eval.

## Model note
Sycophancy is a conversational behavior — use an **instruct/chat target** (e.g.
Llama-3.1-8B-Instruct), not the base `llama31_8b` used elsewhere. All judging uses the
**same target model** as an LLM-judge (self-operationalization).

## Components — reuse vs new
- **Reuse:** `methods/behavioral_transport` (1-D field + integrate + `w_density`);
  `_classify_topology` / `discovered_order` (topology gate on the ranking graph);
  the isometry machinery (dim-agnostic); PCA subspace collection.
- **New:** sycophancy prompt/response dataset; pairwise self-ranking + aggregation;
  a **generate-and-judge** eval harness (sequence-level, not next-token); covariate
  judges.

---

## Pipeline

### D1 — Data
- **Prompts (~50–100):** user turns inviting evaluation/feedback, a mix with
  *flawed* content (to create praise-vs-honesty tension). Topic-varied.
- **Responses:** per prompt, generate K responses spanning a sycophancy range via
  temperature + light system-prompt nudging ("be brutally honest" … "be maximally
  encouraging"). Keep responses **matched within a prompt** to control topic
  confounds. ~hundreds–few thousand (prompt, response) rows.

### S1 — Self-ranking (target model as judge)
- **Pairwise** comparisons ("which response is more sycophantic praise?"), primarily
  **within the same prompt** (topic-controlled), plus a bridging subset across prompts
  for a global scale. Repeat each comparison ≥3× for consistency stats.
- Aggregate pairwise → a 1-D ranking (Bradley–Terry / Elo). `z` = rank per response.
- Persist the **pairwise comparison graph** (for Gate A).

### GATE A — is sycophancy 1-D for THIS model?  (cheapest kill)
- From the pairwise graph: self-consistency (Kendall τ across repeats), **cycle rate**,
  transitivity. Run `_classify_topology` / `discovered_order` on the thresholded graph.
- **PASS:** high self-consistency (τ ≳ 0.7) and a dominant transitive 1-D order → use
  the ranking as `z`, proceed 1-D.
- **FAIL:** intransitive / cyclic / branched → the model itself says sycophancy is
  **multi-mode** (cf. Vennemeyer et al. 2025) → STOP the 1-D path; route to an atlas
  (separate effort). *This is a real, publishable result either way.*

### A1 — Activations
- Run the target model over (prompt + response); collect activations at a mid-late
  layer (sweep a few; pick by Gate B), last-token or mean-over-response. PCA-64.
- Bin responses by rank into W≈10–15 quantile bins → per-bin **centroids** (analogue
  of per-class centroids); `z` = bin index.

### GATE B — isometry: is the ranked axis a steerable activation direction?
- `D_X` = activation centroid distances; `D_Y` = ranking distances |rank_i − rank_j|
  (and/or a behavior manifold fit over the bins). Reuse the isometry machinery.
- **PASS:** isometry r meaningfully > 0 (≳ 0.5) at some layer → the self-ranked axis is
  represented; proceed.
- **FAIL:** r ≈ 0 (like the label-free arm) → the model can *judge* sycophancy but it
  isn't a clean activation direction → no transport will steer it. STOP.

### T1 — Transport (1-D field along the ranking)
- `train_transport` with `ranks` = self-ranking bins, real→real adjacent-rank pairs,
  **`w_density > 0`** (essential for a social trait). Field `v_θ(h)` on PCA activations.
- Steer: integrate from a low-sycophancy centroid toward high (and reverse).

### E1 — Eval (generate-and-judge + covariates)
- Sequence-level: patch the steered activation, **generate** a response, and judge:
  1. **sycophancy** rises monotonically along the steered path (slope, monotonicity);
  2. **covariates** (eval-only, keeps transport 1-D): honesty/accuracy, fluency,
     verbosity — the cross-effect *measurement*;
  3. **naturalness**: coherence + distance to the real-response activation manifold.
- **Baseline:** linear steering (add the sycophancy direction) — does transport give
  smoother, more on-manifold, less side-effecting control?
- **Upgrade path (only if 1-D steering drags honesty):** factored Jacobian transport
  with covariates in the geometry (steer sycophancy, hold honesty on its level-set).

## Success criterion
Transport steering monotonically moves the **target model's own sycophancy ranking** of
its generated responses — more smoothly / more naturally / with less honesty-drift than
linear — demonstrating geometry-aware steering on a social behavior with a **self-defined
1-D coordinate**.

## Risks
- **Judge reliability** drives everything → repeats + consistency stats (Gate A doubles
  as the judge-reliability check).
- **Judgment axis ≠ generation geometry** → Gate B is the guard.
- **Observational entanglement** (1-D axis = sycophancy + correlates) → covariate
  measurement at eval; factored transport only if needed.
- **"Natural" ill-defined** → density force needs a trustworthy real-response reference
  set (the D1 response pool).
- **Base vs instruct** → must use an instruct target.

## Gated order (stop-early)
1. D1 + S1 + **GATE A** (topology/consistency) — cheapest; kills if multi-mode.
2. A1 + **GATE B** (isometry) — kills if axis not steerable.
3. T1 + E1 — only if A and B pass.

## GATE A — RESULT (Llama-3.1-8B-Instruct, 16 prompts × 6 levels, 3 repeats)
- **agreement 0.815** (judge reliable, > 0.70) — cycles are NOT noise.
- **cycle_rate 0.162** (> 0.10 → FAIL) — real but mild intransitivity (random ≈ 0.25, clean 1-D ≈ 0).
- **tau vs system-level spectrum 0.500** — the model doesn't treat our 6 nudging levels as a monotonic sycophancy scale.
- **Verdict: NOT clean 1-D.** A reliable judge that still can't totally-order sycophancy ⇒ distinct modes (flattery / excessive agreement / false deference / warmth) that aren't comparable on one axis — reproducing "sycophancy is not one thing" from the model's own judgments. Do NOT build 1-D transport on this coordinate.
- **Leading cause (testable):** the 6 system levels conflate modes, so cross-mode comparisons are ambiguous → cycles + low tau. **Next: narrow the judge + generation to a SINGLE mode (e.g. excessive agreement with the user's claim) and re-run Gate A.** If a single mode is cleanly 1-D → atlas = per-mode 1-D transport (reuse existing transport). If single modes are still intransitive → genuine multi-dim, needs a 2-D+ coordinate (MDS on the preference matrix to estimate intrinsic dim).
