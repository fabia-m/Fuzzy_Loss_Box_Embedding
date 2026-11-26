# Experiment Setup

## Ontologies

| Ontology | Domain | Classes | Subclass axioms | Disjoint pairs (asserted) | Entailed disjoint | File |
|----------|--------|---------|-----------------|---------------------------|-------------------|------|
| CIM | Common Information Model | 1,933 | 1,931 | 0 | 0 | `data/TheCimOntology.owl` |
| OEO | Open Energy Ontology | 1,565 | 1,568 | 60 | 4,606 | `data/oeo-full.owl` |
| DOID | Human disease | 14,735 | 17,274 | 1,428 | 4,629 | `data/doid.owl` |
| GO | Gene ontology | 51,937 | 58,529 | 29 | 29,043 | `data/go.owl` |

- Ontology stats are **loader output** (`load_owl`), not catalogue figures.
- Subclass training edges = asserted + transitive closure; disjoint edges =
  asserted + propagation along `subClassOf` (one-level, as implemented).
- CIM declares no disjointness → all disjoint metrics are vacuous (empty by
  design, stated in results, not presented as "perfect disjointness").

## Model

| Parameter | Value | Note |
|-----------|-------|------|
| Representation | axis-aligned box, centre + half-size | half_size = softplus(raw) + 1e-6 |
| Membership | sigmoidal, `σ((h − \|x−μ\|)/τ)` | τ = 0.1 **fixed**; half-size `h`, centre `μ` (paper notation) |
| τ handling | fixed, `learn_temperature=False` | learnable rejected (pilot: degenerate) |
| t-norm | product | — |
| implication | Reichenbach (`1 − a + ab`) | — |
| cardinality | `∏_d (2h_d + πτ)` | documented approximation |
| dims | 2, 4, 6, 8, 10, 15, 20, 25, 30 (seed 42, all ontologies); OEO additionally odd 3–11 at seed 42; multi-seed: CIM/OEO d∈{2,4,6,8,10,15}, DOID d∈{2,4,6,8,9,10,11}, GO d∈{2,6} | per-ontology |

## Loss function (5 terms + arbitration tail)

| Term | Weight | Geometric effect |
|------|--------|------------------|
| subclass containment | **2.0** | child box inside parent |
| disjointness (fuzzy overlap, entailed pairs) | **2.0** | no overlap between disjoint classes |
| disjointness (crisp margin-tail, **entailed** closure) | **0.1** | `softplus(0.02 − sep)`; arbitration on the entailed set: removes the DOID basin at 0.1, collapses OEO d6 containment at 0.25+ |
| size (min-box floor) | 0.1 | prevents box collapse |
| oversize | 0.1 | caps box volume (log-volume target, depth-scaled) |
| sibling term | 0 (dropped) | pull-together fights disjointness on sibling-disjoint pairs; cohesion emerges from containment |

## Training

| Parameter | Value |
|-----------|-------|
| optimiser | AdamW |
| learning rate | 0.05 (fixed across dims) |
| steps | 5,000 (plateau verified; 10k gave no gain) |
| weight decay | 1e-4 |
| gradient clip | max_norm 1.0 |
| batch size | 512 (edge batching, all loss terms) |
| schedule | **PLAIN** — all losses active from step 1 (`schedule=None`; curricula hard-disabled) |
| seed | 42 (all ontologies, dims 2–30); CIM/OEO additionally 4/666/1010 (dims ≤ 15) |
| init | centres N(0, 0.1²) |

## Evaluation

| Metric | Definition |
|--------|-----------|
| Geometric subclass violations | child ⊄ parent in any dim (asserted, closure-weighted); % = count / closure subclass edges |
| Geometric disjoint violations | separation < 0.02 margin; % = count / entailed disjoint edges |
| Fuzzy α95 violations | membership of containment/overlap < 0.95 at probe points; reported as *graded quality*, not correctness |
| Satisfaction | mean satisfaction degree [0, 1], subclass & disjoint |
| **Overhang degree (metric B)** | worst-dimension overhang fraction `max_d (over_d + under_d)/child_width_d`; 0 = contained, 1 = fully out; reported as deg>5%/deg>10% + severity over violated edges |
| Violation severity | metric-B `severity_mean_deg` over violated edges: **0.95–0.99** (all-or-nothing; final baseline) |

Metric B is the graded dual of the conjunctive (min-over-dims) containment
conjunctive semantics; volume-based containment was considered and rejected because as a
mass average it can mask a fully broken dimension.

## Run layout & entry points

```
results_fuzzy_entailed/{ontology}/{run_name}/dim_{N}/
  run_name = product_reichenbach          (seed 42)
  run_name = product_reichenbach_seed{s}  (other seeds)
```

- **Notebook** `fuzzy_model.ipynb` — the main source for experiments: config
  cells, universal run cell (select ontologies/seeds/dims, skip-done resume),
  analysis cells (summarise, overhang degree, plots).
- **`run_fuzzy.py`** (repo root) — the single background training runner, fully
  parameterised: `--ontologies` (comma list or `all`), `--seeds` (comma list
  or `all`), `--dims` (`2,6,10`, `2-30` even subset, or `all`),
  `--results-dir`, `--t-norm`/`--implication`,
  `--dry-run`. Resume-safe (`is_done` skips completed dims).
  Examples: `run_fuzzy.py --dry-run` · `run_fuzzy.py --ontologies go --dims 25,30` ·
  `run_fuzzy.py --t-norm minimum --implication godel --results-dir results_operators`.

## Environment

| | |
|---|---|
| Machine | Apple M4 Max (Mac, arm64) |
| RAM | 36 GB unified |
| OS | macOS 26.6.2 (Darwin 25.6.0) |
| Device | MPS (hard-coded; single machine, no fallback) |
| Python | 3.14.2 |
| PyTorch | 2.14.0 |
| Loader | rdflib, custom OWL extraction |

## Known approximations / notes

- **Cardinality integral uses `2h + πτ` per dim** (exact: `2h + 2τ·ln(1 + e^{−h/τ})`; at h→0 the exact term is `2τ·ln2 ≈ 1.39τ`, not `π·τ ≈ 3.14τ`). Verified numerically: up to 2.27× overestimate at h→0, only 8–30 % error in the trained regime (h ≫ τ); absorbed by the empirically calibrated size weights.
- **Overlap-degree probing has two weaknesses** (diagnostic, not decisive; the crisp geometric metric remains the authority): (1) min-over-dims after max-over-points mixes coordinates across probe points (a "phantom point" never evaluated); (2) only the midpoint + both centres are probed — for strongly asymmetric boxes none may lie in the overlap region (counter-example: A=[−100,4], B=[3,8]), so true overlap can be under-estimated by the loss.
- **Crisp corners [m, M] = c ± h are the half-maximum slice**, not the support: for sigmoidal membership μ(c ± h) = σ(0) = 0.5 with non-zero tails everywhere. [m, M] is the true support only for triangular membership or in the τ → 0 limit. All metrics interpret [m, M] crisply — consistent across runs, but reviewers may ask what the "box" of a fuzzy box embedding is.
- **Entail-count weighting is an unnormalised weighted mean**: `weighted = entail_count[i:end] * per_edge_loss` then `.mean()` — i.e. `mean(w_i·l_i)`, not `Σ w_i·l_i / Σ w_i`, with entail_count = log1p(descendants × ancestors) (mean weights ≈ 5–7). In effect an amplified subclass weight; consistent across all runs, so comparability is unaffected.
