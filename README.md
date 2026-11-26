# Fuzzy Learning for Box Embeddings

Code for the paper *"Fuzzy Learning for Box
Embeddings"* (Martens et al.).

Fuzzy-logic box embeddings for ontological knowledge representation: every
OWL class is embedded as an axis-aligned box (centre μ, softplus half-size h)
and fuzzy semantics — sigmoidal membership, t-norm-based containment and
overlap, residuated-style implication — turn the ontology's axioms into
smooth, bounded constraint-satisfaction objectives whose values are
interpretable as graded truth (satisfaction) degrees in [0, 1].

## Highlights

- **Satisfaction degrees, not penalties.** Losses are graded and always
  differentiable, in contrast to hinge/penalty-based box embedding objectives.
- **Fixed operator pair for experimental control:** product t-norm ×
  Reichenbach implication, sigmoidal membership, fixed temperature
  τ = 0.1 (a learnable temperature was rejected in a pilot).
- **Crisp contact tail over the *entailed* disjointness closure**
  (`0.1 · softplus(0.02 − sep)`): removes the seed-dependent disjointness
  basin cells at negligible containment cost.
- **No sibling term.** The sibling loss was probed and removed on evidence
  (it degrades OBO-style ontologies); the code supports it for ablation only.

## Reproducing the reported results

Final configuration (all reported runs; also saved per-run in
`results_fuzzy_entailed/{ontology}/{run}/dim_{N}/config.json`):

| Setting | Value |
|---|---|
| T-norm / implication | product / Reichenbach |
| Membership | sigmoidal, τ = 0.1 fixed |
| Weights | subclass 2.0, disjoint 2.0, contact 0.1 (entailed), oversize 0.1, min-box 0.1, no sibling |
| Optimiser | AdamW, lr 0.05, weight decay 1e-4 |
| Training | 5,000 steps, batch 512, plain (non-curriculum) schedule |
| Seeds | 42 (full ladder), 4, 666, 1010 (multi-seed subsets) |
| Dimensions | {2, 4, 6, 8, 10, 15, 20, 25, 30} (OEO also 3, 5, 7, 9, 11) |

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# download the ontologies — see data/README.md
.venv/bin/python run_fuzzy.py --dry-run            # show the full 104-run plan
.venv/bin/python run_fuzzy.py                       # everything pending (resume-safe)
.venv/bin/python generate_numbers.py        # summary_all.tsv + paper-table rows
```

`run_fuzzy.py` is resume-safe: completed runs are skipped, so re-running the
same command continues an interrupted sweep. Select any subset with
`--ontologies`, `--seeds`, `--dims` (e.g. `--ontologies oeo,doid --seeds
42,4 --dims 2,6,10`); `--t-norm`/`--implication`/`--results-dir` reach the
operator study (`results_operators/`). The Jupyter notebook
(`fuzzy_model.ipynb`) are the interactive
development and analysis sources.

## Results (final baseline, seed 42)

Geometric violation percentages at dims 2 / 6 / 10 (relative to the entailed
closure), from `results_fuzzy_entailed/`:

| Ontology | Classes | subclass % (d2/d6/d10) | disjoint % (d2/d6/d10) |
|---|---|---|---|
| CIM | 1,933 | 0.00 / 0.00 / 0.00 | — (no disjoint axioms) |
| OEO | 1,565 | 2.46 / 1.14 / 5.64 | 0.15 / 0.00 / 0.00 |
| DOID | 14,735 | 0.72 / 2.95 / 3.12 | 0.00 / 0.00 / 0.00 |
| GO | 51,937 | 0.25 / 0.25 / 2.59 | 0.00 / 0.00 / 0.00 |

Multi-seed robustness: the same final config on seeds {4, 666, 1010} over CIM
and OEO (dims ≤ 15), DOID (dims 2–11) and GO (dims 2, 6) — mean ± std rows
are emitted by `scripts/generate_numbers.py`. Known per-seed variance (e.g.
OEO d6 disjoint ≈ 5 % at seed 1010) is reported in the paper's robustness
narrative. The loss-term ablation (`scripts/run_ablation.py`) shows subclass
and disjoint terms are load-bearing while the volume regularisers are
marginal at the DOID d4 optimum — but the contact tail itself is load-bearing
at GO scale.

## Repository layout

```
run_fuzzy.py                    single flexible training runner (resume-safe)
functions.py                    core module: model, t-norms, implications, losses, loader
fuzzy_model.ipynb               main development notebook (interactive runs + analysis)
generate_numbers.py             analysis runner
plots/                          figures
EXPERIMENT_SETUP.md             fixed experimental setup, ontology stats, provenance
FORMULAS_APPENDIX.md            formal definitions of every quantity used in the paper
data/                           OWL ontologies — see data/README.md for sources
```

## Reproducibility notes

- Every run directory stores `config.json` (exact hyperparameters),
  `metrics.json` (evaluation + metadata incl. ontology SHA-256, git revision,
  timestamps, training time), `training_curve.json`, plus per-axiom
  `satisfaction_degrees.csv` / `calibration_pairs.csv` for calibration
  analysis.
- Heavy per-run artefacts (checkpoints `model.pt`, `data.pkl`) are not
  distributed in this repository.
- `config.json` is the only reliable discriminator between runs: run
  directory names encode `{t_norm}_{implication}[_seed{N}]` only.

## Licence

Code: MIT — see [LICENSE](LICENCE).
