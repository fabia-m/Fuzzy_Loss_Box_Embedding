#!/usr/bin/env python3
"""Single flexible training runner for the fuzzy box-embedding repo.

The Jupyter notebook (`fuzzy_model.ipynb`) is the main source for experiments and analysis;
this script is the background launcher for long sweeps that must survive a closed laptop.

    .venv/bin/python run_fuzzy.py --dry-run                          # show the plan
    .venv/bin/python run_fuzzy.py --ontologies oeo,doid --seeds 42,4 --dims 2,4,6
    .venv/bin/python run_fuzzy.py --dims 2-30                        # even ladder
    .venv/bin/python run_fuzzy.py --dims all --ontologies all        # default matrix
    .venv/bin/python run_fuzzy.py --ontologies go --dims 25,30       # GO high-dim tail

Defaults: ontologies all, seeds 42, dims the paper ladder
{2,4,6,8,10,15,20,25,30}, final config (product t-norm · Reichenbach
implication, sub 2.0 / disj 2.0 / contact 0.1 on the ENTAILED closure, no
sibling, PLAIN, 5k steps, tau 0.1 fixed). Resume-safe: `is_done` skips anything
already completed, so re-running the same command continues an interrupted sweep.
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import torch

from functions import FuzzyBoxConfig, is_done, run_dir_name, sweep_dimensions

ONTOLOGIES = {
    "cim": "data/TheCimOntology.owl",
    "oeo": "data/oeo-full.owl",
    "doid": "data/doid.owl",
    "go": "data/go.owl",
}
ALL_SEEDS = [42, 4, 666, 1010]
ALL_DIMS = [2, 4, 6, 8, 10, 15, 20, 25, 30]
DEFAULT_RESULTS_DIR = "results_fuzzy_entailed"


def parse_list(value: str):
    """'2' -> [2] | '2,6,10' -> [2,6,10]; 'all' -> None (caller fills in)."""
    value = value.strip().lower()
    if value == "all":
        return None
    if "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def parse_dims(value: str):
    """Dimensions: '2,6,10' (explicit) | '2-30' (even subset, paper convention)."""
    if value is None or value.strip().lower() == "all":
        return list(ALL_DIMS)
    out = []
    for tok in parse_list(value) or []:
        if "-" in tok:
            a, b = map(int, tok.split("-"))
            lo = a + (a % 2)
            hi = b - (b % 2)
            out.extend(range(lo, hi + 1, 2) if lo <= hi else [a])
        else:
            out.append(int(tok))
    # preserve order, drop duplicates
    return list(dict.fromkeys(out))


def make_cfg(dim, seed, t_norm, implication, steps):
    return FuzzyBoxConfig(
        dim=dim, seed=seed, steps=steps,
        t_norm=t_norm, implication=implication,
        subclass_weight=2.0,
        disjoint_weight=2.0,
        contact_weight=0.1,
        contact_on_entailed=True,  # final config: crisp tail over the ENTAILED closure
        distance_weight=0.0,  # no sibling term
    )


def pct(viol, tot):
    return f"{100.0 * viol / tot:.2f}%" if (viol is not None and tot) else "n/a"


def summary_row(path):
    mf = os.path.join(path, "metrics.json")
    if not os.path.exists(mf):
        return "  (no metrics.json)"
    m = json.load(open(mf))
    return (f"  sub={pct(m.get('subclass_violations_geometric'), m.get('closure_subclass_total')):<8} "
            f"disj={pct(m.get('disjoint_violations_geometric'), m.get('entailed_disjoint_total')):<8} "
            f"sat {m.get('subclass_mean_satisfaction', 0):.3f}/{m.get('disjoint_mean_satisfaction', 0):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ontologies", default="all",
                    help="comma list or 'all' (default) | choices: " + ",".join(ONTOLOGIES))
    ap.add_argument("--seeds", default="42", help="comma list, 'all' -> 42,4,666,1010")
    ap.add_argument("--dims", default="all",
                    help="'2,6,10' | '2-30' (even subset) | 'all' -> " + ",".join(map(str, ALL_DIMS)))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--t-norm", default="product", help="product | minimum | lukasiewicz")
    ap.add_argument("--implication", default="reichenbach",
                    help="godel | lukasiewicz | goguen | reichenbach | kleene_dienes")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only, train nothing")
    args = ap.parse_args()

    ontos_raw = parse_list(args.ontologies)
    ontos = [o.strip().lower() for o in ontos_raw] if ontos_raw else list(ONTOLOGIES)
    bad = [o for o in ontos if o not in ONTOLOGIES]
    if bad:
        ap.error(f"unknown ontologies: {bad} (choose from {', '.join(ONTOLOGIES)})")
    seeds_raw = parse_list(args.seeds)
    seeds = [int(s) for s in seeds_raw] if seeds_raw else list(ALL_SEEDS)
    dims = parse_dims(args.dims)

    print("Final config: %s x %s | sub 2.0 / disj 2.0 / contact 0.1 (entailed), no sibling, PLAIN "
          "| %d steps | %s" % (args.t_norm, args.implication, args.steps, args.device))

    todo = []
    for onto in ontos:
        for dim in dims:
            for seed in seeds:
                cfg = make_cfg(dim, seed, args.t_norm, args.implication, args.steps)
                path = os.path.join(args.results_dir, onto, run_dir_name(cfg, seed))
                if is_done(onto, seed, dim, results_dir=args.results_dir, cfg=cfg):
                    continue
                todo.append((onto, dim, seed, path))

    print(f"\n[{len(todo)} pending / {len(ontos) * len(dims) * len(seeds)} planned] "
          f"-> {args.results_dir}/{{onto}}/{{run}}/dim_{{N}}/\n")
    if args.dry_run:
        for onto, dim, seed, path in todo:
            print(f"  {onto:<4} dim={dim:>2} seed={seed:>4} -> {path}")
        return

    if not todo:
        print("nothing to train.")
        return

    t0 = time.time()
    for j, (onto, dim, seed, path) in enumerate(todo, 1):
        cfg = make_cfg(dim, seed, args.t_norm, args.implication, args.steps)
        print(f"\n[{j}/{len(todo)}] {onto} dim={dim} seed={seed} -> {path}  "
              f"elapsed {time.time() - t0:.0f}s")
        sweep_dimensions(
            ONTOLOGIES[onto],
            dims=[dim],
            device=args.device,
            schedule=None,
            cfg=cfg,
            path=path,          # sweep_dimensions appends dim_{dim} itself
            steps=cfg.steps,
        )
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(summary_row(os.path.join(path, f"dim_{dim}")))

    os.makedirs(args.results_dir, exist_ok=True)
    tsv = os.path.join(args.results_dir, "summary_runs.tsv")
    with open(tsv, "w") as f:
        f.write("ontology\tdim\tseed\tsubclass_viol_pct\tdisjoint_viol_pct\tsub_sat\tdis_sat\n")
        for onto, dim, seed, path in todo:
            mf = os.path.join(path, f"dim_{dim}", "metrics.json")
            m = json.load(open(mf)) if os.path.exists(mf) else {}
            f.write(f"{onto}\t{dim}\t{seed}\t"
                    f"{pct(m.get('subclass_violations_geometric'), m.get('closure_subclass_total'))}\t"
                    f"{pct(m.get('disjoint_violations_geometric'), m.get('entailed_disjoint_total'))}\t"
                    f"{m.get('subclass_mean_satisfaction', float('nan')):.4f}\t"
                    f"{m.get('disjoint_mean_satisfaction', float('nan')):.4f}\n")
    print(f"\nDone in {time.time() - t0:.0f}s. Summary: {tsv}")


if __name__ == "__main__":
    main()
