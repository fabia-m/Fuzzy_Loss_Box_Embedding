"""Regenerate result numbers from metrics.json on disk.

Emits:
  1. summary_all.tsv        -- one row per completed run (all seeds/dims)
  2. Appendix sweep rows    -- seed-42 table, LaTeX + Markdown (- = CIM disj n/a, '--' = not done)
  3. Robustness rows        -- multi-seed mean+/-std and median (LaTeX), for dims with all 4 seeds
  4. Basin / spike cells    -- DOID basin (d10) and OEO spike annotations

Reads ANY run-name directory (Ch. 4 `product_reichenbach*` or Ch. 5 operator
pairs `{t_norm}_{implication}`), so it serves both chapters:

Usage: .venv/bin/python scripts/generate_numbers.py
       .venv/bin/python scripts/generate_numbers.py --results-dir results_operators
"""
import argparse
import glob
import json
import os
import statistics as st

RESULTS = os.environ.get("RESULTS_DIR", "results_fuzzy_nosibling")
DIMS = [2, 4, 6, 8, 10, 15, 20, 25, 30]
SEEDS = [42, 4, 666, 1010]
ONTOS = ["cim", "oeo", "doid", "go"]


def load_rows():
    """(onto, seed, dim) -> metric dict from metrics.json, any run-name dir."""
    rows = {}
    for mf in glob.glob(f"{RESULTS}/*/*/dim_*/metrics.json"):
        p = mf.split(os.sep)
        onto, run = p[-4], p[-3]
        seed = int(run.split("seed")[-1]) if "seed" in run else 42
        dim = int(p[-2].replace("dim_", ""))
        rows[(onto, seed, dim)] = json.load(open(mf))
    return rows


def sub_pct(m):
    t = m.get("closure_subclass_total") or 0
    return 100.0 * (m.get("subclass_violations_geometric") or 0) / t if t else None


def disj_pct(m):
    t = m.get("entailed_disjoint_total") or 0
    return 100.0 * (m.get("disjoint_violations_geometric") or 0) / t if t else None


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def mean_std(vals):
    return st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0


def main():
    global RESULTS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=RESULTS,
                    help="results tree (default: results_fuzzy_nosibling, or $RESULTS_DIR)")
    RESULTS = ap.parse_args().results_dir
    rows = load_rows()

    # ---------- 1. summary_all.tsv ----------
    with open(os.path.join(RESULTS, "summary_all.tsv"), "w") as f:
        f.write("ontology\tdim\tseed\tsubclass_viol_pct\tdisjoint_viol_pct\tsub_sat\tdis_sat\n")
        for onto in ONTOS:
            for seed in SEEDS:
                for dim in DIMS:
                    m = rows.get((onto, seed, dim))
                    if not m:
                        continue
                    f.write(f"{onto}\t{dim}\t{seed}\t{fmt(sub_pct(m))}\t{fmt(disj_pct(m))}\t"
                            f"{m.get('subclass_mean_satisfaction', float('nan')):.4f}\t"
                            f"{m.get('disjoint_mean_satisfaction', float('nan')):.4f}\n")
    print(f"wrote {RESULTS}/summary_all.tsv")

    # ---------- 2. seed-42 sweep rows ----------
    print("\n=== SEED-42 SWEEP TABLE (LaTeX) ===")
    for onto in ONTOS:
        cells = []
        for d in DIMS:
            m = rows.get((onto, 42, d))
            if not m:
                cells.append("--/--" if onto != "cim" else "--/--")
                continue
            s, dj = sub_pct(m), disj_pct(m)
            cells.append(f"{s:.2f}/--" if onto == "cim" else f"{s:.2f}/{dj:.2f}")
        print(f"{onto.upper()} & " + " & ".join(cells) + r" \\")

    print("\n=== SEED-42 SWEEP TABLE (Markdown) ===")
    for onto in ONTOS:
        cells = []
        for d in DIMS:
            m = rows.get((onto, 42, d))
            if not m:
                cells.append("–/–")
                continue
            s, dj = sub_pct(m), disj_pct(m)
            cells.append(f"{s:.2f}/–" if onto == "cim" else f"{s:.2f}/{dj:.2f}")
        print(f"| {onto.upper()} | " + " | ".join(cells) + " |")

    # ---------- 3. robustness rows (all-4-seed dims) ----------
    print("\n=== ROBUSTNESS (LaTeX) ===")
    TABLE_DIMS = [2, 4, 6, 8, 10, 15]  # paper table columns (tab:robustness)
    for onto in ["oeo", "doid", "go"]:
        for stat, fn in [("mean±std", lambda v: f"{mean_std(v)[0]:.2f}±{mean_std(v)[1]:.2f}"),
                         ("median", lambda v: f"{st.median(v):.2f}")]:
            cells = []
            for d in TABLE_DIMS:
                if not all((onto, s, d) in rows for s in SEEDS):
                    cells.append(r"\textendash")
                    continue
                subs = [sub_pct(rows[(onto, s, d)]) for s in SEEDS]
                dis  = [disj_pct(rows[(onto, s, d)]) for s in SEEDS]
                cells.append(f"{fn(subs)}/{fn(dis)}")
            print(f"{onto.upper()} & {stat} & " + " & ".join(cells) + r" \\")
        dims = [d for d in TABLE_DIMS
                if all((onto, s, d) in rows for s in SEEDS)]
        print(f"  # dims with all 4 seeds ({onto}): d" + ", d".join(map(str, dims)))

    # ---------- 4. spike / basin annotation numbers ----------
    print("\n=== ANNOTATIONS ===")
    for d in [10, 15]:  # OEO spikes
        v = [disj_pct(rows[("oeo", s, d)]) for s in SEEDS if ("oeo", s, d) in rows]
        if v:
            m, sd = mean_std(v)
            print(f"OEO ms disj d{d}: {m:.2f}±{sd:.2f}  (cells: {[f'{x:.2f}' for x in v]})")
    # DOID basin cells: per-seed worst disj in low-dim window
    for s in SEEDS:
        worst = None
        for d in [2, 4, 6, 8, 9, 10, 11]:
            m = rows.get(("doid", s, d))
            if m:
                v = disj_pct(m)
                if v is not None and (worst is None or v > worst[1]):
                    worst = (d, v)
        if worst:
            print(f"DOID s{s} basin: d{worst[0]} {worst[1]:.2f}%")
    # DOID ms disj at d2/d6/d10 for the "median 0.00%" claim
    for d in [2, 6, 10]:
        v = [disj_pct(rows[("doid", s, d)]) for s in SEEDS]
        print(f"DOID ms disj d{d}: cells {[f'{x:.2f}' for x in v]}  median {st.median(v):.2f}")

    # ---------- sanity ----------
    print(f"\ntotal runs on disk: {len(rows)}")


if __name__ == "__main__":
    main()
