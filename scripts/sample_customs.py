#!/usr/bin/env python3
"""
Step 1 of the pipeline. Reads data/raw/<year>.csv and writes

    data/pool/balanced_pool.csv   per-chapter reservoirs (<= R rows per chapter per year)
    data/pool/natural_sample.csv  uniform reservoir (K rows per year, natural distribution)
    data/pool/sampling_log.json   seed, parameters, per-year population counts per chapter

Usage:
    python scripts/sample_customs.py --data-dir data/raw --out-dir data/pool
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsclf.sampling import OUT_COLUMNS, sample_file
from hsclf.tabular import read_csv, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/pool")
    ap.add_argument("--per-chapter", type=int, default=400, help="R: max rows per chapter per year")
    ap.add_argument("--k-natural", type=int, default=5000, help="uniform rows per year")
    ap.add_argument("--seed", type=int, default=170)
    ap.add_argument("--years", default="2012-2025")
    a = ap.parse_args()

    lo, _, hi = a.years.partition("-")
    years = range(int(lo), int(hi or lo) + 1)
    per_year_dir = os.path.join(a.out_dir, "years")
    os.makedirs(per_year_dir, exist_ok=True)
    # Each year is sampled on its own and saved to data/pool/years/, so you can download,
    # sample and delete one raw file at a time. The merged files below are rebuilt every run.
    for y in years:
        path = os.path.join(a.data_dir, f"{y}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        t0 = time.time()
        rng = random.Random(a.seed * 10000 + y)  # one reproducible RNG per year
        p, n, st = sample_file(path, a.per_chapter, a.k_natural, rng, y)
        st["pool_rows"], st["natural_rows"] = len(p), len(n)
        st["params"] = {"seed": a.seed, "per_chapter": a.per_chapter, "k_natural": a.k_natural}
        write_csv(os.path.join(per_year_dir, f"pool_{y}.csv"), p, OUT_COLUMNS)
        write_csv(os.path.join(per_year_dir, f"natural_{y}.csv"), n, OUT_COLUMNS)
        with open(os.path.join(per_year_dir, f"log_{y}.json"), "w") as fh:
            json.dump(st, fh, indent=2)
        print(f"[{y}] eligible={st['eligible']:,} chapters={len(st['chapter_counts'])} "
              f"pool={len(p):,} natural={len(n):,} malformed={st['malformed']} "
              f"bad_chapter={st['bad_chapter']} ({time.time() - t0:.0f}s)")

    pool, nat, log = [], [], {"seed": a.seed, "per_chapter": a.per_chapter, "k_natural": a.k_natural, "years": {}}
    for y in range(1990, 2100):
        lp = os.path.join(per_year_dir, f"log_{y}.json")
        if not os.path.exists(lp):
            continue
        pool += read_csv(os.path.join(per_year_dir, f"pool_{y}.csv"))
        nat += read_csv(os.path.join(per_year_dir, f"natural_{y}.csv"))
        with open(lp) as fh:
            log["years"][y] = json.load(fh)
    write_csv(os.path.join(a.out_dir, "balanced_pool.csv"), pool, OUT_COLUMNS)
    write_csv(os.path.join(a.out_dir, "natural_sample.csv"), nat, OUT_COLUMNS)
    with open(os.path.join(a.out_dir, "sampling_log.json"), "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"years merged: {sorted(log['years'])} | balanced_pool.csv: {len(pool):,} rows | natural_sample.csv: {len(nat):,} rows")


if __name__ == "__main__":
    main()
