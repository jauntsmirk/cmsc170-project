#!/usr/bin/env python3
"""
Step 2. Turns data/pool/*.csv into the final splits in data/processed/.

    train      balanced pool rows of the train years, <= --cap-train rows per chapter
    val        balanced pool rows of the validation year(s), <= --cap-eval per chapter
    test_bal   balanced pool rows of the test years, <= --cap-eval per chapter
    test_nat   natural (uniform) sample rows of the test years: real-world distribution
    val_nat    natural sample rows of the validation year(s)

plus balance_report.csv (population vs. split counts per chapter) and build_log.json.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsclf.sampling import OUT_COLUMNS
from hsclf.split import balance_report, cap_per_chapter, parse_years
from hsclf.tabular import read_csv, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="data/pool")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--train-years", default="2012-2021")
    ap.add_argument("--val-years", default="2022")
    ap.add_argument("--test-years", default="2023-2025")
    ap.add_argument("--cap-train", type=int, default=1500)
    ap.add_argument("--cap-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=170)
    a = ap.parse_args()

    ty, vy, sy = parse_years(a.train_years), parse_years(a.val_years), parse_years(a.test_years)
    assert not (ty & vy or ty & sy or vy & sy), "year ranges must not overlap"
    pool = read_csv(os.path.join(a.pool_dir, "balanced_pool.csv"))
    nat = read_csv(os.path.join(a.pool_dir, "natural_sample.csv"))
    with open(os.path.join(a.pool_dir, "sampling_log.json")) as fh:
        slog = json.load(fh)
    rng = random.Random(a.seed)

    def in_years(rows, ys):
        return [r for r in rows if int(r["year"]) in ys]

    sets = {
        "train": cap_per_chapter(in_years(pool, ty), a.cap_train, rng),
        "val": cap_per_chapter(in_years(pool, vy), a.cap_eval, rng),
        "test_bal": cap_per_chapter(in_years(pool, sy), a.cap_eval, rng),
        "val_nat": in_years(nat, vy),
        "test_nat": in_years(nat, sy),
    }
    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in sets.items():
        write_csv(os.path.join(a.out_dir, f"{name}.csv"), rows, OUT_COLUMNS)

    population = Counter()
    for y, st in slog["years"].items():
        if int(y) in ty:
            population.update(st["chapter_counts"])
    table = balance_report({k: sets[k] for k in ("train", "val", "test_bal", "test_nat")}, population)
    write_csv(os.path.join(a.out_dir, "balance_report.csv"), table, list(table[0]))

    def spread(rows):
        c = Counter(r["hs_chapter"] for r in rows)
        return {"rows": len(rows), "chapters": len(c), "min": min(c.values()), "max": max(c.values())}

    log = {"args": vars(a), **{k: spread(v) for k, v in sets.items()}}
    with open(os.path.join(a.out_dir, "build_log.json"), "w") as fh:
        json.dump(log, fh, indent=2)
    for k, v in log.items():
        if k != "args":
            print(f"{k:9s} rows={v['rows']:>7,} chapters={v['chapters']:>3} per-chapter min={v['min']} max={v['max']}")


if __name__ == "__main__":
    main()
