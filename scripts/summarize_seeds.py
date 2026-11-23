#!/usr/bin/env python3
"""
Summarizes results/seeds.csv into results/seeds_summary.md.

  * Seed results: for every configuration (method + settings) and test set,
    the mean and sample standard deviation (ddof=1) over the seeds that ran.
  * Feature sweep: validation macro-F1 for every vocabulary size and bigram
    setting (phase "sweep"), one column per method.

Usage:
    python scripts/summarize_seeds.py
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

METRICS = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("top3", "Top-3 acc."),
           ("value_err", "Value on wrong rows"), ("ch01_predicted", "Predicted as ch. 01")]
CONFIG = ["method", "max_features", "bigrams", "prior", "exclude_chapters", "train_frac"]


def mean_std(values):
    """Mean and sample standard deviation (ddof=1). std is None with fewer than 2 values."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None)


def fmt(metric, mean, std):
    if metric == "value_err":  # share of pesos -> percent
        mean, std = mean * 100, (None if std is None else std * 100)
        return f"{mean:.1f}%" + ("" if std is None else f" ± {std:.1f}")
    if metric == "ch01_predicted":
        return f"{mean:.0f}" + ("" if std is None else f" ± {std:.0f}")
    return f"{mean:.4f}" + ("" if std is None else f" ± {std:.4f}")


def summarize(rows):
    """{(config..., test_set): {"n": seeds, "seeds": [...], metric: (mean, std)}} for seed rows."""
    groups = defaultdict(list)
    for r in rows:
        if r["phase"] == "seeds":
            groups[tuple(r[c] for c in CONFIG) + (r["test_set"],)].append(r)
    out = {}
    for key, rs in groups.items():
        s = {"n": len(rs), "seeds": sorted(int(r["seed"]) for r in rs)}
        for m, _ in METRICS:
            s[m] = mean_std([float(r[m]) for r in rs])
        out[key] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/seeds.csv")
    ap.add_argument("--out", default="results/seeds_summary.md")
    a = ap.parse_args()
    with open(a.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    lines = ["# Results over seeds", "",
             "Mean ± sample standard deviation (ddof=1) over the seeds listed. "
             "The seed changes which training rows survive the per-chapter cap; "
             "the test sets are the same for every seed.", ""]
    summary = summarize(rows)
    for test_set, title in (("test_bal", "Balanced test (2023-2025, 200 rows per chapter)"),
                            ("test_nat", "Natural test (2023-2025, real chapter mix)")):
        keys = sorted(k for k in summary if k[-1] == test_set)
        if not keys:
            continue
        lines += [f"## {title}", "",
                  "| Method | Vocab | Bigrams | Prior | Excluded | Train frac | Seeds | "
                  + " | ".join(t for _, t in METRICS) + " |",
                  "|---|---|---|---|---|---|---|" + "---|" * len(METRICS)]
        for k in keys:
            s = summary[k]
            cells = [fmt(m, *s[m]) for m, _ in METRICS]
            lines.append(f"| {k[0]} | {k[1]} | {'yes' if k[2] == '1' else 'no'} | {k[3]} | {k[4] or '-'} | "
                         f"{k[5]} | {s['n']} | " + " | ".join(cells) + " |")
        lines.append("")

    sweep = [r for r in rows if r["phase"] == "sweep"]
    if sweep:
        methods = sorted({r["method"] for r in sweep})
        cell = {(r["max_features"], r["bigrams"], r["method"]): r for r in sweep}
        settings = sorted({(r["max_features"], r["bigrams"]) for r in sweep}, key=lambda t: (int(t[0]), t[1]))
        lines += ["## Feature sweep (validation macro-F1, seed 170)", "",
                  "| Vocab | Bigrams | " + " | ".join(methods) + " |",
                  "|---|---|" + "---|" * len(methods)]
        for v, b in settings:
            vals = []
            for m in methods:
                r = cell.get((v, b, m))
                vals.append(f"{float(r['macro_f1']):.4f} (best {r['best_param']})" if r else "")
            lines.append(f"| {v} | {'yes' if b == '1' else 'no'} | " + " | ".join(vals) + " |")
        lines.append("")

    with open(a.out, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()