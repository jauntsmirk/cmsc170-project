#!/usr/bin/env python3
"""
Milestone 3 runner. Runs every method through hsclf.experiment.run_one and appends
one row per result to results/seeds.csv as soon as each run finishes.

Two phases:
  sweep  choose the features. Every vocabulary size x bigram setting, seed 170,
         scored on VALIDATION only (the test sets are never touched here).
  seeds  the comparison. Each method with its best sweep setting, over several
         seeds, scored on both test sets. Seed 170 reproduces Milestone 2.

Safe to stop and rerun: a run whose rows are already in the CSV is skipped.

Usage:
    python3 scripts/run_seeds.py --phase sweep
    python3 scripts/run_seeds.py --phase seeds
    python3 scripts/run_seeds.py --phase seeds --methods nb,perceptron --seeds 170-172
    python3 scripts/run_seeds.py --phase seeds --prior real          # real chapter priors
    python3 scripts/run_seeds.py --phase sweep --numbers 1           # keep "220v"-style tokens

An older results CSV without the "numbers" and "prior_strength" columns is upgraded
in place the first time new rows are added (old rows get numbers=0).
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsclf.experiment import log, run_one
from hsclf.split import parse_years

FIELDS = ["phase", "method", "seed", "max_features", "bigrams", "numbers", "prior", "exclude_chapters",
          "train_frac", "best_param", "prior_strength", "test_set", "accuracy", "macro_f1", "weighted_f1",
          "top3", "value_err", "ch01_predicted", "seconds"]
METHOD_NAMES = {"nb": "naive_bayes", "perceptron": "perceptron", "knn": "knn"}
EXPECTED_TEST_SETS = {"sweep": {"val"}, "seeds": {"test_bal", "test_nat"}}


def run_key(r):
    """Identifies one run: same key = same settings, so it does not need to run again."""
    return (r["phase"], r["method"], str(r["seed"]), str(r["max_features"]), str(int(r["bigrams"])),
            str(int(r.get("numbers") or 0)),  # rows from before the numbers option count as 0
            r["prior"], r["exclude_chapters"], str(float(r["train_frac"])))


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def upgrade_csv(path):
    """Rewrites a CSV made before the "numbers" and "prior_strength" columns existed,
    so new rows can be appended under one header. Old rows get numbers=0 (letters only)."""
    with open(path, newline="") as fh:
        header = next(csv.reader(fh), [])
    if header == FIELDS:
        return
    old = read_rows(path)
    for r in old:
        r.setdefault("numbers", "0")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(old)


def append_rows(path, rows):
    if os.path.exists(path):
        upgrade_csv(path)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def done_keys(rows):
    """Keys of runs that already have every expected test set in the CSV."""
    seen = {}
    for r in rows:
        seen.setdefault(run_key(r), set()).add(r["test_set"])
    return {k for k, sets in seen.items() if sets >= EXPECTED_TEST_SETS[k[0]]}


def best_features(rows, method, numbers=0):
    """Best (max_features, bigrams) for a method by validation macro-F1 in the sweep rows
    made with the same numbers setting."""
    sweep = [r for r in rows if r["phase"] == "sweep" and r["method"] == method
             and int(r.get("numbers") or 0) == numbers
             and r["prior"] == "balanced" and r["exclude_chapters"] == "" and float(r["train_frac"]) == 1.0]
    if not sweep:
        return None
    b = max(sweep, key=lambda r: float(r["macro_f1"]))
    return int(b["max_features"]), int(b["bigrams"]), float(b["macro_f1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sweep", "seeds"], required=True)
    ap.add_argument("--methods", default="nb,perceptron,knn")
    ap.add_argument("--seeds", default="170-174", help="seeds phase only, e.g. 170-174 or 170,171")
    ap.add_argument("--vocab", default="2000,5000,10000", help="sweep phase: vocabulary sizes to try")
    ap.add_argument("--bigrams", default="0,1", help="sweep phase: 0 = words only, 1 = words + bigrams")
    ap.add_argument("--prior", choices=["balanced", "real"], default="balanced",
                    help="seeds phase: real = real chapter priors, strength chosen on the 2022 natural sample")
    ap.add_argument("--numbers", type=int, choices=[0, 1], default=0,
                    help="1 = keep number+unit tokens such as 220v (text.tokenize)")
    ap.add_argument("--out", default="results/seeds.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    methods = [METHOD_NAMES[m] for m in a.methods.split(",")]
    rows = read_rows(a.out)
    done = done_keys(rows)

    # the list of runs to do: (method, seed, max_features, bigrams)
    plan = []
    if a.phase == "sweep":
        for m in methods:
            for v in [int(x) for x in a.vocab.split(",")]:
                for b in [int(x) for x in a.bigrams.split(",")]:
                    plan.append((m, 170, v, b))
    else:
        settings = {}
        for m in methods:
            best = best_features(rows, m, a.numbers)
            if best is None and a.numbers:
                best = best_features(rows, m, 0)
                if best is not None:
                    log(f"{m}: no sweep rows with numbers=1, using the best numbers=0 setting")
            if best is None:
                settings[m] = (5000, 0)
                log(f"{m}: no sweep rows found, using the Milestone 2 setting vocab=5000 bigrams=0")
            else:
                settings[m] = best[:2]
                log(f"{m}: best sweep setting vocab={best[0]} bigrams={best[1]} (val macro-F1 {best[2]:.4f})")
        for s in sorted(parse_years(a.seeds)):  # parse_years also reads "170-174"
            for m in methods:
                plan.append((m, s, *settings[m]))

    log(f"{len(plan)} runs planned, writing to {a.out}")
    t_all = time.time()
    for i, (m, s, v, b) in enumerate(plan, 1):
        prior = a.prior if a.phase == "seeds" else "balanced"  # the sweep never uses the prior
        key = run_key({"phase": a.phase, "method": m, "seed": s, "max_features": v, "bigrams": b,
                       "numbers": a.numbers, "prior": prior, "exclude_chapters": "", "train_frac": 1.0})
        if key in done:
            log(f"run {i}/{len(plan)}: {m} seed={s} vocab={v} bigrams={b} already done, skipping")
            continue
        log(f"===== run {i}/{len(plan)}: {m} seed={s} vocab={v} bigrams={b} =====")
        out = run_one(m, s, max_features=v, bigrams=bool(b), phase=a.phase,
                      prior=prior, numbers=bool(a.numbers))
        append_rows(a.out, out)  # saved immediately, so a crash loses at most this run
        for r in out:
            log(f"  saved: {r['test_set']} acc={r['accuracy']} macro-F1={r['macro_f1']} ({r['seconds']}s)")
    log(f"finished in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()