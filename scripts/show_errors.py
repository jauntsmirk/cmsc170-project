#!/usr/bin/env python3
"""
Prints example misclassified descriptions from the natural test set, using the same
protocol as run_experiments.py (tf-idf, vocabulary fitted on train+val, averaged Perceptron,
5 epochs, seed 170). Also writes results/example_errors.csv.

Usage:
    python scripts/show_errors.py                      # 25 random errors
    python scripts/show_errors.py --true 85 --pred 90  # only errors of one kind
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsclf.models import Perceptron
from hsclf.tabular import read_csv, write_csv
from hsclf.text import Vectorizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--true", default=None, help="only errors whose true chapter is this (e.g. 85)")
    ap.add_argument("--pred", default=None, help="only errors predicted as this chapter (e.g. 90)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=170)
    a = ap.parse_args()

    print("Loading data...", flush=True)
    train = read_csv(os.path.join(a.data_dir, "train.csv")) + read_csv(os.path.join(a.data_dir, "val.csv"))
    test = read_csv(os.path.join(a.data_dir, "test_nat.csv"))
    chapters = sorted({r["hs_chapter"] for r in train + test})
    cid = {c: i for i, c in enumerate(chapters)}
    y = np.array([cid[r["hs_chapter"]] for r in train])

    print("Building features...", flush=True)
    vec = Vectorizer(mode="tfidf", max_features=5000)
    Xtr = vec.fit_transform(r["goods_description"] for r in train)
    Xte = vec.transform(r["goods_description"] for r in test)
    print("Training the Perceptron (about 1 to 3 minutes, please wait, no progress bar)...", flush=True)
    model = Perceptron(len(chapters), a.epochs, True, a.seed).fit(Xtr, y)
    print("Predicting...", flush=True)
    pred = model.predict(Xte)

    errors = []
    for r, p in zip(test, pred):
        t, pc = r["hs_chapter"], chapters[p]
        if t != pc and (a.true is None or t == a.true) and (a.pred is None or pc == a.pred):
            errors.append({"year": r["year"], "true_chapter": t, "predicted_chapter": pc,
                           "true_hs_code": r["hs_code"], "description": r["goods_description"]})
    print(f"{len(errors):,} matching errors out of {len(test):,} natural test rows")
    shown = random.Random(a.seed).sample(errors, min(a.n, len(errors)))
    os.makedirs(a.out_dir, exist_ok=True)
    write_csv(os.path.join(a.out_dir, "example_errors.csv"), shown, list(shown[0]) if shown else ["description"])
    for e in shown:
        print(f"true {e['true_chapter']} -> predicted {e['predicted_chapter']} | {e['description'][:90]}")


if __name__ == "__main__":
    main()
