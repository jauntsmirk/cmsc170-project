"""
One experiment = one method, one seed, one feature setting.

Used by scripts/run_seeds.py, and later by the priors, learning-curve and
noisy-chapter runs. It follows the same protocol as run_experiments.py:

  1. build train and validation for this seed. The seed decides which rows
     survive the per-chapter cap, so every method (Naive Bayes included) sees
     a different training sample per seed. The test sets stay fixed, so every
     seed is scored on exactly the same rows.
  2. fit the vocabulary on train only, tune hyperparameters by macro-F1 on val
  3. refit vocabulary + model on train + val, then evaluate on both test sets

With seed 170 and default settings, train and val are row-for-row the same as
data/processed/train.csv and val.csv (the Milestone 2 splits).
"""
import os
import random
import time

import numpy as np

from hsclf.metrics import evaluate, macro_f1
from hsclf.models import KNN, MultinomialNB, Perceptron
from hsclf.split import cap_per_chapter, parse_years
from hsclf.tabular import read_csv
from hsclf.text import Vectorizer

# Settings shared with build_dataset.py (same defaults)
TRAIN_YEARS = parse_years("2012-2021")
VAL_YEARS = parse_years("2022")
CAP_TRAIN, CAP_EVAL = 1500, 200

# Hyperparameter grids tuned on validation (same as Milestone 2)
GRIDS = {
    "naive_bayes": [0.01, 0.1, 0.5, 1.0], # alpha (0.5 wins, inside the grid)
    "perceptron": [2, 3, 5, 10], # epochs (Milestone 2 tried only 5 and 10)
    "knn": [1, 5, 15, 30, 50], # k (Milestone 2 stopped at 15, which kept winning)
}
FEATURES = {"naive_bayes": "count", "perceptron": "tfidf", "knn": "tfidf"}
KNN_VAL_LIMIT = 4000  # random validation rows used to tune k-NN (speed)

_DATA = {}  # loaded CSVs, kept between calls so files are read only once

def log(msg):
    """Prints a timestamped progress line right away, so long runs never look frozen."""
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_data(pool_dir="data/pool", processed_dir="data/processed"):
    """Reads the balanced pool and the two fixed test sets (once per process)."""
    if not _DATA:
        _DATA["pool"] = read_csv(os.path.join(pool_dir, "balanced_pool.csv"))
        _DATA["test_bal"] = read_csv(os.path.join(processed_dir, "test_bal.csv"))
        _DATA["test_nat"] = read_csv(os.path.join(processed_dir, "test_nat.csv"))
    return _DATA


def build_splits(seed, exclude_chapters=(), train_frac=1.0):
    """
    Train and validation rows for one seed, built from the balanced pool exactly
    as build_dataset.py does (one Random(seed), train capped first, then val).
    exclude_chapters drops those chapters from every split.
    train_frac < 1 keeps a random share of the training rows (learning curve).
    """
    d = load_data()
    rng = random.Random(seed)
    train = cap_per_chapter([r for r in d["pool"] if int(r["year"]) in TRAIN_YEARS], CAP_TRAIN, rng)
    val = cap_per_chapter([r for r in d["pool"] if int(r["year"]) in VAL_YEARS], CAP_EVAL, rng)
    tests = {"test_bal": d["test_bal"], "test_nat": d["test_nat"]}
    if exclude_chapters:
        ex = set(exclude_chapters)
        keep = lambda rows: [r for r in rows if r["hs_chapter"] not in ex]
        train, val = keep(train), keep(val)
        tests = {k: keep(v) for k, v in tests.items()}
    if train_frac < 1.0:
        k = int(round(len(train) * train_frac))
        idx = sorted(random.Random(seed).sample(range(len(train)), k))
        train = [train[i] for i in idx]
    return train, val, tests


def _texts(rows):
    return [r["goods_description"] for r in rows]


def _values(rows):
    return np.array([float(r["dutiable_value_php"] or 0) for r in rows])


def _make(method, param, C, seed, prior):
    """Builds an untrained model for one grid value."""
    if method == "naive_bayes":
        if prior != "balanced":
            raise NotImplementedError("prior=%r: waiting for B's priors code in models.py" % prior)
        return MultinomialNB(C, alpha=param)
    if method == "perceptron":
        if prior != "balanced":
            raise NotImplementedError("prior=%r: waiting for B's priors code in models.py" % prior)
        return Perceptron(C, epochs=param, averaged=True, seed=seed)
    if method == "knn":
        return KNN(C, k=param)
    raise ValueError("unknown method: " + method)


def _tune(method, Xtr, ytr, Xv, yv, C, seed, prior):
    """Returns (best_param, best_val_macro_f1, all scores) by macro-F1 on validation."""
    scores = {}
    if method == "knn":
        if KNN_VAL_LIMIT and KNN_VAL_LIMIT < len(yv):
            iv = np.sort(np.random.RandomState(seed).choice(len(yv), KNN_VAL_LIMIT, replace=False))
            Xv, yv = Xv.take_rows(iv), yv[iv]
        # one neighbour search with the largest k, then every smaller k is a slice of it
        kmax = max(GRIDS["knn"])
        model = KNN(C, k=kmax).fit(Xtr, ytr)
        sims, labels = model.neighbours(Xv, kmax)
        for k in GRIDS["knn"]:
            pred = model.scores_from_neighbours(sims, labels, k).argmax(1)
            scores[k] = macro_f1(evaluate(yv, pred, C)[1])
            log(f"  [knn] k={k} val macro-F1={scores[k]:.4f}")
    else:
        for p in GRIDS[method]:
            model = _make(method, p, C, seed, prior).fit(Xtr, ytr)
            scores[p] = macro_f1(evaluate(yv, model.predict(Xv), C)[1])
            log(f"  [{method}] param={p} val macro-F1={scores[p]:.4f}")
    best = max(scores, key=scores.get)  # first one wins a tie (grid order)
    return best, scores[best], scores


def run_one(method, seed, max_features=5000, bigrams=False, prior="balanced",
            exclude_chapters=(), train_frac=1.0, phase="seeds"):
    """
    Runs one method once and returns a list of result rows (dicts).

    phase="sweep": tune on validation only and return ONE row (test_set="val").
                   Used to choose vocabulary size and bigrams without touching test.
    phase="seeds": full protocol, returns one row per test set (test_bal, test_nat).
    """
    t0 = time.time()
    log(f"[{method}] seed={seed} vocab={max_features} bigrams={bigrams} phase={phase}: start")
    train, val, tests = build_splits(seed, exclude_chapters, train_frac)
    log(f"[{method}] train={len(train):,} val={len(val):,} rows, tuning on validation...")
    chapters = sorted({r["hs_chapter"] for r in train + val}
                      | {r["hs_chapter"] for rows in tests.values() for r in rows})
    cid = {c: i for i, c in enumerate(chapters)}
    C = len(chapters)
    y = lambda rows: np.array([cid[r["hs_chapter"]] for r in rows])
    mode = FEATURES[method]

    base = {"phase": phase, "method": method, "seed": seed, "max_features": max_features,
            "bigrams": int(bigrams), "prior": prior,
            "exclude_chapters": " ".join(sorted(exclude_chapters)), "train_frac": train_frac}

    # 1-2. vocabulary on train only, tune on validation
    vec = Vectorizer(mode=mode, max_features=max_features, bigrams=bigrams)
    Xtr = vec.fit_transform(_texts(train))
    Xv = vec.transform(_texts(val))
    best, best_f1, _ = _tune(method, Xtr, y(train), Xv, y(val), C, seed, prior)

    if phase == "sweep":
        row = dict(base, best_param=best, test_set="val", accuracy="", macro_f1=round(best_f1, 4),
                   weighted_f1="", top3="", value_err="", ch01_predicted="",
                   seconds=round(time.time() - t0, 1))
        return [row]

    # 3. refit vocabulary and model on train + val, evaluate both test sets
    log(f"[{method}] best={best}, refitting on train+val and testing...")
    trval = train + val
    vec = Vectorizer(mode=mode, max_features=max_features, bigrams=bigrams)
    Xtv = vec.fit_transform(_texts(trval))
    model = _make(method, best, C, seed, prior).fit(Xtv, y(trval))
    rows = []
    for name, test_rows in tests.items():
        s = model.scores(vec.transform(_texts(test_rows)))
        pred = s.argmax(1)
        m, _ = evaluate(y(test_rows), pred, C, scores=s, value=_values(test_rows))
        rows.append(dict(base, best_param=best, test_set=name,
                         accuracy=round(m["accuracy"], 4), macro_f1=round(m["macro_f1"], 4),
                         weighted_f1=round(m["weighted_f1"], 4), top3=round(m["top3_accuracy"], 4),
                         value_err=round(m["value_weighted_error"], 4),
                         ch01_predicted=int(np.sum(pred == cid["01"])) if "01" in cid else 0,
                         seconds=round(time.time() - t0, 1)))
    return rows