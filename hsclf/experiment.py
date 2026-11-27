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

prior="real" adds one step between 2 and 3: the strength of the real chapter
prior (see adjust_for_prior in models.py) is chosen by accuracy on the 2022
natural sample (natural_sample.csv), so the test sets are still never used to
choose anything. numbers=True keeps "220v"-style tokens (see text.tokenize).

With seed 170 and default settings, train and val are row-for-row the same as
data/processed/train.csv and val.csv (the Milestone 2 splits).
"""
import json
import os
import random
import time

import numpy as np

from hsclf.metrics import evaluate, macro_f1
from hsclf.models import KNN, MultinomialNB, Perceptron, adjust_for_prior, prior_log_ratio
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
# prior="real": strengths tried on the 2022 natural sample. 0 = balanced (no correction),
# 1 = full Bayes correction. Larger values are there for the Perceptron, whose scores are
# not log-probabilities, so its right strength is not known in advance.
PRIOR_GRID = [0, 0.25, 0.5, 1, 2, 4]
PRIORS = ("balanced", "real")

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


def load_prior_data(pool_dir="data/pool"):
    """
    Extra files for prior="real" (read once, only when needed):
      nat  natural_sample.csv: the 2022 rows are used to choose the prior strength
      log  sampling_log.json: the true row count of every chapter in every year
    """
    if "nat" not in _DATA:
        _DATA["nat"] = read_csv(os.path.join(pool_dir, "natural_sample.csv"))
    if "log" not in _DATA:
        with open(os.path.join(pool_dir, "sampling_log.json")) as fh:
            _DATA["log"] = json.load(fh)
    return _DATA["nat"], _DATA["log"]


def real_chapter_counts(log_data, years, chapters):
    """Population count of each chapter (in `chapters` order) summed over `years`,
    from the sampling log. These are counts in the FULL files, not in our sample."""
    counts = np.zeros(len(chapters))
    pos = {c: i for i, c in enumerate(chapters)}
    for y in years:
        year_log = log_data["years"].get(str(y)) or log_data["years"].get(y)
        if year_log is None:
            raise KeyError(f"sampling_log.json has no entry for year {y}")
        for ch, n in year_log["chapter_counts"].items():
            if ch in pos:
                counts[pos[ch]] += n
    return counts


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
    """Builds an untrained model for one grid value. The prior is applied to the
    scores afterwards (adjust_for_prior), so every model is trained the same way."""
    if prior not in PRIORS:
        raise ValueError("unknown prior: %r (use one of %s)" % (prior, PRIORS))
    if method == "naive_bayes":
        return MultinomialNB(C, alpha=param)
    if method == "perceptron":
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


def _tune_prior(method, model, Xn, yn, C, log_ratio, seed):
    """
    Chooses the prior strength by ACCURACY on the 2022 natural sample (real chapter mix).
    Accuracy, not macro-F1, because the prior is meant for traffic with the real mix,
    and macro-F1 deliberately ignores how common each chapter is.
    `model` is already trained on the training rows. Returns (best_strength, all accuracies).
    """
    if method == "knn" and KNN_VAL_LIMIT and KNN_VAL_LIMIT < len(yn):
        idx = np.sort(np.random.RandomState(seed).choice(len(yn), KNN_VAL_LIMIT, replace=False))
        Xn, yn = Xn.take_rows(idx), yn[idx]
    base = model.scores(Xn)  # scored once; every strength reuses it
    accs = {}
    for st in PRIOR_GRID:
        pred = adjust_for_prior(base, log_ratio, st, votes=(method == "knn")).argmax(1)
        accs[st] = float(np.mean(pred == yn))
        log(f"  [{method}] prior strength={st} natural-2022 accuracy={accs[st]:.4f}")
    best = max(accs, key=accs.get)  # first one wins a tie, so ties keep the weaker prior
    return best, accs


def run_one(method, seed, max_features=5000, bigrams=False, prior="balanced",
            exclude_chapters=(), train_frac=1.0, phase="seeds", numbers=False):
    """
    Runs one method once and returns a list of result rows (dicts).

    phase="sweep": tune on validation only and return ONE row (test_set="val").
                   Used to choose vocabulary size and bigrams without touching test.
                   The prior is not used here (validation is balanced).
    phase="seeds": full protocol, returns one row per test set (test_bal, test_nat).
    prior="real":  real chapter priors, strength chosen on the 2022 natural sample.
    numbers=True:  keep "220v"-style tokens (see text.tokenize).
    """
    if prior not in PRIORS:
        raise ValueError("unknown prior: %r (use one of %s)" % (prior, PRIORS))
    t0 = time.time()
    log(f"[{method}] seed={seed} vocab={max_features} bigrams={bigrams} numbers={numbers} "
        f"prior={prior} phase={phase}: start")
    train, val, tests = build_splits(seed, exclude_chapters, train_frac)
    log(f"[{method}] train={len(train):,} val={len(val):,} rows, tuning on validation...")
    chapters = sorted({r["hs_chapter"] for r in train + val}
                      | {r["hs_chapter"] for rows in tests.values() for r in rows})
    cid = {c: i for i, c in enumerate(chapters)}
    C = len(chapters)
    y = lambda rows: np.array([cid[r["hs_chapter"]] for r in rows])
    mode = FEATURES[method]

    base = {"phase": phase, "method": method, "seed": seed, "max_features": max_features,
            "bigrams": int(bigrams), "numbers": int(numbers), "prior": prior,
            "exclude_chapters": " ".join(sorted(exclude_chapters)), "train_frac": train_frac,
            "prior_strength": ""}

    # 1-2. vocabulary on train only, tune on validation
    vec = Vectorizer(mode=mode, max_features=max_features, bigrams=bigrams, numbers=numbers)
    Xtr = vec.fit_transform(_texts(train))
    Xv = vec.transform(_texts(val))
    best, best_f1, _ = _tune(method, Xtr, y(train), Xv, y(val), C, seed, prior)

    if phase == "sweep":
        row = dict(base, best_param=best, test_set="val", accuracy="", macro_f1=round(best_f1, 4),
                   weighted_f1="", top3="", value_err="", ch01_predicted="",
                   seconds=round(time.time() - t0, 1))
        return [row]

    # 2b. prior="real": choose the prior strength on the 2022 natural sample.
    #     Real shares come from the full-file counts of the TRAINING years only.
    strength = 0
    if prior == "real":
        nat, log_data = load_prior_data()
        ex = set(exclude_chapters)
        val_nat = [r for r in nat if int(r["year"]) in VAL_YEARS
                   and r["hs_chapter"] in cid and r["hs_chapter"] not in ex]
        ratio_tr = prior_log_ratio(real_chapter_counts(log_data, TRAIN_YEARS, chapters),
                                   np.bincount(y(train), minlength=C))
        m_tr = _make(method, best, C, seed, prior).fit(Xtr, y(train))
        log(f"[{method}] choosing prior strength on {len(val_nat):,} natural 2022 rows...")
        strength, _ = _tune_prior(method, m_tr, vec.transform(_texts(val_nat)), y(val_nat),
                                  C, ratio_tr, seed)
        base["prior_strength"] = strength

    # 3. refit vocabulary and model on train + val, evaluate both test sets
    log(f"[{method}] best={best} prior strength={strength}, refitting on train+val and testing...")
    trval = train + val
    vec = Vectorizer(mode=mode, max_features=max_features, bigrams=bigrams, numbers=numbers)
    Xtv = vec.fit_transform(_texts(trval))
    model = _make(method, best, C, seed, prior).fit(Xtv, y(trval))
    if prior == "real":  # the refit also sees 2022, so its real shares include 2022 (never test years)
        ratio = prior_log_ratio(real_chapter_counts(log_data, TRAIN_YEARS | VAL_YEARS, chapters),
                                np.bincount(y(trval), minlength=C))
    rows = []
    for name, test_rows in tests.items():
        s = model.scores(vec.transform(_texts(test_rows)))
        if prior == "real":
            s = adjust_for_prior(s, ratio, strength, votes=(method == "knn"))
        pred = s.argmax(1)
        m, _ = evaluate(y(test_rows), pred, C, scores=s, value=_values(test_rows))
        rows.append(dict(base, best_param=best, test_set=name,
                         accuracy=round(m["accuracy"], 4), macro_f1=round(m["macro_f1"], 4),
                         weighted_f1=round(m["weighted_f1"], 4), top3=round(m["top3_accuracy"], 4),
                         value_err=round(m["value_weighted_error"], 4),
                         ch01_predicted=int(np.sum(pred == cid["01"])) if "01" in cid else 0,
                         seconds=round(time.time() - t0, 1)))
    return rows
