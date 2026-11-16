#!/usr/bin/env python3
"""
Step 3. Fits the vectorizer on TRAIN only, tunes each model on VAL, then evaluates
on both test sets (balanced and natural). Writes results/ (json, markdown, csv, png).

Protocol (same for every method):
  1. fit vocabulary/idf on train
  2. pick hyperparameters by macro-F1 on val
  3. refit on train + val (vocabulary refit too) and report on test_bal and test_nat
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsclf.metrics import evaluate, macro_f1, per_class_prf
from hsclf.models import KNN, MultinomialNB, Perceptron
from hsclf.tabular import read_csv, write_csv
from hsclf.text import Vectorizer


def load(path):
    rows = read_csv(path)
    return rows, [r["goods_description"] for r in rows], [r["hs_chapter"] for r in rows], \
        np.array([float(r["dutiable_value_php"] or 0) for r in rows])


def random_subset(n, k, seed):
    """k distinct row indices out of n, seeded and sorted. (Never take the first k rows:
    the CSV files are sorted by chapter and by year, so the head is a biased subset.)"""
    return np.sort(np.random.RandomState(seed).choice(n, size=k, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--bigrams", action="store_true")
    ap.add_argument("--methods", default="nb,perceptron,knn")
    ap.add_argument("--knn-test-limit", type=int, default=0, help="evaluate k-NN on first N test rows only (0=all)")
    ap.add_argument("--knn-val-limit", type=int, default=4000, help="tune k-NN on the first N validation rows only (0=all)")
    ap.add_argument("--seed", type=int, default=170)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    D = {n: load(os.path.join(a.data_dir, f"{n}.csv")) for n in ["train", "val", "test_bal", "test_nat"]}
    chapters = sorted(set(D["train"][2]) | set(D["val"][2]) | set(D["test_bal"][2]) | set(D["test_nat"][2]))
    cid = {c: i for i, c in enumerate(chapters)}
    C = len(chapters)
    Y = {n: np.array([cid[c] for c in D[n][2]]) for n in D}
    print(f"{C} chapters | " + " ".join(f"{n}={len(Y[n]):,}" for n in D))

    def featurize(train_texts, mode, eval_sets):
        vec = Vectorizer(mode=mode, max_features=a.max_features, bigrams=a.bigrams)
        Xtr = vec.fit_transform(train_texts)
        return Xtr, {n: vec.transform(D[n][1]) for n in eval_sets}, vec

    results, per_class = {}, {}
    trval_texts = D["train"][1] + D["val"][1]
    Ytrval = np.concatenate([Y["train"], Y["val"]])

    def run(name, mode, make_model, grid, fit_kwargs=None):
        t0 = time.time()
        Xtr, Xe, _ = featurize(D["train"][1], mode, ["val"])
        tuning = []
        for params in grid:
            m = make_model(**params).fit(Xtr, Y["train"])
            Xv, yv = Xe["val"], Y["val"]
            if name == "knn" and a.knn_val_limit and a.knn_val_limit < len(yv):
                iv = random_subset(len(yv), a.knn_val_limit, a.seed)  # random: the files are sorted by chapter/year
                Xv, yv = Xv.take_rows(iv), yv[iv]
            tuning.append({"params": params, "val_macro_f1": macro_f1(evaluate(yv, m.predict(Xv), C)[1])})
            print(f"  [{name}] {params} val macro-F1 = {tuning[-1]['val_macro_f1']:.4f}")
        best = max(tuning, key=lambda t: t["val_macro_f1"])["params"]
        Xtv, Xt, _ = featurize(trval_texts, mode, ["test_bal", "test_nat"])
        model = make_model(**best).fit(Xtv, Ytrval)
        res = {"features": mode, "best_params": best, "tuning": tuning, "fit_seconds": round(time.time() - t0, 1)}
        for tn in ("test_bal", "test_nat"):
            X, y, vals = Xt[tn], Y[tn], D[tn][3]
            if name == "knn" and a.knn_test_limit and a.knn_test_limit < len(y):
                it = random_subset(len(y), a.knn_test_limit, a.seed)
                X, y, vals = X.take_rows(it), y[it], vals[it]
            s = model.scores(X)
            m, cm = evaluate(y, s.argmax(1), C, scores=s, value=vals)
            res[tn] = m
            if tn == "test_nat":
                np.save(os.path.join(a.out_dir, f"confusion_{name}_test_nat.npy"), cm)
                _, rec, f1, sup = per_class_prf(cm)
                per_class[name] = (rec, f1, sup)
        if hasattr(model, "history"):
            res["perceptron_history"] = model.history
        results[name] = res
        print(f"  [{name}] best={best} test_bal acc={res['test_bal']['accuracy']:.4f} F1={res['test_bal']['macro_f1']:.4f} | "
              f"test_nat acc={res['test_nat']['accuracy']:.4f} F1={res['test_nat']['macro_f1']:.4f}")

    methods = a.methods.split(",")
    if "nb" in methods:
        run("naive_bayes", "count", lambda alpha: MultinomialNB(C, alpha), [{"alpha": x} for x in (0.01, 0.1, 0.5, 1.0)])
    if "perceptron" in methods:
        run("perceptron", "tfidf", lambda epochs: Perceptron(C, epochs, True, a.seed), [{"epochs": e} for e in (5, 10)])
    if "knn" in methods:
        run("knn", "tfidf", lambda k: KNN(C, k), [{"k": k} for k in (1, 5, 15)])
    # fix key so per_class uses the same names
    with open(os.path.join(a.out_dir, "results.json"), "w") as fh:
        json.dump({"chapters": chapters, "results": results}, fh, indent=2, default=float)

    # markdown summary + per-chapter table
    lines = ["| Method | Features | Best params | Bal. acc | Bal. macro-F1 | Nat. acc | Nat. macro-F1 | Nat. top-3 | Nat. value-wtd error |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        b, t = r["test_bal"], r["test_nat"]
        lines.append(f"| {n} | {r['features']} | {r['best_params']} | {b['accuracy']:.3f} | {b['macro_f1']:.3f} | "
                     f"{t['accuracy']:.3f} | {t['macro_f1']:.3f} | {t['top3_accuracy']:.3f} | {t['value_weighted_error']:.3f} |")
    with open(os.path.join(a.out_dir, "results_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    rows = []
    for i, ch in enumerate(chapters):
        row = {"chapter": ch}
        for n, (rec, f1, sup) in per_class.items():
            row[f"{n}_recall"], row[f"{n}_f1"], row["test_nat_support"] = round(rec[i], 4), round(f1[i], 4), int(sup[i])
        rows.append(row)
    write_csv(os.path.join(a.out_dir, "per_chapter_test_nat.csv"), rows, list(rows[0]))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
