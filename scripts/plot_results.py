#!/usr/bin/env python3
"""
Milestone 3 figures and comparison table. Reads results/ only (no raw data needed).
Uses the standard library, numpy and matplotlib.

Writes to results/figures/:
  fig1_methods.png        three methods, balanced vs natural test (mean +/- std, dots = seeds)
  fig2_prior.png          balanced prior vs real chapter prior
  fig3_sweep.png          validation macro-F1 vs vocabulary size (words, bigrams, numbers)
  fig4_per_chapter.png    per-chapter recall on the natural test set (Milestone 2 run)
  fig5_perceptron.png     Perceptron training mistake rate per epoch (Milestone 2 run)
  fig6_tradeoff.png       balanced macro-F1 vs natural macro-F1 per seed, with prior strength
  fig7_learning_curve.png macro-F1 vs share of training rows (only if train_frac runs exist)
  fig8_exclude_ch01.png   all chapters vs chapter 01 excluded (only if those runs exist)
and results/m3_comparison.md (mean +/- sample std table).

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --vocab 40000 --bigrams 1
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

METHODS = ["naive_bayes", "perceptron", "knn"]
NAME = {"naive_bayes": "Naive Bayes", "perceptron": "Perceptron", "knn": "k-NN"}
COLOR = {"naive_bayes": "#1f77b4", "perceptron": "#d62728", "knn": "#2ca02c"}
TEST_NAME = {"test_bal": "Balanced test", "test_nat": "Natural test"}
METRIC_NAME = {"accuracy": "Accuracy", "macro_f1": "Macro-F1"}


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def sel(rows, **kw):
    """Rows whose columns equal the given values (compared as text)."""
    out = []
    for r in rows:
        r = dict(r)
        r["numbers"] = r.get("numbers") or "0"  # rows from before the column existed
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            out.append(r)
    return out


def ms(v):
    """Mean and sample std (ddof=1); std is 0 for a single value."""
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def seed_values(rows, a, method, prior, test_set, metric, exclude="", frac="1.0"):
    rs = sel(rows, phase="seeds", method=method, prior=prior, test_set=test_set, numbers="0",
             exclude_chapters=exclude, train_frac=frac, max_features=a.vocab, bigrams=a.bigrams)
    rs.sort(key=lambda r: int(r["seed"]))
    return [float(r[metric]) for r in rs if r.get(metric) not in (None, "")]


def grouped_bars(ax, groups, series, data, colors, labels, ylabel, title):
    """data[(group, series)] = list of per-seed values."""
    n = len(series)
    w = 0.8 / n
    for j, s in enumerate(series):
        xs, means, errs = [], [], []
        for i, g in enumerate(groups):
            v = data.get((g, s))
            if not v:
                continue
            m, sd = ms(v)
            x = i + (j - (n - 1) / 2) * w
            xs.append(x), means.append(m), errs.append(sd)
            ax.scatter(np.full(len(v), x) + np.linspace(-w / 5, w / 5, len(v)), v,
                       s=9, color="black", alpha=0.55, zorder=3)
        ax.bar(xs, means, w * 0.95, yerr=errs, capsize=3, color=colors[s], label=labels[s], alpha=0.85)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labels.get(g, g) for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def fig_methods(rows, a, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = dict(NAME, **TEST_NAME)
    for ax, (metric, title) in zip(axes, [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
        data = {(ts, m): seed_values(rows, a, m, "balanced", ts, metric)
                for ts in TEST_NAME for m in METHODS}
        grouped_bars(ax, list(TEST_NAME), METHODS, data, COLOR, labels, title, title)
        ax.set_ylim(0.3, 0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Three methods, balanced prior (bars: mean ± std over seeds, dots: each seed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_prior(rows, a, out):
    methods = [m for m in METHODS if seed_values(rows, a, m, "real", "test_nat", "accuracy")]
    if not methods:
        print("no real-prior rows, skipping fig2")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [("test_nat", "accuracy"), ("test_nat", "macro_f1"), ("test_bal", "accuracy"), ("test_bal", "macro_f1")]
    colors = {"balanced": "#7f7f7f", "real": "#ff7f0e"}
    labels = dict(NAME, balanced="Balanced prior", real="Real chapter prior")
    for ax, (ts, metric) in zip(axes.ravel(), panels):
        data = {(m, p): seed_values(rows, a, m, p, ts, metric) for m in methods for p in colors}
        grouped_bars(ax, methods, list(colors), data, colors, labels, METRIC_NAME[metric],
                     f"{TEST_NAME[ts]}: {METRIC_NAME[metric]}")
        ax.set_ylim(0.4, 0.8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Effect of real chapter priors (dots: each seed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_sweep(rows, out):
    sweep = sel(rows, phase="sweep", prior="balanced", exclude_chapters="", train_frac="1.0")
    if not sweep:
        print("no sweep rows, skipping fig3")
        return
    variants = {("0", "0"): ("-", "o", "words"), ("1", "0"): ("--", "s", "+ bigrams"),
                ("0", "1"): (":", "x", "+ numbers"), ("1", "1"): ("-.", "^", "+ bigrams + numbers")}
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHODS:
        for (b, n), (ls, mk, _) in variants.items():
            pts = sorted((int(r["max_features"]), float(r["macro_f1"]))
                         for r in sweep if r["method"] == m and r["bigrams"] == b and r["numbers"] == n)
            if pts:
                ax.plot(*zip(*pts), ls=ls, marker=mk, color=COLOR[m], alpha=0.85)
    ax.set_xscale("log")
    ticks = [2000, 5000, 10000, 20000, 40000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["2k", "5k", "10k", "20k", "40k"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Vocabulary size (max features)")
    ax.set_ylabel("Validation macro-F1 (seed 170)")
    ax.set_title("Feature sweep (validation only)", fontsize=10)
    ax.grid(alpha=0.3)
    h1 = [plt.Line2D([], [], color=COLOR[m], label=NAME[m]) for m in METHODS]
    h2 = [plt.Line2D([], [], color="black", ls=ls, marker=mk, label=lab) for ls, mk, lab in variants.values()]
    l1 = ax.legend(handles=h1, loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=h2, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_per_chapter(path, out, top=10):
    if not os.path.exists(path):
        print(path, "missing, skipping fig4")
        return
    rows = [r for r in load(path) if int(r["test_nat_support"]) > 0]
    sup = np.array([int(r["test_nat_support"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for m in METHODS:
        key = {"naive_bayes": "naive_bayes_recall", "perceptron": "perceptron_recall", "knn": "knn_recall"}[m]
        ax.scatter(sup, [float(r[key]) for r in rows], s=14, alpha=0.6, color=COLOR[m], label=NAME[m])
    for r in sorted(rows, key=lambda r: -int(r["test_nat_support"]))[:6]:
        ax.annotate(r["chapter"], (int(r["test_nat_support"]), float(r["perceptron_recall"])),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Chapter size in natural test set (rows, log)")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs chapter size (Milestone 2 models)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    big = sorted(rows, key=lambda r: -int(r["test_nat_support"]))[:top]
    x = np.arange(len(big))
    for j, m in enumerate(METHODS):
        ax.bar(x + (j - 1) * 0.27, [float(r[f"{m}_recall"]) for r in big], 0.27, color=COLOR[m], label=NAME[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['chapter']}\n(n={r['test_nat_support']})" for r in big], fontsize=7)
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall on the {top} largest chapters", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_perceptron(path, out):
    if not os.path.exists(path):
        print(path, "missing, skipping fig5")
        return
    with open(path) as fh:
        hist = json.load(fh)["results"]["perceptron"].get("perceptron_history", [])
    if not hist:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([h["epoch"] for h in hist], [h["train_mistake_rate"] for h in hist], marker="o", color=COLOR["perceptron"])
    ax.set_xticks([h["epoch"] for h in hist])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training mistake rate")
    ax.set_title("Perceptron mistakes per epoch (Milestone 2 run)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_tradeoff(rows, a, out):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for m in METHODS:
        for prior, mk in (("balanced", "o"), ("real", "^")):
            rs = {ts: {int(r["seed"]): r for r in sel(
                rows, phase="seeds", method=m, prior=prior, test_set=ts, numbers="0", exclude_chapters="",
                train_frac="1.0", max_features=a.vocab, bigrams=a.bigrams)} for ts in TEST_NAME}
            for s in sorted(set(rs["test_bal"]) & set(rs["test_nat"])):
                x, y = float(rs["test_bal"][s]["macro_f1"]), float(rs["test_nat"][s]["macro_f1"])
                ax.scatter(x, y, marker=mk, color=COLOR[m], s=40, alpha=0.85)
                if prior == "real":
                    ax.annotate(rs["test_nat"][s]["prior_strength"], (x, y), fontsize=7,
                                xytext=(4, 3), textcoords="offset points")
    h = [plt.Line2D([], [], color=COLOR[m], marker="s", ls="", label=NAME[m]) for m in METHODS]
    h += [plt.Line2D([], [], color="black", marker="o", ls="", label="balanced prior"),
          plt.Line2D([], [], color="black", marker="^", ls="", label="real prior (label = strength)")]
    ax.legend(handles=h, fontsize=8)
    ax.set_xlabel("Balanced-test macro-F1")
    ax.set_ylabel("Natural-test macro-F1")
    ax.set_title("What the real prior trades away and gains, per seed", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_learning_curve(rows, a, out):
    fracs = sorted({float(r["train_frac"]) for r in rows
                    if r["phase"] == "seeds" and r["prior"] == "balanced" and r["exclude_chapters"] == ""} | {1.0})
    if len(fracs) < 2:
        print("no train_frac runs, skipping fig7")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ts in zip(axes, TEST_NAME):
        for m in METHODS:
            xs, ys, es = [], [], []
            for f in fracs:
                v = seed_values(rows, a, m, "balanced", ts, "macro_f1", frac=str(f))
                if v:
                    mean, sd = ms(v)
                    xs.append(f), ys.append(mean), es.append(sd)
            if len(xs) > 1:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=COLOR[m], label=NAME[m])
        ax.set_xlabel("Share of training rows")
        ax.set_ylabel("Macro-F1")
        ax.set_title(TEST_NAME[ts], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Learning curve (balanced prior, mean ± std over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_exclude(rows, a, out):
    if not sel(rows, phase="seeds", exclude_chapters="01"):
        print("no chapter-01 exclusion runs, skipping fig8")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    groups = ["", "01"]
    labels = dict(NAME, **{"": "All chapters", "01": "Chapter 01 excluded"})
    colors = {"": "#7f7f7f", "01": "#9467bd"}
    for ax, (ts, metric) in zip(axes, [("test_nat", "macro_f1"), ("test_nat", "accuracy")]):
        data = {(m, g): seed_values(rows, a, m, "balanced", ts, metric, exclude=g) for m in METHODS for g in groups}
        grouped_bars(ax, METHODS, groups, data, colors, labels, METRIC_NAME[metric],
                     f"{TEST_NAME[ts]}: {METRIC_NAME[metric]}")
        ax.set_ylim(0.3, 0.7)
    axes[0].legend(fontsize=8)
    fig.suptitle("What chapter 01 costs (the test sets also lose chapter 01 rows)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_table(rows, a, path):
    cols = [("test_bal", "accuracy", "Bal. acc"), ("test_bal", "macro_f1", "Bal. macro-F1"),
            ("test_nat", "accuracy", "Nat. acc"), ("test_nat", "macro_f1", "Nat. macro-F1"),
            ("test_nat", "top3", "Nat. top-3"), ("test_nat", "value_err", "Nat. value on wrong rows"),
            ("test_nat", "ch01_predicted", "Nat. predicted as ch. 01")]
    lines = ["| Method | Prior | Seeds | " + " | ".join(c[2] for c in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for prior in ("balanced", "real"):
        for m in METHODS:
            vals = [seed_values(rows, a, m, prior, ts, k) for ts, k, _ in cols]
            if not vals[0]:
                continue
            cells = []
            for (ts, k, _), v in zip(cols, vals):
                mean, sd = ms(v)
                if k == "value_err":
                    cells.append(f"{mean * 100:.1f}% ± {sd * 100:.1f}")
                elif k == "ch01_predicted":
                    cells.append(f"{mean:.0f} ± {sd:.0f}")
                else:
                    cells.append(f"{mean:.4f} ± {sd:.4f}")
            lines.append(f"| {NAME[m]} | {prior} | {len(vals[0])} | " + " | ".join(cells) + " |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--vocab", default="40000")
    ap.add_argument("--bigrams", default="1")
    a = ap.parse_args()
    fig_dir = os.path.join(a.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    rows = load(os.path.join(a.results_dir, "seeds.csv"))

    fig_methods(rows, a, os.path.join(fig_dir, "fig1_methods.png"))
    fig_prior(rows, a, os.path.join(fig_dir, "fig2_prior.png"))
    fig_sweep(rows, os.path.join(fig_dir, "fig3_sweep.png"))
    fig_per_chapter(os.path.join(a.results_dir, "per_chapter_test_nat.csv"), os.path.join(fig_dir, "fig4_per_chapter.png"))
    fig_perceptron(os.path.join(a.results_dir, "results.json"), os.path.join(fig_dir, "fig5_perceptron.png"))
    fig_tradeoff(rows, a, os.path.join(fig_dir, "fig6_tradeoff.png"))
    fig_learning_curve(rows, a, os.path.join(fig_dir, "fig7_learning_curve.png"))
    fig_exclude(rows, a, os.path.join(fig_dir, "fig8_exclude_ch01.png"))
    write_table(rows, a, os.path.join(a.results_dir, "m3_comparison.md"))
    print("figures in", fig_dir)


if __name__ == "__main__":
    main()
