"""
Tests for the real chapter priors (models.prior_log_ratio, models.adjust_for_prior,
experiment.run_one(prior="real")) and the numbers option of the tokenizer.
Tiny fake data only, so no download is needed.
"""
import csv
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hsclf import experiment
from hsclf.models import MultinomialNB, adjust_for_prior, prior_log_ratio
from hsclf.text import Vectorizer, tokenize


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestNumbersTokens(unittest.TestCase):
    def test_default_unchanged(self):
        # Milestone 2 behaviour: letters only
        self.assertEqual(tokenize("220V MOTOR 1.5HP"), ["motor", "hp"])

    def test_units_kept_with_digits_replaced(self):
        self.assertEqual(tokenize("220V MOTOR 1.5HP", numbers=True), ["#v", "motor", "hp", "#hp"])
        self.assertEqual(tokenize("CABLE 6M", numbers=True), ["cable", "#m"])

    def test_same_unit_same_token(self):
        self.assertEqual(tokenize("110V FAN", numbers=True), tokenize("220V FAN", numbers=True))

    def test_lone_numbers_and_refs_still_dropped(self):
        self.assertEqual(tokenize("BOLT 2000 (QTY 2,000 PCS) [TR-H1692]", numbers=True), ["bolt"])

    def test_numbers_mode_only_adds_words(self):
        for s in ["USB2 HUB", "M6X20 BOLT", "220V MOTOR 1.5HP", "PALM OLEIN IN BULK", "X200 LAPTOP 15.6IN"]:
            self.assertTrue(set(tokenize(s)) <= set(tokenize(s, numbers=True)), s)

    def test_vectorizer_default_matches_old(self):
        texts = ["220V MOTOR", "220V MOTOR PUMP", "110V MOTOR", "CABLE 6M", "CABLE 6M", "CABLE 6M USB2"]
        old = Vectorizer(mode="count", min_df=1).fit(texts).vocab
        self.assertNotIn("#v", old)
        new = Vectorizer(mode="count", min_df=1, numbers=True).fit(texts).vocab
        self.assertIn("#v", new)
        self.assertTrue(set(old) <= set(new))


class TestPriorMath(unittest.TestCase):
    def test_equal_shares_give_zero(self):
        np.testing.assert_allclose(prior_log_ratio([10, 10, 10], [5, 5, 5]), 0, atol=1e-12)

    def test_zero_count_is_finite(self):
        self.assertTrue(np.all(np.isfinite(prior_log_ratio([0, 100], [50, 50]))))

    def test_strength_zero_is_identity(self):
        s = np.random.RandomState(0).rand(4, 3)
        lr = np.array([0.5, -1.0, 2.0])
        self.assertIs(adjust_for_prior(s, lr, 0), s)
        self.assertIs(adjust_for_prior(s, lr, 0, votes=True), s)

    def test_votes_are_multiplied(self):
        votes = np.array([[1.0, 1.0]])
        lr = np.log(np.array([3.0, 1.0]))
        np.testing.assert_allclose(adjust_for_prior(votes, lr, 1, votes=True), [[3.0, 1.0]])

    def test_naive_bayes_strength_one_is_real_prior(self):
        # With alpha=1 the model's own prior uses the same add-one smoothing as prior_log_ratio,
        # so strength 1 must equal Naive Bayes trained with the real prior from the start.
        texts = ["red apple", "green apple", "red car", "fast car", "red fast car", "green pear"]
        y = np.array([0, 0, 1, 1, 1, 0])
        X = Vectorizer(mode="count", min_df=1).fit_transform(texts)
        nb = MultinomialNB(2, alpha=1.0).fit(X, y)
        real = np.array([900, 100])
        adjusted = adjust_for_prior(nb.scores(X), prior_log_ratio(real, np.bincount(y)), 1)
        nb.log_prior = np.log((real + 1.0) / (real + 1.0).sum())
        np.testing.assert_allclose(adjusted, nb.scores(X), atol=1e-12)

    def test_prior_fixes_ambiguous_rows(self):
        # "motor" appears equally in both chapters in training (balanced), but in real traffic
        # chapter 0 is nine times more common. Without the prior the model cannot tell them
        # apart; with it, ambiguous rows go to the common chapter, which is right 90% of the time.
        texts = ["pump motor"] * 20 + ["motor"] * 20 + ["cable motor"] * 20 + ["motor"] * 20
        y = np.array([0] * 40 + [1] * 40)
        vec = Vectorizer(mode="count", min_df=1)
        nb = MultinomialNB(2, alpha=1.0).fit(vec.fit_transform(texts), y)
        test_y = np.array([0] * 90 + [1] * 10)
        S = nb.scores(vec.transform(["motor"] * 100))
        acc = lambda s: np.mean(s.argmax(1) == test_y)
        lr = prior_log_ratio([900, 100], np.bincount(y))
        self.assertGreaterEqual(acc(adjust_for_prior(S, lr, 1)), 0.9)
        self.assertLessEqual(acc(S), 0.9)


def fake_rows(year, chapter, n, desc):
    return [{"year": str(year), "hs_chapter": chapter, "source_row": f"{year}-{chapter}-{i}-{desc}",
             "goods_description": desc, "dutiable_value_php": "1"} for i in range(n)]


class TestRunOneReal(unittest.TestCase):
    def setUp(self):
        pool, nat = [], []
        # In the balanced training data a bare "motor" is more often chapter 85 (40 vs 20 rows),
        # so a balanced model sends "motor" to 85. In real traffic chapter 84 is 9x more common,
        # and the natural sets below are mostly 84, so the prior should move "motor" to 84.
        for y in (2015, 2016, 2022, 2024):
            pool += fake_rows(y, "84", 40, "pump motor 220V") + fake_rows(y, "84", 20, "motor")
            pool += fake_rows(y, "85", 20, "cable motor 6M") + fake_rows(y, "85", 40, "motor")
            pool += fake_rows(y, "01", 30, "live goat")
        nat += fake_rows(2022, "84", 80, "motor") + fake_rows(2022, "85", 10, "motor") + fake_rows(2022, "01", 10, "live goat")
        counts = {"84": 9000, "85": 1000, "01": 1000}
        experiment._DATA.clear()
        experiment._DATA.update({
            "pool": [r for r in pool if int(r["year"]) != 2024],
            "test_bal": [r for r in pool if int(r["year"]) == 2024],
            "test_nat": fake_rows(2024, "84", 90, "motor") + fake_rows(2024, "85", 10, "motor"),
            "nat": nat,
            "log": {"years": {str(y): {"chapter_counts": counts} for y in range(2012, 2023)}},
        })

    def tearDown(self):
        experiment._DATA.clear()

    def test_real_prior_runs_and_helps_natural(self):
        # k-NN: the prior can only reweight chapters that are already among the k neighbours
        # (0 votes * any weight = 0). On this toy the 120 "motor" rows are identical, so they all
        # tie and a small k may hold only chapter 85. k=200 takes every "motor" row, so the votes
        # split 1 : 2 (84 : 85) as the data says, and the test checks only the prior.
        old_grid = experiment.GRIDS["knn"]
        experiment.GRIDS["knn"] = [200]
        self.addCleanup(experiment.GRIDS.__setitem__, "knn", old_grid)
        for method in ("naive_bayes", "perceptron", "knn"):
            bal = {r["test_set"]: r for r in experiment.run_one(method, 170, max_features=50, prior="balanced")}
            real = {r["test_set"]: r for r in experiment.run_one(method, 170, max_features=50, prior="real")}
            self.assertEqual(set(real), {"test_bal", "test_nat"})
            self.assertIn(real["test_nat"]["prior_strength"], experiment.PRIOR_GRID)
            self.assertEqual(bal["test_nat"]["prior_strength"], "")
            # The Perceptron's scores are not probabilities, and on this toy it already sends "motor"
            # to 84, so its tuner may rightly pick strength 0. Naive Bayes and k-NN must improve.
            if method == "perceptron":
                self.assertGreaterEqual(real["test_nat"]["accuracy"], bal["test_nat"]["accuracy"])
            else:
                self.assertGreater(real["test_nat"]["prior_strength"], 0, method)
                self.assertGreater(real["test_nat"]["accuracy"], bal["test_nat"]["accuracy"], method)

    def test_numbers_option_recorded(self):
        rows = experiment.run_one("naive_bayes", 170, max_features=50, numbers=True)
        self.assertTrue(all(r["numbers"] == 1 for r in rows))

    def test_unknown_prior_rejected(self):
        with self.assertRaises(ValueError):
            experiment.run_one("naive_bayes", 170, prior="uniform")


class TestCsvUpgrade(unittest.TestCase):
    OLD = ["phase", "method", "seed", "max_features", "bigrams", "prior", "exclude_chapters",
           "train_frac", "best_param", "test_set", "accuracy", "macro_f1", "weighted_f1", "top3",
           "value_err", "ch01_predicted", "seconds"]

    def test_old_csv_upgraded_and_appended(self):
        rs = load_script("run_seeds")
        old_row = {"phase": "seeds", "method": "knn", "seed": "170", "max_features": "5000", "bigrams": "0",
                   "prior": "balanced", "exclude_chapters": "", "train_frac": "1.0", "best_param": "15",
                   "test_set": "test_nat", "accuracy": "0.4", "macro_f1": "0.3", "weighted_f1": "0.3",
                   "top3": "0.6", "value_err": "0.4", "ch01_predicted": "5", "seconds": "9"}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seeds.csv")
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.OLD)
                w.writeheader()
                w.writerow(old_row)
            new_row = dict(old_row, numbers=1, prior="real", prior_strength=1, seed="171")
            rs.append_rows(path, [new_row])
            with open(path, newline="") as fh:
                self.assertEqual(next(csv.reader(fh)), rs.FIELDS)
            rows = rs.read_rows(path)
            self.assertEqual([r["numbers"] for r in rows], ["0", "1"])
            self.assertEqual(rows[0]["accuracy"], "0.4")  # old values kept
            # an old row and a new code-side key with numbers=0 are the same run
            self.assertEqual(rs.run_key(rows[0]), rs.run_key(dict(old_row, numbers=0)))
            self.assertNotEqual(rs.run_key(rows[0]), rs.run_key(rows[1]))

    def test_summary_reads_old_rows(self):
        s = load_script("summarize_seeds")
        row = {"phase": "seeds", "method": "knn", "seed": "170", "max_features": "5000", "bigrams": "0",
               "prior": "balanced", "exclude_chapters": "", "train_frac": "1.0", "test_set": "test_nat",
               "accuracy": "0.4", "macro_f1": "0.3", "top3": "0.6", "value_err": "0.4", "ch01_predicted": "5"}
        out = s.summarize([row])  # no "numbers" column at all
        self.assertEqual(list(out)[0][6], "0")


if __name__ == "__main__":
    unittest.main()
