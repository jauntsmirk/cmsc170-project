"""
Tests for the Milestone 3 runner (hsclf/experiment.py, scripts/run_seeds.py,
scripts/summarize_seeds.py). They use a tiny fake pool, so no data download is needed.
"""
import importlib.util
import os
import sys
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hsclf import experiment


def load_script(name):
    """Imports scripts/<name>.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_rows(year, chapter, n):
    return [{"year": str(year), "hs_chapter": chapter, "source_row": f"{year}-{chapter}-{i}",
             "goods_description": f"item {chapter}", "dutiable_value_php": "1"} for i in range(n)]


class TestSplits(unittest.TestCase):
    def setUp(self):
        # 3,000 rows of chapter 84 per training year: far above the 1,500 cap, so the seed matters.
        # Chapter 01 has only 50 rows per year: kept whole by every seed.
        pool = []
        for y in (2015, 2016):
            pool += fake_rows(y, "84", 3000) + fake_rows(y, "01", 50)
        pool += fake_rows(2022, "84", 500) + fake_rows(2022, "01", 20)
        experiment._DATA.clear()
        experiment._DATA.update({"pool": pool, "test_bal": fake_rows(2024, "84", 10) + fake_rows(2024, "01", 5),
                                 "test_nat": fake_rows(2024, "84", 30)})

    def tearDown(self):
        experiment._DATA.clear()

    def ids(self, rows):
        return [r["source_row"] for r in rows]

    def test_same_seed_same_rows(self):
        a, b = experiment.build_splits(170)[0], experiment.build_splits(170)[0]
        self.assertEqual(self.ids(a), self.ids(b))

    def test_different_seed_different_rows(self):
        a, b = experiment.build_splits(170)[0], experiment.build_splits(171)[0]
        self.assertNotEqual(set(self.ids(a)), set(self.ids(b)))
        self.assertEqual(len(a), len(b))  # same size: 1,500 of chapter 84 + all 100 of chapter 01

    def test_caps_and_rare_chapter_kept(self):
        train, val, _ = experiment.build_splits(170)
        self.assertEqual(sum(r["hs_chapter"] == "84" for r in train), 1500)
        self.assertEqual(sum(r["hs_chapter"] == "01" for r in train), 100)
        self.assertEqual(sum(r["hs_chapter"] == "84" for r in val), 200)

    def test_test_sets_fixed_across_seeds(self):
        t1, t2 = experiment.build_splits(170)[2], experiment.build_splits(171)[2]
        self.assertEqual(self.ids(t1["test_bal"]), self.ids(t2["test_bal"]))

    def test_exclude_chapters_everywhere(self):
        train, val, tests = experiment.build_splits(170, exclude_chapters=("01",))
        for rows in [train, val, tests["test_bal"], tests["test_nat"]]:
            self.assertFalse(any(r["hs_chapter"] == "01" for r in rows))

    def test_train_frac(self):
        full = experiment.build_splits(170)[0]
        half = experiment.build_splits(170, train_frac=0.5)[0]
        self.assertEqual(len(half), round(len(full) * 0.5))
        self.assertTrue(set(self.ids(half)) <= set(self.ids(full)))


class TestRunSeeds(unittest.TestCase):
    def setUp(self):
        self.rs = load_script("run_seeds")

    def row(self, phase, test_set, **kw):
        r = {"phase": phase, "method": "knn", "seed": "170", "max_features": "5000", "bigrams": "0",
             "prior": "balanced", "exclude_chapters": "", "train_frac": "1.0", "test_set": test_set,
             "macro_f1": "0.5"}
        r.update(kw)
        return r

    def test_done_needs_every_test_set(self):
        half = [self.row("seeds", "test_bal")]
        full = half + [self.row("seeds", "test_nat")]
        self.assertEqual(self.rs.done_keys(half), set())   # interrupted run: must rerun
        self.assertEqual(len(self.rs.done_keys(full)), 1)

    def test_key_matches_csv_and_python_values(self):
        from_csv = self.row("sweep", "val")
        from_code = {"phase": "sweep", "method": "knn", "seed": 170, "max_features": 5000, "bigrams": False,
                     "prior": "balanced", "exclude_chapters": "", "train_frac": 1.0}
        self.assertEqual(self.rs.run_key(from_csv), self.rs.run_key(from_code))

    def test_best_features_uses_validation_sweep(self):
        rows = [self.row("sweep", "val", max_features="5000", macro_f1="0.60"),
                self.row("sweep", "val", max_features="40000", bigrams="1", macro_f1="0.65"),
                self.row("seeds", "test_bal", max_features="2000", macro_f1="0.99")]  # test rows ignored
        self.assertEqual(self.rs.best_features(rows, "knn"), (40000, 1, 0.65))


class TestSummary(unittest.TestCase):
    def test_mean_and_sample_std(self):
        s = load_script("summarize_seeds")
        mean, std = s.mean_std([0.5072, 0.5104])
        self.assertAlmostEqual(mean, 0.5088)
        self.assertAlmostEqual(std, 0.0032 / np.sqrt(2))  # ddof=1: |a - b| / sqrt(2) for two values
        self.assertIsNone(s.mean_std([0.5])[1])          # one seed: no std


if __name__ == "__main__":
    unittest.main()