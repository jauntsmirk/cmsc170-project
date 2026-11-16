import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsclf.metrics import confusion_matrix, macro_f1, per_class_prf, top_k_accuracy, value_weighted_error
from hsclf.models import KNN, MultinomialNB, Perceptron
from hsclf.sampling import Reservoir, hs_chapter
from hsclf.split import cap_per_chapter, parse_years
from hsclf.text import Vectorizer, tokenize


class TestSampling(unittest.TestCase):
    def test_reservoir_is_uniform(self):
        hits = np.zeros(20)
        for seed in range(4000):
            r = Reservoir(5, random.Random(seed))
            for i in range(20):
                r.add(i)
            for i in r.items:
                hits[i] += 1
        p = hits / 4000
        self.assertTrue(np.all(np.abs(p - 0.25) < 0.04), p)  # each item kept w.p. 5/20

    def test_reservoir_keeps_all_if_small(self):
        r = Reservoir(10, random.Random(0))
        for i in range(4):
            r.add(i)
        self.assertEqual(sorted(r.items), [0, 1, 2, 3])

    def test_chapter(self):
        self.assertEqual(hs_chapter("15119090000"), "15")
        self.assertEqual(hs_chapter("1012100000"), "01")  # lost leading zero

    def test_cap_per_chapter(self):
        rows = [{"hs_chapter": "01"}] * 50 + [{"hs_chapter": "02"}] * 3
        out = cap_per_chapter(rows, 10, random.Random(0))
        self.assertEqual(sum(r["hs_chapter"] == "01" for r in out), 10)
        self.assertEqual(sum(r["hs_chapter"] == "02" for r in out), 3)  # rare chapter kept whole

    def test_years(self):
        self.assertEqual(parse_years("2012-2014,2016"), {2012, 2013, 2014, 2016})


class TestText(unittest.TestCase):
    def test_tokenize_strips_refs_and_qty(self):
        t = tokenize("COCONUT OIL 5L [TR-H1692] (QTY 2,000 PCS) for sale")
        self.assertEqual(t, ["coconut", "oil", "sale"])

    def test_vectorizer_fit_on_train_only(self):
        v = Vectorizer(mode="count", min_df=1).fit(["apple pie", "apple tart"])
        X = v.transform(["apple banana"])  # banana unseen -> ignored
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(X.data.tolist(), [1.0])

    def test_tfidf_rows_unit_norm(self):
        v = Vectorizer(mode="tfidf", min_df=1)
        X = v.fit_transform(["red apple", "green apple pie", "pie"])
        self.assertTrue(np.allclose(np.linalg.norm(X.dense(0, 3), axis=1), 1.0, atol=1e-5))


class TestMetrics(unittest.TestCase):
    def test_hand_computed(self):
        y = [0, 0, 1, 1, 2]
        p = [0, 1, 1, 1, 0]
        cm = confusion_matrix(y, p, 3)
        self.assertEqual(cm.tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 0]])
        prec, rec, f1, sup = per_class_prf(cm)
        self.assertAlmostEqual(rec[0], 0.5)
        self.assertAlmostEqual(prec[1], 2 / 3)
        self.assertAlmostEqual(f1[1], 0.8)
        self.assertAlmostEqual(macro_f1(cm), (0.5 + 0.8 + 0.0) / 3)

    def test_top_k_and_value(self):
        s = np.array([[0.1, 0.5, 0.4], [0.7, 0.2, 0.1]])
        self.assertEqual(top_k_accuracy([1, 1], s, 1), 0.5)
        self.assertEqual(top_k_accuracy([2, 1], s, 2), 1.0)
        self.assertAlmostEqual(value_weighted_error([0, 1], [0, 0], [10, 30]), 0.75)


def toy():
    from hsclf.text import Vectorizer
    texts = ["fish fresh salmon", "frozen fish fillet", "salmon fillet fish", "cotton shirt men",
             "cotton fabric roll", "men shirt cotton", "steel bolt screw", "screw nut steel", "bolt steel nut"]
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return texts, y


class TestModels(unittest.TestCase):
    def test_nb_laplace_matches_hand_computation(self):
        v = Vectorizer(mode="count", min_df=1).fit(["aa bb", "aa cc"])
        X = v.transform(["aa bb", "aa cc"])
        nb = MultinomialNB(2, alpha=1.0).fit(X, np.array([0, 1]))
        # vocab a,b,c ; class 0 counts a=1,b=1,c=0 -> P(b|0) = (1+1)/(2+3)
        self.assertAlmostEqual(float(np.exp(nb.log_lik[0, v.vocab["bb"]])), 2 / 5)
        self.assertAlmostEqual(float(np.exp(nb.log_lik[0, v.vocab["cc"]])), 1 / 5)

    def test_all_models_learn_toy(self):
        texts, y = toy()
        v = Vectorizer(mode="tfidf", min_df=1)
        X = v.fit_transform(texts)
        for m in (MultinomialNB(3), Perceptron(3, epochs=20), KNN(3, k=3)):
            pred = m.fit(X, y).predict(X)
            self.assertTrue((pred == y).all(), type(m).__name__)

    def test_knn_chunking_equivalent(self):
        texts, y = toy()
        X = Vectorizer(mode="tfidf", min_df=1).fit_transform(texts)
        a = KNN(3, k=3).fit(X, y).scores(X)
        b = KNN(3, k=3, q_batch=2, t_chunk=4).fit(X, y).scores(X)
        self.assertTrue(np.allclose(a, b, atol=1e-5))

    def test_perceptron_deterministic(self):
        texts, y = toy()
        X = Vectorizer(mode="tfidf", min_df=1).fit_transform(texts)
        w1 = Perceptron(3, 5, seed=1).fit(X, y).W
        w2 = Perceptron(3, 5, seed=1).fit(X, y).W
        self.assertTrue(np.array_equal(w1, w2))


if __name__ == "__main__":
    unittest.main()
