"""
From-scratch classifiers (numpy only). All take a CSR matrix and integer labels 0..C-1.

    MultinomialNB   Naive Bayes with Laplace (add-alpha) smoothing, log-space scoring
    Perceptron      multi-class (one weight vector per chapter), optional averaging
    KNN             cosine k-NN on L2-normalised tf-idf, chunked matrix products

Every model exposes fit(X, y), scores(X) -> (n, C) array, predict(X) -> labels.
"""
import numpy as np

BATCH = 2048


# ---------------------------------------------------------------------------
# Real chapter priors
#
# Training is balanced (every chapter capped at the same count), so every model
# learns as if all chapters were equally common. Real customs traffic is not:
# a few chapters (84, 85, 39) are a large share of all rows. These two functions
# correct for that after the model has scored a description, by Bayes' rule:
#
#     P_real(chapter | words)  is proportional to  P_train(chapter | words) * real_share / train_share
#
# `strength` scales the correction: 0 = no change (the balanced model as before),
# 1 = the full Bayes correction. It is chosen on the 2022 natural sample, never on test.
# ---------------------------------------------------------------------------

def prior_log_ratio(real_counts, train_counts):
    """log(real share / training share) per chapter. Both shares are add-one smoothed,
    so a chapter with a count of 0 still gets a finite value."""
    real = np.asarray(real_counts, dtype=np.float64) + 1.0
    train = np.asarray(train_counts, dtype=np.float64) + 1.0
    return np.log(real / real.sum()) - np.log(train / train.sum())


def adjust_for_prior(scores, log_ratio, strength, votes=False):
    """
    Applies the prior correction to an (n, C) score array.
      votes=False  Naive Bayes and Perceptron: scores are log-scale, so ADD
                   strength * log_ratio (for Naive Bayes with strength 1 this swaps
                   the balanced prior for the real one).
      votes=True   k-NN: scores are summed votes, so MULTIPLY each chapter's votes
                   by ratio ** strength (exp(strength * log_ratio)).
    """
    if strength == 0:
        return scores
    if votes:
        return scores * np.exp(strength * log_ratio)
    return scores + strength * log_ratio


class MultinomialNB:
    def __init__(self, n_classes, alpha=1.0):
        self.C, self.alpha = n_classes, alpha

    def fit(self, X, y):
        y = np.asarray(y)
        V = X.shape[1]
        counts = np.zeros((self.C, V), dtype=np.float64)
        for s in range(0, X.shape[0], BATCH):
            e = min(s + BATCH, X.shape[0])
            Xb = X.dense(s, e).astype(np.float64)
            onehot = np.zeros((e - s, self.C))
            onehot[np.arange(e - s), y[s:e]] = 1.0
            counts += onehot.T @ Xb
        class_n = np.bincount(y, minlength=self.C).astype(np.float64)
        # Laplace smoothing: P(w|c) = (count(w,c) + alpha) / (total(c) + alpha * V)
        self.log_lik = np.log(counts + self.alpha) - np.log(counts.sum(1, keepdims=True) + self.alpha * V)
        # prior, smoothed so a chapter absent from training is never -inf
        self.log_prior = np.log((class_n + self.alpha) / (class_n.sum() + self.alpha * self.C))
        return self

    def scores(self, X):
        out = np.empty((X.shape[0], self.C))
        for s in range(0, X.shape[0], BATCH):
            e = min(s + BATCH, X.shape[0])
            out[s:e] = X.dense(s, e) @ self.log_lik.T + self.log_prior
        return out

    def predict(self, X):
        return self.scores(X).argmax(1)


class Perceptron:
    """
    Multi-class perceptron. For each sample: predict argmax_c w_c . x; on a mistake
        w_true += x ;  w_pred -= x
    (bias handled as an always-on feature). Samples are shuffled every epoch with a
    seeded RNG. averaged=True returns the average of all weight vectors seen
    (Daume's O(nnz) trick), which is far more stable than the last weights.
    """

    def __init__(self, n_classes, epochs=10, averaged=True, seed=170):
        self.C, self.epochs, self.averaged, self.seed = n_classes, epochs, averaged, seed
        self.history = []

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n, V = X.shape
        W = np.zeros((self.C, V + 1))      # last column = bias
        U = np.zeros_like(W)               # sum of c * update, for averaging
        rng = np.random.RandomState(self.seed)
        c = 1
        for ep in range(self.epochs):
            mistakes = 0
            for i in rng.permutation(n):
                idx, val = X.row(i)
                s = W[:, idx] @ val + W[:, V]
                p = int(s.argmax())
                t = int(y[i])
                if p != t:
                    mistakes += 1
                    W[t, idx] += val
                    W[t, V] += 1.0
                    W[p, idx] -= val
                    W[p, V] -= 1.0
                    if self.averaged:
                        U[t, idx] += c * val
                        U[t, V] += c
                        U[p, idx] -= c * val
                        U[p, V] -= c
                c += 1
            self.W_last = W.copy()
            self.W = W - U / c if self.averaged else W.copy()
            rec = {"epoch": ep + 1, "train_mistakes": mistakes, "train_mistake_rate": mistakes / n}
            if X_val is not None:
                rec["val_accuracy"] = float(np.mean(self.predict(X_val) == np.asarray(y_val)))
            self.history.append(rec)
            if mistakes == 0:
                break
        return self

    def scores(self, X):
        V = X.shape[1]
        out = np.empty((X.shape[0], self.C))
        for s in range(0, X.shape[0], BATCH):
            e = min(s + BATCH, X.shape[0])
            out[s:e] = X.dense(s, e) @ self.W[:, :V].T + self.W[:, V]
        return out

    def predict(self, X):
        return self.scores(X).argmax(1)


class KNN:
    """
    Cosine k-NN. Inputs must be L2-normalised, so cosine similarity = dot product.
    Training rows are processed in chunks, keeping a running top-k per query.
    Vote weight = similarity (weighted=True) or 1.
    """

    def __init__(self, n_classes, k=5, weighted=True, q_batch=2048, t_chunk=20000):
        self.C, self.k, self.weighted, self.qb, self.tc = n_classes, k, weighted, q_batch, t_chunk

    def fit(self, X, y):
        self.X, self.y = X, np.asarray(y)
        return self

    def neighbours(self, Xq, k):
        """(sims, labels) of the k nearest training rows for every query row, best first."""
        nq = Xq.shape[0]
        best_s = np.full((nq, k), -np.inf, dtype=np.float32)
        best_y = np.zeros((nq, k), dtype=np.int64)
        for qs in range(0, nq, self.qb):
            qe = min(qs + self.qb, nq)
            Q = Xq.dense(qs, qe)
            bs, by = best_s[qs:qe], best_y[qs:qe]
            for ts in range(0, self.X.shape[0], self.tc):
                te = min(ts + self.tc, self.X.shape[0])
                S = Q @ self.X.dense(ts, te).T
                cat_s = np.concatenate([bs, S], axis=1)
                cat_y = np.concatenate([by, np.broadcast_to(self.y[ts:te], S.shape)], axis=1)
                top = np.argpartition(-cat_s, k - 1, axis=1)[:, :k]
                bs = np.take_along_axis(cat_s, top, 1)
                by = np.take_along_axis(cat_y, top, 1)
            order = np.argsort(-bs, axis=1)
            best_s[qs:qe] = np.take_along_axis(bs, order, 1)
            best_y[qs:qe] = np.take_along_axis(by, order, 1)
        return best_s, best_y

    def scores_from_neighbours(self, sims, labels, k=None):
        k = k or self.k
        sims, labels = sims[:, :k], labels[:, :k]
        w = np.maximum(sims, 0) if self.weighted else np.ones_like(sims)
        out = np.zeros((sims.shape[0], self.C))
        rows = np.repeat(np.arange(sims.shape[0]), k)
        np.add.at(out, (rows, labels.ravel()), w.ravel())
        return out

    def scores(self, X):
        s, l = self.neighbours(X, self.k)
        return self.scores_from_neighbours(s, l)

    def predict(self, X):
        return self.scores(X).argmax(1)
