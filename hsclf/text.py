"""
Text preprocessing and feature extraction, written from scratch (re + numpy).

Pipeline for one description:
    raw -> clean() -> tokenize() -> Vocabulary.encode() -> CSR bag-of-words
    -> optional tf-idf weighting -> optional L2 normalisation

The vocabulary and the idf weights are fitted on the TRAINING rows only.
"""
import re
from collections import Counter

import numpy as np

BRACKET_RE = re.compile(r"\[[^\]]*\]")                    # shipment refs: [TR-H1692]
QTY_RE = re.compile(r"\(\s*(qty|quantity)\b[^)]*\)", re.I)  # (QTY 2,000 PCS)
TOKEN_RE = re.compile(r"[a-z]+")

STOPWORDS = frozenset("""a an and are as at be by for from in is it of on or the to with
pcs pc ctn ctns pkg pkgs pack packed packing box boxes qty unit units set sets""".split())


def clean(text):
    text = BRACKET_RE.sub(" ", text)
    text = QTY_RE.sub(" ", text)
    return text.lower()


def tokenize(text, min_len=2, stopwords=STOPWORDS):
    return [t for t in TOKEN_RE.findall(clean(text)) if len(t) >= min_len and t not in stopwords]


def with_bigrams(tokens):
    return tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]


class CSR:
    """Minimal compressed-sparse-row matrix: indptr, indices, data, shape."""

    def __init__(self, indptr, indices, data, shape):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = shape

    def row(self, i):
        s, e = self.indptr[i], self.indptr[i + 1]
        return self.indices[s:e], self.data[s:e]

    def dense(self, start, stop):
        """Rows start..stop as a dense float32 array."""
        out = np.zeros((stop - start, self.shape[1]), dtype=np.float32)
        s, e = self.indptr[start], self.indptr[stop]
        rows = np.repeat(np.arange(stop - start), np.diff(self.indptr[start:stop + 1]))
        out[rows, self.indices[s:e]] = self.data[s:e]
        return out

    def row_sums(self):
        sums = np.zeros(self.shape[0], dtype=np.float64)
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        np.add.at(sums, rows, self.data)
        return sums

    def scale_columns(self, w):
        return CSR(self.indptr, self.indices, self.data * w[self.indices], self.shape)

    def l2_normalize(self):
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        sq = np.zeros(self.shape[0], dtype=np.float64)
        np.add.at(sq, rows, self.data.astype(np.float64) ** 2)
        norm = np.sqrt(sq)
        norm[norm == 0] = 1.0
        return CSR(self.indptr, self.indices, self.data / norm[rows].astype(np.float32), self.shape)

    def take_rows(self, idx):
        idx = np.asarray(idx)
        counts = self.indptr[idx + 1] - self.indptr[idx]
        indptr = np.concatenate([[0], np.cumsum(counts)])
        parts_i = [self.indices[self.indptr[i]:self.indptr[i + 1]] for i in idx]
        parts_d = [self.data[self.indptr[i]:self.indptr[i + 1]] for i in idx]
        ind = np.concatenate(parts_i) if parts_i else np.array([], dtype=np.int32)
        dat = np.concatenate(parts_d) if parts_d else np.array([], dtype=np.float32)
        return CSR(indptr, ind, dat, (len(idx), self.shape[1]))


class Vectorizer:
    """
    mode: 'count' | 'binary' | 'tfidf'      (tfidf = count * idf, then L2-normalised)
    Vocabulary = tokens with document frequency >= min_df, top `max_features` by df
    (ties broken alphabetically so the result is deterministic).
    """

    def __init__(self, mode="tfidf", max_features=5000, min_df=3, bigrams=False):
        self.mode, self.max_features, self.min_df, self.bigrams = mode, max_features, min_df, bigrams
        self.vocab, self.idf = None, None

    def _tokens(self, text):
        t = tokenize(text)
        return with_bigrams(t) if self.bigrams else t

    def fit(self, texts):
        df = Counter()
        n = 0
        for t in texts:
            n += 1
            df.update(set(self._tokens(t)))
        kept = [(tok, c) for tok, c in df.items() if c >= self.min_df]
        kept.sort(key=lambda tc: (-tc[1], tc[0]))
        kept = kept[: self.max_features]
        self.vocab = {tok: i for i, (tok, _) in enumerate(kept)}
        dfv = np.array([c for _, c in kept], dtype=np.float64)
        self.idf = (np.log((1.0 + n) / (1.0 + dfv)) + 1.0).astype(np.float32)  # smoothed idf
        return self

    def transform(self, texts):
        indptr, indices, data = [0], [], []
        for t in texts:
            counts = Counter(self.vocab[tok] for tok in self._tokens(t) if tok in self.vocab)
            for j in sorted(counts):
                indices.append(j)
                data.append(1.0 if self.mode == "binary" else float(counts[j]))
            indptr.append(len(indices))
        X = CSR(indptr, indices, data, (len(indptr) - 1, len(self.vocab)))
        if self.mode == "tfidf":
            X = X.scale_columns(self.idf).l2_normalize()
        return X

    def fit_transform(self, texts):
        texts = list(texts)
        return self.fit(texts).transform(texts)
