#!/usr/bin/env python3
"""
Writes FAKE yearly CSVs (both header formats) into data/synthetic_raw/ so the
whole pipeline can be tested without downloading the real 12 GB. The chapters
have Zipf-like imbalance and chapter-specific words plus shipment-reference noise.
NOT real data: never report results from it.
"""
import argparse
import csv
import os
import random

WORDS_PER_CHAPTER = 12
COMMON = ["brand", "new", "pcs", "ctns", "packed", "in", "boxes", "type", "model", "for", "sale",
          "assorted", "size", "made", "of", "grade", "white", "black", "set"]


def make_vocab(rng, n_chapters):
    letters = "abcdefghijklmnoprstuvw"
    vocab = {}
    for c in range(n_chapters):
        vocab[c] = ["".join(rng.choice(letters) for _ in range(rng.randint(4, 9)))
                    for _ in range(WORDS_PER_CHAPTER)]
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/synthetic_raw")
    ap.add_argument("--chapters", type=int, default=30)
    ap.add_argument("--rows-per-year", type=int, default=6000)
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    chapter_ids = [f"{c:02d}" for c in range(1, 98) if c != 77][: a.chapters]
    vocab = make_vocab(rng, a.chapters)
    weights = [1.0 / (i + 1) for i in range(a.chapters)]  # Zipf-like imbalance
    lo, _, hi = a.years.partition("-")
    os.makedirs(a.out_dir, exist_ok=True)
    for y in range(int(lo), int(hi) + 1):
        old = y <= 2019
        header = (["uid", "ty", "tq", "tm", "entry", "hscode", "goodsdescription", "p", "q", "dutiablevaluephp"]
                  if old else ["HSCODE", "GOODSDESCRIPTION", "UOM", "QUANTITY", "year", "month", "DUTIABLEVALUEPHP"])
        with open(os.path.join(a.out_dir, f"{y}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for i in range(a.rows_per_year):
                c = rng.choices(range(a.chapters), weights)[0]
                words = rng.sample(vocab[c], rng.randint(2, 4)) + rng.sample(COMMON, rng.randint(1, 3))
                if rng.random() < 0.15:  # label noise-ish: borrow a word from another chapter
                    words.append(rng.choice(vocab[rng.randrange(a.chapters)]))
                rng.shuffle(words)
                desc = " ".join(words).upper()
                if rng.random() < 0.5:
                    desc += f" [TR-H{rng.randint(1000, 9999)}] (QTY {rng.randint(1, 9)},{rng.randint(100, 999)} PCS)"
                hs = chapter_ids[c] + "".join(str(rng.randint(0, 9)) for _ in range(9))
                val = str(round(rng.lognormvariate(10, 1.5), 2))
                if old:
                    w.writerow([i, y, 1, rng.randint(1, 12), "E", hs, desc, "P", rng.randint(1, 500), val])
                else:
                    w.writerow([hs, desc, "KGM", rng.randint(1, 500), y, rng.randint(1, 12), val])
    print("wrote synthetic CSVs to", a.out_dir)


if __name__ == "__main__":
    main()
