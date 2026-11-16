# How we draw the random samples and even out the HS chapter counts

## Why this is needed
The raw files hold about 65 million rows and the HS chapters are very uneven: a few chapters
(for example machinery, plastics, textiles) have millions of rows and some have only a few
hundred. A plain uniform sample would give a model that sees almost no examples of rare chapters,
and plain accuracy would hide that. We therefore keep two samples and evaluate on both.

## Step A. Random sampling (one streaming pass per yearly file)
Code: `hsclf/sampling.py`, run by `scripts/sample_customs.py`.

1. Read `<year>.csv` row by row. The full file is never loaded into memory.
2. Drop rows that are malformed (wrong number of columns), have no digits-only HS code, have an
   empty description, or whose 2-digit chapter is not in 01 to 97 (77 is reserved by the WCO).
   Counts of every drop reason are logged per year.
3. Chapter = first two digits of the 11-digit PH code (codes shorter than 11 digits are
   left-padded with zeros first; the code-length histogram is logged to check this).
4. Reservoir sampling (Vitter's Algorithm R) runs on the eligible rows:
   * **Natural sample**: one reservoir of K = 5,000 rows per year. Every eligible row of that year
     has the same probability K / N of being kept. It keeps the real chapter distribution.
   * **Balanced pool**: one reservoir of R = 400 rows per chapter per year. Inside a chapter and a
     year every row has the same probability of being kept.
5. Reproducibility: the RNG for each year is `random.Random(170 * 10000 + year)`.
   `data/pool/sampling_log.json` stores the seed, the parameters and the true population count of
   every chapter in every year (used later for reweighting and for the report).

## Step B. Making the chapter counts even
Code: `hsclf/split.py` (`cap_per_chapter`), run by `scripts/build_dataset.py`.

1. Because of step A4 every chapter already contributes at most 400 rows per year, so no big
   chapter can dominate any year.
2. **Cap per chapter**: for each chapter, if it has more than `cap` rows in a split, keep a uniform
   random subset of `cap` rows (train cap 1,500; validation and balanced-test cap 200).
3. **Rare chapters are kept whole, never duplicated.** We do not oversample or copy rows, because
   copies of one row in train and test would inflate the scores. A chapter with fewer rows than
   the cap simply has fewer rows. All such chapters are listed in `balance_report.csv`
   (population count and the count that landed in train, val, test_bal, test_nat).
4. Chapters with no rows in some split (for example a chapter that never appears after 2022) are
   reported, not hidden. Macro-F1 averages over chapters that appear in the test set.
5. Results are always reported on two test sets:
   * `test_bal`: even chapter counts (what macro-F1 measures best), and
   * `test_nat`: the natural distribution of 2023 to 2025 (what a real customs office would see).

## What we will state in the report
* Population count per chapter (from the log), rows per chapter before and after capping.
* How many chapters are "thin" (fewer than 30 test rows) and how their scores are reported.
* The limitation: the balanced training set is not the real distribution, so accuracy on
  `test_nat` can differ from accuracy on `test_bal`. Showing both is the point.
