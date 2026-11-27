# Milestone 3 runner (task A)

Runs Naive Bayes, the Perceptron and k-NN over several seeds and feature settings,
saves every result to `results/seeds.csv`, and summarizes them as mean ± std in
`results/seeds_summary.md`.

## Setup: use a virtual environment

NumPy installed with `apt` can use a slow reference BLAS (we measured 5 GFLOPS
vs 418 GFLOPS with pip's NumPy on the same laptop: about 80x slower). Use a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate        # every new terminal
pip install -r requirements.txt
```

Check the speed (should be in the hundreds of GFLOPS):

```bash
python -c "
import numpy as np, time
a=np.random.rand(2048,5000).astype(np.float32); b=np.random.rand(5000,2000).astype(np.float32)
a@b
t=time.time(); a@b; dt=time.time()-t; print(f'{dt:.2f}s  {2*2048*5000*2000/dt/1e9:.0f} GFLOPS')
"
```

## Run it

Needs `data/pool/balanced_pool.csv` and `data/processed/test_*.csv` (steps 1 and 3 of the run guide).

```bash
# 1. feature sweep: every vocab size x bigram setting, seed 170, validation only (~25 min)
python scripts/run_seeds.py --phase sweep --vocab 2000,5000,10000,20000,40000

# 2. seeds: each method with its best sweep setting, seeds 170-174, both test sets (~100 min)
python scripts/run_seeds.py --phase seeds

# 3. summary: mean and sample std per method
python scripts/summarize_seeds.py
```

Runs are saved as they finish, and a rerun skips anything already in the CSV, so it is
safe to stop and restart. For long runs, start them in the background:
`nohup python scripts/run_seeds.py --phase seeds > results/seeds_run.log 2>&1 &`

Timings are from an Intel Core Ultra 9 185H with pip's NumPy. k-NN takes most of the time.

## Design decisions

- **The seed changes the training data.** Each seed decides which rows survive the
  1,500-per-chapter cap (`build_splits` in `hsclf/experiment.py`). Without this,
  Naive Bayes would give identical results for every seed (std = 0).
- **Test sets are fixed.** Every seed is scored on the same test rows, so the spread comes
  only from training. The std therefore measures sensitivity to the training sample, not
  test-sample noise (about 0.7 points, Milestone 2 report): treat gaps under ~1 point as ties.
- **Seed 170 = Milestone 2.** It rebuilds `data/processed/train.csv` and `val.csv` row for row.
- **Features chosen on validation only.** The sweep never looks at the test sets.
- **Wider grids than Milestone 2.** Perceptron epochs {2, 3, 5, 10}, k-NN k {1, 5, 15, 30, 50},
  because 5 epochs and k = 15 were at the edge of the old grids.

## Results (5 seeds, vocab 40,000 with bigrams)

| Method | Balanced acc | Balanced macro-F1 | Natural acc | Natural macro-F1 | Natural top-3 |
|---|---|---|---|---|---|
| Perceptron | 0.7227 ± 0.0016 | 0.7219 ± 0.0019 | 0.5480 ± 0.0030 | 0.4813 ± 0.0030 | 0.7323 ± 0.0025 |
| Naive Bayes | 0.6991 ± 0.0011 | 0.7007 ± 0.0013 | 0.5500 ± 0.0031 | 0.4834 ± 0.0010 | 0.7502 ± 0.0005 |
| k-NN | 0.6514 ± 0.0094 | 0.6502 ± 0.0091 | 0.4504 ± 0.0231 | 0.3999 ± 0.0134 | 0.6327 ± 0.0443 |

Full tables, including value on wrong rows and chapter-01 predictions: `results/seeds_summary.md`.

- **The Perceptron wins on the balanced test.** It leads Naive Bayes by 2.4 points, far beyond the seed spread.
- **Naive Bayes and the Perceptron tie on the natural test.** The 0.2-point accuracy gap is smaller than the std. Naive Bayes is clearly better on top-3.
- **Every method gains 3.5 to 4.5 points over Milestone 2**, from using every word plus bigrams.
- **Bigrams help only when they don't crowd out words.** Under a 5k or 10k cap they took slots from single words and lowered scores. Above the ~33k available features they add about 1 point.
- **k-NN is the least stable.** Its std is 5–10x the others', and its best k changes with the seed.

## For B and C

Everything goes through
`run_one(method, seed, max_features, bigrams, prior, exclude_chapters, train_frac, phase)` in `hsclf/experiment.py`.

- **Priors (B): done.** `run_seeds.py --phase seeds --prior real`. See "Real chapter priors" below.
- **Number tokens: done.** `run_seeds.py --phase sweep --numbers 1`, then `--phase seeds --numbers 1`.
  See "Number tokens" below.
- **Learning curve (B):** pass `train_frac` (e.g. 0.1, 0.25, 0.5).
- **Noisy chapters (C):** pass `exclude_chapters=("01",)`. It drops those chapters from every split.
- **Count of rows predicted as chapter 01:** already in the `ch01_predicted` column.

`run_seeds.py` currently loops only over methods, seeds and features. Call `run_one` from your own
script (or add a flag) and append to `results/seeds.csv` with `append_rows` so the summary picks it up.

Tests: `tests/test_runner.py` (fake data, no download needed).
## Real chapter priors (`--prior real`)

Training is balanced, so every model treats all chapters as equally common. Real traffic is
not balanced: a few chapters (84, 85, 39) are a large share of all rows. With `--prior real`,
each model's scores are corrected by Bayes' rule after scoring (`adjust_for_prior` in
`hsclf/models.py`):

    P_real(chapter | words)  is proportional to  P_train(chapter | words) * real_share / train_share

- **Real shares** come from the full-file chapter counts in `data/pool/sampling_log.json`,
  using the training years only (2012-2021; 2012-2022 for the final refit). Never the test years.
- **Strength** scales the correction: 0 = balanced model as before, 1 = full Bayes correction.
  It is chosen from `PRIOR_GRID = [0, 0.25, 0.5, 1, 2, 4]` by accuracy on the 2022 rows of
  `natural_sample.csv` (real chapter mix, not a test set). Ties keep the weaker strength.
- **Per method:** Naive Bayes and the Perceptron add `strength * log(real/train)` to their scores
  (for Naive Bayes, strength 1 is exactly swapping the balanced prior for the real one). k-NN
  multiplies each chapter's votes by `(real/train) ** strength`.
- **k-NN limit:** the prior can only reweight chapters already among the k neighbours
  (0 votes times any weight is 0), so it helps less with small k.
- **What to expect:** higher natural-test accuracy, and possibly lower macro-F1 and balanced-test
  scores, because the prior favours common chapters. Report both priors side by side.
- The chosen strength is saved in the `prior_strength` column and shown in the summary.

## Number tokens (`--numbers 1`)

The default tokenizer keeps letters only, so "220V MOTOR 1.5HP" becomes `motor hp`. With
`--numbers 1`, a number glued to a unit or name is kept with its digits replaced by `#`:
`#v motor hp #hp`. So "110V" and "220V" count as the same word, while a lone number such as
"2000" is still dropped. Every old token is kept ("USB2" gives `usb` and `usb#`), so this
option only adds words. Run the sweep with `--numbers 1` first so the seeds phase can pick the
best vocabulary for it (if there is no such sweep, it falls back to the `--numbers 0` setting).

## Results CSV columns

`numbers` and `prior_strength` were added. An older `results/seeds.csv` is upgraded in place the
first time new rows are appended (old rows get `numbers=0`); existing values are not changed.
With the default settings (`--prior balanced --numbers 0`) results are identical to before.

Tests: `tests/test_priors_numbers.py` (fake data, no download needed).
