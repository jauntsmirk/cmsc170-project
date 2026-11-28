# Milestone 2: Progress Report (Team Kramer)

**Project:** Classifying Philippine Import Goods into HS Chapters from Their Descriptions  
**Repository:** https://github.com/jauntsmirk/cmsc170-project

---

## A. Progress Overview

- **Data loading.** A streaming reader handles both header formats (2012–2019 and 2020–2025) and maps them to one schema. It skips malformed rows, rows without an HS code or description, and rows with an invalid chapter, and logs every count. (`hsclf/sampling.py`)
- **Random sampling.** Reservoir sampling (Algorithm R) on each yearly file with a fixed seed per year. It keeps a natural sample (5,000 rows per year) and a balanced pool (up to 400 rows per chapter per year). The full chapter counts are logged. (`scripts/sample_customs.py`, `docs/SAMPLING.md`)
- **Evening out chapter counts.** Each chapter is capped at 1,500 rows for training and 200 for validation and balanced test. Chapters below the cap are kept whole and never copied. `balance_report.csv` records each chapter's population and final counts. (`hsclf/split.py`, `scripts/build_dataset.py`)
- **Split by year.** Train 2012–2021, validation 2022, test 2023–2025. Two test sets: balanced (200 rows per chapter) and natural (real chapter mix).
- **Preprocessing.** Lowercasing, removal of shipment references and quantity notes, tokenizer, stopwords, a 5,000-word vocabulary fitted on training text only, count and tf-idf features, and a hand-written sparse matrix. (`hsclf/text.py`)
- **Metrics from scratch.** Accuracy, macro-F1, weighted F1, top-3 accuracy, confusion matrix, per-chapter precision and recall, and the share of dutiable value (PHP) on wrong predictions. (`hsclf/metrics.py`)
- **Methods from scratch.** Naive Bayes with Laplace smoothing, a multi-class averaged Perceptron, and cosine k-NN. (`hsclf/models.py`)
- **Tests.** 14 unit tests, including metrics and Naive Bayes checked against hand computation.

---

## B. Data We Sampled

| Measure (2012–2025) | Value |
|---|---|
| Eligible rows in the full data | 65,149,026 |
| Chapters found in every year | 96 |
| Malformed rows | 0 |
| Invalid-chapter rows | 0 |
| Balanced pool rows kept | 517,161 |
| Natural sample rows kept | 70,000 |

| Split | Rows | Chapters present | Rows per chapter |
|---|---|---|---|
| Train (2012–2021) | 143,051 | 96 | 551 to 1,500 |
| Validation (2022) | 19,050 | 96 | 50 to 200 |
| Test, balanced (2023–2025) | 19,200 | 96 | 200 |
| Test, natural (2023–2025) | 15,000 | 92 | 1 to 2,420 |

---

## C. Initial Results

One run, tested on 2023–2025. Hyperparameters were chosen on the 2022 validation year, then each model was refit on train plus validation.

| Method | Features | Best setting | Balanced acc | Balanced macro-F1 | Natural acc | Natural macro-F1 | Natural top-3 | Value on wrong rows (natural) |
|---|---|---|---|---|---|---|---|---|
| Naive Bayes | counts | alpha 0.5 | 0.663 | 0.663 | 0.507 | 0.429 | 0.717 | 31.8% |
| Perceptron | tf-idf | 5 epochs | 0.686 | 0.685 | 0.502 | 0.437 | 0.697 | 28.1% |
| k-NN | tf-idf | k = 15 | 0.628 | 0.625 | 0.418 | 0.370 | 0.602 | 44.7% |

For scale: random guessing among 96 chapters gives about 1%, and always guessing the biggest chapter gives 16.1% on the natural test.

- The Perceptron and Naive Bayes are close. The Perceptron leads by 2.2 points on the balanced test (more than test-sample noise of about 0.7 points); on natural accuracy Naive Bayes leads by 0.5 points, which is within noise.
- k-NN is last on every measure, 6 to 9 points behind the best method.
- Natural scores are lower than balanced scores: the training set is balanced, while the natural test is dominated by large, harder chapters.
- The best k (15) and epoch count (5) sit at the edge of the settings tried. k-NN was tuned on 4,000 random validation rows to save time.

**Where the Perceptron goes wrong (natural test).** The weakest chapters with at least 20 test rows are 39 plastics (recall 0.28), 73 iron and steel articles (0.28), 84 machinery (0.35), 85 electrical machinery (0.40) and 90 instruments (0.44). The most common mistakes all involve chapters 84 and 85: electrical to instruments (181 rows), machinery to vehicles (126), electrical to machinery (108), machinery to electrical (107) and machinery to instruments (106). Short part descriptions such as "SPARE PARTS" or "ADAPTOR" fit several of these chapters.

**Chapter 01 side effect.** Chapter 01 (live animals) has only 5,035 rows in 2012–2021 (0.017%) and 1 row in the natural test set, yet the balanced training set keeps 1,500 of its rows, mostly unrelated goods. The Perceptron labels 289 natural test rows (1.9%) as chapter 01, and none is correct.

**Example errors.** The Perceptron got 7,463 of the 15,000 natural test rows wrong. A random sample of its mistakes (`python scripts/show_errors.py`) shows a few patterns:

| True chapter | Predicted | Description | Likely reason |
|---|---|---|---|
| 85 Electrical | 90 Instruments | PCBA | One-word description; circuit boards also appear under instruments |
| 85 Electrical | 01 Live animals | SERVO MOTOR | Too short; chapter 01 absorbs electrical items (label noise) |
| 85 Electrical | 01 Live animals | SPARE PARTS FOR FEED MACHINERY - DOSING BIN MOTOR | The word "feed" probably pulls it toward chapter 01 |
| 84 Machinery | 87 Vehicles | PART OF YAMAHA PRODUCT - CYLINDER HEAD ASSEMBLY | Names a vehicle brand, not the type of part |
| 70 Glass | 69 Ceramics | OPAL GLASS SOUP PLATE | Near miss: tableware can be glass or ceramic |
| 96 Miscellaneous | 74 Copper | GOLDEN BRASS ZIPPERS | The material word (brass) points to a metal chapter |
| 61 Knitted apparel | 62 Woven apparel | PE IMPACT SHORTS | Near miss between two clothing chapters |

The reasons are our reading of each case, not tested explanations.

---

## D. Division of Tasks

| Member | Task | Can explain |
|---|---|---|
| Ivan Joas Managat | Data download and sampling | `sampling.py`, `sample_customs.py` |
| Jose Anrhed Pesimo | Preprocessing and features | `text.py` |
| Mark Justin Feliciano | Metrics and splits | `metrics.py`, `split.py`, `build_dataset.py` |
| Jose Anrhed Pesimo | Models and experiments | `models.py`, `run_experiments.py` |

---

## E. Problems Encountered

- The first dataset could not be used for the project and was too large for our devices, so we changed topic and now sample one file at a time.
- Two header formats and inconsistent country codes. Only use columns common to both.
- Strong chapter imbalance, handled by per-chapter sampling and capping.
- **10-digit HS codes from 2020.** About 1.4% to 1.7% of rows each year have 10 digits because the export dropped the leading zero of chapters 01–09. We pad codes to 11 digits and verified it: every 10-digit row lands in chapters 01–09, and no 11-digit code starts with 0 from 2020 on.
- **Label noise.** The source labels contain data-entry errors. In our sample of chapter 01, only about 4 of 25 rows were real live animals; the rest were items like grommets, printer parts and cables. This limits accuracy in that chapter and lowers macro-F1.
- **Volume jump.** Eligible rows rise from 3.17 million in 2020 to 6.81 million in 2021 and 9.43 million in 2025, and the source format changed around 2020. We sample equally per year, so the sample is not proportional to volume.
- **Thin chapters in the natural test set.** 4 chapters never appear and some have a single row.
- **Our own mistakes, fixed.** The first sampler overwrote its output on every run, so it now saves each year separately. An early k-NN run used the first rows of sorted test files, which biased it to the first chapters and one year; it now uses a seeded random subset, and the biased numbers were discarded.

---

## F. Plan for Milestone 3

- Tune features (vocabulary size, word pairs) and compare methods fairly over several random seeds.
- Try class priors that follow the real chapter frequencies, to reduce over-prediction of rare and noisy chapters.
- Plot a learning curve to justify the sample size.
- Add a second result that excludes noisy chapters such as chapter 01, to show what label noise costs.
- Optionally add a Decision Tree (CART) as a fourth method.
- Build the GitHub Pages project page and the presentation.
