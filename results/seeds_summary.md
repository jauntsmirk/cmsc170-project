# Results over seeds

Mean ± sample standard deviation (ddof=1) over the seeds listed. The seed changes which training rows survive the per-chapter cap; the test sets are the same for every seed.

## Balanced test (2023-2025, 200 rows per chapter)

| Method | Vocab | Bigrams | Numbers | Prior | Excluded | Train frac | Seeds | Accuracy | Macro-F1 | Top-3 acc. | Value on wrong rows | Predicted as ch. 01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 40000 | yes | no | balanced | - | 1.0 | 5 | 0.6514 ± 0.0094 | 0.6502 ± 0.0091 | 0.7775 ± 0.0227 | 8.3% ± 0.5 | 310 ± 72 |
| naive_bayes | 40000 | yes | no | balanced | - | 1.0 | 5 | 0.6991 ± 0.0011 | 0.7007 ± 0.0013 | 0.8290 ± 0.0006 | 7.1% ± 0.4 | 213 ± 9 |
| naive_bayes | 40000 | yes | no | real (strength 1, 2) | - | 1.0 | 5 | 0.6448 ± 0.0304 | 0.6729 ± 0.0195 | 0.7773 ± 0.0280 | 13.5% ± 11.1 | 42 ± 8 |
| perceptron | 40000 | yes | no | balanced | - | 1.0 | 5 | 0.7227 ± 0.0016 | 0.7219 ± 0.0019 | 0.8408 ± 0.0022 | 7.0% ± 0.4 | 146 ± 10 |
| perceptron | 40000 | yes | no | real (strength 0.25, 0.5) | - | 1.0 | 5 | 0.6497 ± 0.0566 | 0.6792 ± 0.0415 | 0.7765 ± 0.0474 | 18.3% ± 16.5 | 40 ± 21 |

## Natural test (2023-2025, real chapter mix)

| Method | Vocab | Bigrams | Numbers | Prior | Excluded | Train frac | Seeds | Accuracy | Macro-F1 | Top-3 acc. | Value on wrong rows | Predicted as ch. 01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 40000 | yes | no | balanced | - | 1.0 | 5 | 0.4504 ± 0.0231 | 0.3999 ± 0.0134 | 0.6327 ± 0.0443 | 39.2% ± 5.3 | 481 ± 180 |
| naive_bayes | 40000 | yes | no | balanced | - | 1.0 | 5 | 0.5500 ± 0.0031 | 0.4834 ± 0.0010 | 0.7502 ± 0.0005 | 26.7% ± 0.6 | 273 ± 7 |
| naive_bayes | 40000 | yes | no | real (strength 1, 2) | - | 1.0 | 5 | 0.6210 ± 0.0018 | 0.5876 ± 0.0098 | 0.8266 ± 0.0026 | 20.2% ± 0.8 | 6 ± 4 |
| perceptron | 40000 | yes | no | balanced | - | 1.0 | 5 | 0.5480 ± 0.0030 | 0.4813 ± 0.0030 | 0.7323 ± 0.0025 | 26.8% ± 1.4 | 225 ± 11 |
| perceptron | 40000 | yes | no | real (strength 0.25, 0.5) | - | 1.0 | 5 | 0.6275 ± 0.0042 | 0.5951 ± 0.0111 | 0.8245 ± 0.0029 | 20.5% ± 1.0 | 4 ± 2 |

## Feature sweep (validation macro-F1, seed 170)

| Vocab | Bigrams | Numbers | knn | naive_bayes | perceptron |
|---|---|---|---|---|---|
| 2000 | no | no | 0.5745 (best 30) | 0.6112 (best 0.5) | 0.6267 (best 10) |
| 2000 | yes | no | 0.5490 (best 5) | 0.5910 (best 0.1) | 0.6148 (best 5) |
| 5000 | no | no | 0.6256 (best 30) | 0.6616 (best 0.5) | 0.6825 (best 3) |
| 5000 | yes | no | 0.5912 (best 50) | 0.6390 (best 0.5) | 0.6643 (best 5) |
| 10000 | no | no | 0.6387 (best 5) | 0.6839 (best 0.5) | 0.7006 (best 5) |
| 10000 | yes | no | 0.6241 (best 50) | 0.6661 (best 0.5) | 0.6911 (best 5) |
| 20000 | no | no | 0.6415 (best 15) | 0.6875 (best 0.1) | 0.7068 (best 5) |
| 20000 | yes | no | 0.6412 (best 15) | 0.6867 (best 0.1) | 0.7084 (best 5) |
| 40000 | no | no | 0.6415 (best 15) | 0.6875 (best 0.1) | 0.7068 (best 5) |
| 40000 | yes | no | 0.6536 (best 15) | 0.6957 (best 0.1) | 0.7157 (best 5) |
| 20000 | no | yes |  | 0.6846 (best 0.1) | 0.7057 (best 3) |
| 20000 | yes | yes |  | 0.6772 (best 0.1) | 0.7075 (best 5) |
| 40000 | no | yes |  | 0.6846 (best 0.1) | 0.7057 (best 3) |
| 40000 | yes | yes |  | 0.6898 (best 0.1) | 0.7183 (best 3) |
