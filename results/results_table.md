| Method | Features | Best params | Bal. acc | Bal. macro-F1 | Nat. acc | Nat. macro-F1 | Nat. top-3 | Nat. value-wtd error |
|---|---|---|---|---|---|---|---|---|
| naive_bayes | count | {'alpha': 0.5} | 0.663 | 0.663 | 0.507 | 0.429 | 0.717 | 0.318 |
| perceptron | tfidf | {'epochs': 5} | 0.686 | 0.685 | 0.502 | 0.437 | 0.697 | 0.281 |
| knn | tfidf | {'k': 15} | 0.628 | 0.625 | 0.418 | 0.370 | 0.602 | 0.447 |
