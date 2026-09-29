# SemanticDiff evaluation

Pairs: 388 (dev 192, test 196)

| source               |   dev |   test |
|:---------------------|------:|-------:|
| synthetic:compound   |    20 |     20 |
| synthetic:condition  |    18 |     18 |
| synthetic:entity     |    11 |      8 |
| synthetic:modality   |    18 |     18 |
| synthetic:negation   |    18 |     18 |
| synthetic:numeric    |    18 |     16 |
| synthetic:paraphrase |    45 |     45 |
| synthetic:reversal   |     8 |     17 |
| synthetic:scope      |    18 |     18 |
| synthetic:temporal   |    18 |     18 |

Baseline thresholds (tuned on dev to maximise macro-F1):

`{"lexical": 0.032257064516129, "tfidf": 0.1603557120032716, "embedding": 0.05612891912460327, "embed+nli": {"cos": 0.0, "entail": 0.05}}`

## Task A — did the meaning change? (binary)

| split   | system                  |   accuracy |   precision |   recall |    f1 |   macro_f1 |
|:--------|:------------------------|-----------:|------------:|---------:|------:|-----------:|
| test    | lexical                 |      0.770 |       0.770 |    1.000 | 0.870 |      0.435 |
| test    | tfidf                   |      0.531 |       0.722 |    0.636 | 0.676 |      0.412 |
| test    | embedding               |      0.709 |       0.841 |    0.768 | 0.803 |      0.625 |
| test    | embed+nli               |      0.816 |       0.832 |    0.954 | 0.889 |      0.680 |
| test    | semanticdiff            |      0.985 |       0.987 |    0.993 | 0.990 |      0.978 |
| test    | semanticdiff −NLI       |      0.980 |       0.987 |    0.987 | 0.987 |      0.971 |
| test    | semanticdiff −unit-norm |      0.954 |       0.949 |    0.993 | 0.971 |      0.931 |

### Accuracy per source category (test) — paraphrase = specificity

| category   |   n |   lexical |   tfidf |   embedding |   embed+nli |   semanticdiff |   semanticdiff −NLI |   semanticdiff −unit-norm |
|:-----------|----:|----------:|--------:|------------:|------------:|---------------:|--------------------:|--------------------------:|
| compound   |  20 |     1.000 |   0.950 |       1.000 |       1.000 |          1.000 |               1.000 |                     1.000 |
| condition  |  18 |     1.000 |   1.000 |       0.833 |       0.944 |          1.000 |               1.000 |                     1.000 |
| entity     |   8 |     1.000 |   1.000 |       0.875 |       1.000 |          1.000 |               1.000 |                     1.000 |
| modality   |  18 |     1.000 |   0.722 |       0.111 |       0.778 |          1.000 |               1.000 |                     1.000 |
| negation   |  18 |     1.000 |   0.111 |       1.000 |       1.000 |          1.000 |               1.000 |                     1.000 |
| numeric    |  16 |     1.000 |   0.688 |       0.875 |       1.000 |          1.000 |               1.000 |                     1.000 |
| paraphrase |  45 |     0.000 |   0.178 |       0.511 |       0.356 |          0.956 |               0.956 |                     0.822 |
| reversal   |  17 |     1.000 |   0.882 |       0.941 |       0.941 |          0.941 |               0.882 |                     0.941 |
| scope      |  18 |     1.000 |   0.056 |       0.889 |       0.944 |          1.000 |               1.000 |                     1.000 |
| temporal   |  18 |     1.000 |   0.500 |       0.444 |       1.000 |          1.000 |               1.000 |                     1.000 |

### Danger zones (test)

| system                  |   recall | small edit, meaning changed (n=104) |   specificity | big edit, meaning kept (n=17) |
|:------------------------|-----------------------------------------------:|----------------------------------------------:|
| lexical                 |                                          1.000 |                                         0.000 |
| tfidf                   |                                          0.490 |                                         0.000 |
| embedding               |                                          0.731 |                                         0.176 |
| embed+nli               |                                          0.942 |                                         0.412 |
| semanticdiff            |                                          0.990 |                                         0.882 |
| semanticdiff −NLI       |                                          0.981 |                                         0.882 |
| semanticdiff −unit-norm |                                          0.990 |                                         0.765 |

### McNemar exact test (test): SemanticDiff vs baselines

| comparison                |   p_value |
|:--------------------------|----------:|
| semanticdiff vs lexical   |     0.000 |
| semanticdiff vs tfidf     |     0.000 |
| semanticdiff vs embedding |     0.000 |
| semanticdiff vs embed+nli |     0.000 |

## Task B — what changed? (multi-label, 8 categories)

| split   | system                  |   micro_f1 |   macro_f1 |   exact_match |   hamming_loss |
|:--------|:------------------------|-----------:|-----------:|--------------:|---------------:|
| test    | keyword-diff            |      0.644 |      0.681 |         0.480 |          0.114 |
| test    | semanticdiff            |      0.994 |      0.993 |         0.990 |          0.001 |
| test    | semanticdiff −NLI       |      0.976 |      0.969 |         0.959 |          0.005 |
| test    | semanticdiff −unit-norm |      0.977 |      0.978 |         0.959 |          0.005 |

### Per-category F1 (test)

| category   |   keyword-diff |   semanticdiff |   semanticdiff −NLI |   semanticdiff −unit-norm |
|:-----------|---------------:|---------------:|--------------------:|--------------------------:|
| CONDITION  |          0.538 |          1.000 |               1.000 |                     1.000 |
| ENTITY     |          0.500 |          1.000 |               1.000 |                     1.000 |
| MODALITY   |          0.677 |          1.000 |               1.000 |                     1.000 |
| NEGATION   |          1.000 |          1.000 |               1.000 |                     1.000 |
| NUMERIC    |          0.821 |          1.000 |               1.000 |                     1.000 |
| REVERSAL   |          0.889 |          0.947 |               0.750 |                     0.947 |
| SCOPE      |          0.449 |          1.000 |               1.000 |                     1.000 |
| TEMPORAL   |          0.575 |          1.000 |               1.000 |                     0.880 |

### SemanticDiff confusion matrix (single-label test pairs; rows = gold)

|           |   NUMERIC |   TEMPORAL |   ENTITY |   MODALITY |   NEGATION |   SCOPE |   CONDITION |   REVERSAL |   NONE |   MULTI |
|:----------|----------:|-----------:|---------:|-----------:|-----------:|--------:|------------:|-----------:|-------:|--------:|
| NUMERIC   |        16 |          0 |        0 |          0 |          0 |       0 |           0 |          0 |      0 |       0 |
| TEMPORAL  |         0 |         18 |        0 |          0 |          0 |       0 |           0 |          0 |      0 |       0 |
| ENTITY    |         0 |          0 |        8 |          0 |          0 |       0 |           0 |          0 |      0 |       0 |
| MODALITY  |         0 |          0 |        0 |         18 |          0 |       0 |           0 |          0 |      0 |       0 |
| NEGATION  |         0 |          0 |        0 |          0 |         18 |       0 |           0 |          0 |      0 |       0 |
| SCOPE     |         0 |          0 |        0 |          0 |          0 |      18 |           0 |          0 |      0 |       0 |
| CONDITION |         0 |          0 |        0 |          0 |          0 |       0 |          18 |          0 |      0 |       0 |
| REVERSAL  |         0 |          0 |        0 |          0 |          0 |       0 |           0 |         16 |      1 |       0 |
| NONE      |         0 |          0 |        0 |          0 |          0 |       0 |           0 |          0 |     45 |       0 |
| MULTI     |         0 |          0 |        0 |          0 |          0 |       0 |           0 |          0 |      0 |       0 |
