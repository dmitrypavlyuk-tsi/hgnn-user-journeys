# Coveo stage5: seed-aware comparison

Two-level bootstrap: 1000 resamples of 42,727 test sessions and, per model, of its seeds. Welch t-test on per-seed test Macro-F1.

| Model | Seeds | Macro-F1 mean ± s.d. | Per-seed | 95% CI (two-level) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | 5 | 0.6448 ± 0.0066 | 0.634, 0.649, 0.648, 0.650, 0.643 | [0.6349, 0.6532] |
| HGNN_full_3layer_meanpool | 5 | 0.6260 ± 0.0139 | 0.635, 0.625, 0.603, 0.639, 0.628 | [0.6129, 0.6382] |
| HomoGraphSAGE_3layer_meanpool | 5 | 0.5837 ± 0.0472 | 0.617, 0.513, 0.575, 0.636, 0.577 | [0.5412, 0.6174] |
| LightGBM | 3 | 0.5807 ± 0.0009 | 0.580, 0.582, 0.580 | [0.5735, 0.5876] |
| HGNN_full | 5 | 0.5465 ± 0.0226 | 0.517, 0.561, 0.575, 0.541, 0.538 | [0.5287, 0.5634] |
| Markov-3 | 1 | 0.5286 ± 0.0000 | 0.529 | [0.5236, 0.5332] |
| GRU+Time+SF+PageItem | 5 | 0.4962 ± 0.0598 | 0.470, 0.475, 0.501, 0.439, 0.596 | [0.4554, 0.5498] |

## Δ vs HGNN_full

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | +0.0982 | [+0.0791, +0.1174] | 0.000 | 0.000 |
| HGNN_full_3layer_meanpool | +0.0795 | [+0.0590, +0.1012] | 0.000 | 0.000 |
| HomoGraphSAGE_3layer_meanpool | +0.0372 | [-0.0084, +0.0733] | 0.102 | 0.166 |
| LightGBM | +0.0342 | [+0.0161, +0.0526] | 0.000 | 0.027 |
| Markov-3 | -0.0179 | [-0.0354, +0.0007] | 0.054 | – |
| GRU+Time+SF+PageItem | -0.0504 | [-0.0965, +0.0049] | 0.086 | 0.137 |

## Δ vs GRU+Time+SF+PageItem

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | +0.1486 | [+0.0957, +0.1895] | 0.000 | 0.005 |
| HGNN_full_3layer_meanpool | +0.1298 | [+0.0788, +0.1731] | 0.000 | 0.007 |
| HomoGraphSAGE_3layer_meanpool | +0.0875 | [+0.0233, +0.1424] | 0.004 | 0.035 |
| LightGBM | +0.0846 | [+0.0308, +0.1255] | 0.000 | 0.034 |
| HGNN_full | +0.0504 | [-0.0049, +0.0965] | 0.086 | 0.137 |
| Markov-3 | +0.0324 | [-0.0207, +0.0748] | 0.198 | – |

## Δ vs SASRec+Time+SF+PageItem

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HGNN_full_3layer_meanpool | -0.0188 | [-0.0316, -0.0073] | 0.000 | 0.037 |
| HomoGraphSAGE_3layer_meanpool | -0.0611 | [-0.1024, -0.0274] | 0.000 | 0.044 |
| LightGBM | -0.0640 | [-0.0725, -0.0556] | 0.000 | 0.000 |
| HGNN_full | -0.0982 | [-0.1174, -0.0791] | 0.000 | 0.000 |
| Markov-3 | -0.1162 | [-0.1250, -0.1061] | 0.000 | – |
| GRU+Time+SF+PageItem | -0.1486 | [-0.1895, -0.0957] | 0.000 | 0.005 |

## Δ vs HGNN_full_3layer_meanpool

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | +0.0188 | [+0.0073, +0.0316] | 0.000 | 0.037 |
| HomoGraphSAGE_3layer_meanpool | -0.0423 | [-0.0835, -0.0081] | 0.018 | 0.116 |
| LightGBM | -0.0453 | [-0.0571, -0.0318] | 0.000 | 0.002 |
| HGNN_full | -0.0795 | [-0.1012, -0.0590] | 0.000 | 0.000 |
| Markov-3 | -0.0974 | [-0.1099, -0.0833] | 0.000 | – |
| GRU+Time+SF+PageItem | -0.1298 | [-0.1731, -0.0788] | 0.000 | 0.007 |

## Δ vs HomoGraphSAGE_3layer_meanpool

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | +0.0611 | [+0.0274, +0.1024] | 0.000 | 0.044 |
| HGNN_full_3layer_meanpool | +0.0423 | [+0.0081, +0.0835] | 0.018 | 0.116 |
| LightGBM | -0.0029 | [-0.0360, +0.0375] | 0.828 | 0.896 |
| HGNN_full | -0.0372 | [-0.0733, +0.0084] | 0.102 | 0.166 |
| Markov-3 | -0.0551 | [-0.0891, -0.0133] | 0.002 | – |
| GRU+Time+SF+PageItem | -0.0875 | [-0.1424, -0.0233] | 0.004 | 0.035 |
