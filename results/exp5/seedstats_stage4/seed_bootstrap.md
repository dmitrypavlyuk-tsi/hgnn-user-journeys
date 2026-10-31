# Coveo stage4: seed-aware comparison

Two-level bootstrap: 1000 resamples of 42,727 test sessions and, per model, of its seeds. Welch t-test on per-seed test Macro-F1.

| Model | Seeds | Macro-F1 mean ± s.d. | Per-seed | 95% CI (two-level) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | 3 | 0.5902 ± 0.0020 | 0.592, 0.588, 0.590 | [0.5820, 0.5980] |
| HGNN_full_3layer_meanpool | 3 | 0.5828 ± 0.0142 | 0.597, 0.584, 0.568 | [0.5685, 0.5972] |
| LightGBM | 3 | 0.5807 ± 0.0009 | 0.580, 0.582, 0.580 | [0.5731, 0.5880] |
| HGNN_full | 3 | 0.5513 ± 0.0305 | 0.517, 0.561, 0.575 | [0.5181, 0.5752] |
| Markov-3 | 1 | 0.5286 ± 0.0000 | 0.529 | [0.5238, 0.5331] |
| HomoGraphSAGE_3layer_meanpool | 3 | 0.5221 ± 0.0360 | 0.563, 0.496, 0.507 | [0.4948, 0.5624] |
| GRU+Time+SF+PageItem | 3 | 0.4799 ± 0.0468 | 0.508, 0.426, 0.505 | [0.4274, 0.5114] |

## Δ vs HGNN_full

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | +0.0389 | [+0.0149, +0.0722] | 0.000 | 0.157 |
| HGNN_full_3layer_meanpool | +0.0315 | [+0.0026, +0.0630] | 0.026 | 0.209 |
| LightGBM | +0.0295 | [+0.0050, +0.0639] | 0.004 | 0.236 |
| Markov-3 | -0.0227 | [-0.0473, +0.0109] | 0.108 | – |
| HomoGraphSAGE_3layer_meanpool | -0.0292 | [-0.0706, +0.0153] | 0.212 | 0.346 |
| GRU+Time+SF+PageItem | -0.0714 | [-0.1230, -0.0259] | 0.000 | 0.102 |

## Δ vs GRU+Time+SF+PageItem

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | +0.1103 | [+0.0798, +0.1611] | 0.000 | 0.055 |
| HGNN_full_3layer_meanpool | +0.1029 | [+0.0663, +0.1528] | 0.000 | 0.052 |
| LightGBM | +0.1009 | [+0.0692, +0.1523] | 0.000 | 0.065 |
| HGNN_full | +0.0714 | [+0.0259, +0.1230] | 0.000 | 0.102 |
| Markov-3 | +0.0487 | [+0.0157, +0.1005] | 0.000 | – |
| HomoGraphSAGE_3layer_meanpool | +0.0422 | [-0.0078, +0.0956] | 0.172 | 0.287 |

## Δ vs SASRec+Time+SF+PageItem

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HGNN_full_3layer_meanpool | -0.0074 | [-0.0214, +0.0059] | 0.298 | 0.464 |
| LightGBM | -0.0094 | [-0.0173, -0.0022] | 0.006 | 0.007 |
| HGNN_full | -0.0389 | [-0.0722, -0.0149] | 0.000 | 0.157 |
| Markov-3 | -0.0616 | [-0.0701, -0.0524] | 0.000 | – |
| HomoGraphSAGE_3layer_meanpool | -0.0681 | [-0.0947, -0.0285] | 0.000 | 0.082 |
| GRU+Time+SF+PageItem | -0.1103 | [-0.1611, -0.0798] | 0.000 | 0.055 |

## Δ vs HGNN_full_3layer_meanpool

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | +0.0074 | [-0.0059, +0.0214] | 0.298 | 0.464 |
| LightGBM | -0.0020 | [-0.0163, +0.0118] | 0.786 | 0.828 |
| HGNN_full | -0.0315 | [-0.0630, -0.0026] | 0.026 | 0.209 |
| Markov-3 | -0.0542 | [-0.0692, -0.0381] | 0.000 | – |
| HomoGraphSAGE_3layer_meanpool | -0.0607 | [-0.0905, -0.0230] | 0.000 | 0.085 |
| GRU+Time+SF+PageItem | -0.1029 | [-0.1528, -0.0663] | 0.000 | 0.052 |

## Δ vs HomoGraphSAGE_3layer_meanpool

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | +0.0681 | [+0.0285, +0.0947] | 0.000 | 0.082 |
| HGNN_full_3layer_meanpool | +0.0607 | [+0.0230, +0.0905] | 0.000 | 0.085 |
| LightGBM | +0.0587 | [+0.0186, +0.0854] | 0.000 | 0.106 |
| HGNN_full | +0.0292 | [-0.0153, +0.0706] | 0.212 | 0.346 |
| Markov-3 | +0.0065 | [-0.0337, +0.0348] | 0.600 | – |
| GRU+Time+SF+PageItem | -0.0422 | [-0.0956, +0.0078] | 0.172 | 0.287 |
