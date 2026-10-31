# GA4 stage5: seed-aware comparison

Two-level bootstrap: 1000 resamples of 49,093 test sessions and, per model, of its seeds. Welch t-test on per-seed test Macro-F1.

| Model | Seeds | Macro-F1 mean ± s.d. | Per-seed | 95% CI (two-level) |
|---|---|---|---|---|
| HGNN_full_3layer_meanpool | 3 | 0.6671 ± 0.0056 | 0.670, 0.661, 0.671 | [0.6571, 0.6772] |
| SASRec+Time+SF+PageItem | 3 | 0.6650 ± 0.0040 | 0.661, 0.669, 0.664 | [0.6565, 0.6741] |
| HGNN_full_attnpool | 3 | 0.6639 ± 0.0021 | 0.664, 0.662, 0.666 | [0.6550, 0.6724] |
| SASRec+Time+SF | 3 | 0.6637 ± 0.0009 | 0.664, 0.663, 0.665 | [0.6563, 0.6712] |
| HGNN_full_meanpool | 3 | 0.6636 ± 0.0034 | 0.661, 0.667, 0.662 | [0.6554, 0.6727] |
| GRU+Time+SF+PageItem | 3 | 0.6634 ± 0.0041 | 0.668, 0.661, 0.661 | [0.6544, 0.6726] |
| HomoGraphSAGE_meanpool | 3 | 0.6629 ± 0.0039 | 0.660, 0.667, 0.662 | [0.6539, 0.6724] |
| HomoGraphSAGE_3layer_meanpool | 3 | 0.6608 ± 0.0057 | 0.663, 0.665, 0.654 | [0.6511, 0.6712] |
| HomoGraphSAGE+PageItemFeat | 3 | 0.6516 ± 0.0039 | 0.656, 0.649, 0.650 | [0.6436, 0.6602] |
| SASRec+Time | 3 | 0.6502 ± 0.0039 | 0.654, 0.650, 0.647 | [0.6421, 0.6586] |
| HGNN_no_page_item | 3 | 0.6488 ± 0.0047 | 0.653, 0.650, 0.644 | [0.6403, 0.6567] |
| HGNN_full | 3 | 0.6487 ± 0.0048 | 0.653, 0.644, 0.649 | [0.6397, 0.6577] |
| LightGBM | 3 | 0.6484 ± 0.0008 | 0.649, 0.648, 0.648 | [0.6401, 0.6562] |
| HGNN_no_page_item_no_SF | 3 | 0.5717 ± 0.0060 | 0.575, 0.565, 0.575 | [0.5634, 0.5800] |
| Markov-3 | 1 | 0.5531 ± 0.0000 | 0.553 | [0.5486, 0.5580] |

## Δ vs HGNN_full

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HGNN_full_3layer_meanpool | +0.0184 | [+0.0098, +0.0274] | 0.000 | 0.013 |
| SASRec+Time+SF+PageItem | +0.0163 | [+0.0082, +0.0247] | 0.000 | 0.012 |
| HGNN_full_attnpool | +0.0152 | [+0.0081, +0.0233] | 0.000 | 0.019 |
| SASRec+Time+SF | +0.0150 | [+0.0092, +0.0214] | 0.000 | 0.029 |
| HGNN_full_meanpool | +0.0149 | [+0.0080, +0.0228] | 0.000 | 0.015 |
| GRU+Time+SF+PageItem | +0.0148 | [+0.0063, +0.0239] | 0.002 | 0.017 |
| HomoGraphSAGE_meanpool | +0.0143 | [+0.0068, +0.0234] | 0.000 | 0.018 |
| HomoGraphSAGE_3layer_meanpool | +0.0122 | [+0.0043, +0.0222] | 0.002 | 0.050 |
| HomoGraphSAGE+PageItemFeat | +0.0030 | [-0.0027, +0.0094] | 0.306 | 0.454 |
| SASRec+Time | +0.0016 | [-0.0056, +0.0089] | 0.626 | 0.682 |
| HGNN_no_page_item | +0.0002 | [-0.0069, +0.0072] | 0.948 | 0.966 |
| LightGBM | -0.0003 | [-0.0068, +0.0066] | 0.942 | 0.928 |
| HGNN_no_page_item_no_SF | -0.0770 | [-0.0857, -0.0685] | 0.000 | 0.000 |
| Markov-3 | -0.0955 | [-0.1042, -0.0859] | 0.000 | – |

## Δ vs GRU+Time+SF+PageItem

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HGNN_full_3layer_meanpool | +0.0036 | [-0.0044, +0.0114] | 0.366 | 0.422 |
| SASRec+Time+SF+PageItem | +0.0015 | [-0.0049, +0.0084] | 0.648 | 0.669 |
| HGNN_full_attnpool | +0.0005 | [-0.0057, +0.0068] | 0.886 | 0.871 |
| SASRec+Time+SF | +0.0002 | [-0.0077, +0.0081] | 0.926 | 0.928 |
| HGNN_full_meanpool | +0.0002 | [-0.0066, +0.0070] | 0.960 | 0.963 |
| HomoGraphSAGE_meanpool | -0.0005 | [-0.0071, +0.0063] | 0.882 | 0.883 |
| HomoGraphSAGE_3layer_meanpool | -0.0026 | [-0.0114, +0.0052] | 0.594 | 0.559 |
| HomoGraphSAGE+PageItemFeat | -0.0118 | [-0.0192, -0.0034] | 0.006 | 0.023 |
| SASRec+Time | -0.0132 | [-0.0215, -0.0048] | 0.000 | 0.016 |
| HGNN_no_page_item | -0.0146 | [-0.0232, -0.0061] | 0.000 | 0.016 |
| HGNN_full | -0.0148 | [-0.0239, -0.0063] | 0.002 | 0.017 |
| LightGBM | -0.0151 | [-0.0234, -0.0073] | 0.000 | 0.021 |
| HGNN_no_page_item_no_SF | -0.0918 | [-0.1021, -0.0813] | 0.000 | 0.000 |
| Markov-3 | -0.1103 | [-0.1194, -0.1010] | 0.000 | – |

## Δ vs SASRec+Time+SF+PageItem

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HGNN_full_3layer_meanpool | +0.0021 | [-0.0058, +0.0098] | 0.546 | 0.629 |
| HGNN_full_attnpool | -0.0011 | [-0.0068, +0.0051] | 0.718 | 0.711 |
| SASRec+Time+SF | -0.0013 | [-0.0083, +0.0063] | 0.710 | 0.638 |
| HGNN_full_meanpool | -0.0014 | [-0.0070, +0.0053] | 0.644 | 0.674 |
| GRU+Time+SF+PageItem | -0.0015 | [-0.0084, +0.0049] | 0.648 | 0.669 |
| HomoGraphSAGE_meanpool | -0.0020 | [-0.0079, +0.0044] | 0.524 | 0.559 |
| HomoGraphSAGE_3layer_meanpool | -0.0042 | [-0.0120, +0.0036] | 0.302 | 0.368 |
| HomoGraphSAGE+PageItemFeat | -0.0133 | [-0.0201, -0.0057] | 0.000 | 0.015 |
| SASRec+Time | -0.0147 | [-0.0224, -0.0064] | 0.000 | 0.010 |
| HGNN_no_page_item | -0.0161 | [-0.0242, -0.0078] | 0.000 | 0.011 |
| HGNN_full | -0.0163 | [-0.0247, -0.0082] | 0.000 | 0.012 |
| LightGBM | -0.0166 | [-0.0236, -0.0090] | 0.000 | 0.016 |
| HGNN_no_page_item_no_SF | -0.0933 | [-0.1025, -0.0835] | 0.000 | 0.000 |
| Markov-3 | -0.1118 | [-0.1205, -0.1031] | 0.000 | – |

## Δ vs HGNN_full_3layer_meanpool

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| SASRec+Time+SF+PageItem | -0.0021 | [-0.0098, +0.0058] | 0.546 | 0.629 |
| HGNN_full_attnpool | -0.0032 | [-0.0091, +0.0038] | 0.324 | 0.438 |
| SASRec+Time+SF | -0.0034 | [-0.0112, +0.0061] | 0.412 | 0.405 |
| HGNN_full_meanpool | -0.0035 | [-0.0098, +0.0039] | 0.314 | 0.420 |
| GRU+Time+SF+PageItem | -0.0036 | [-0.0114, +0.0044] | 0.366 | 0.422 |
| HomoGraphSAGE_meanpool | -0.0041 | [-0.0108, +0.0035] | 0.278 | 0.357 |
| HomoGraphSAGE_3layer_meanpool | -0.0063 | [-0.0152, +0.0023] | 0.152 | 0.248 |
| HomoGraphSAGE+PageItemFeat | -0.0154 | [-0.0230, -0.0068] | 0.000 | 0.022 |
| SASRec+Time | -0.0168 | [-0.0254, -0.0071] | 0.000 | 0.017 |
| HGNN_no_page_item | -0.0182 | [-0.0268, -0.0088] | 0.000 | 0.013 |
| HGNN_full | -0.0184 | [-0.0274, -0.0098] | 0.000 | 0.013 |
| LightGBM | -0.0187 | [-0.0269, -0.0098] | 0.000 | 0.027 |
| HGNN_no_page_item_no_SF | -0.0954 | [-0.1055, -0.0841] | 0.000 | 0.000 |
| Markov-3 | -0.1139 | [-0.1234, -0.1040] | 0.000 | – |

## Δ vs HomoGraphSAGE_3layer_meanpool

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HGNN_full_3layer_meanpool | +0.0063 | [-0.0023, +0.0152] | 0.152 | 0.248 |
| SASRec+Time+SF+PageItem | +0.0042 | [-0.0036, +0.0120] | 0.302 | 0.368 |
| HGNN_full_attnpool | +0.0031 | [-0.0042, +0.0111] | 0.430 | 0.455 |
| SASRec+Time+SF | +0.0029 | [-0.0057, +0.0100] | 0.470 | 0.478 |
| HGNN_full_meanpool | +0.0028 | [-0.0045, +0.0101] | 0.446 | 0.520 |
| GRU+Time+SF+PageItem | +0.0026 | [-0.0052, +0.0114] | 0.594 | 0.559 |
| HomoGraphSAGE_meanpool | +0.0021 | [-0.0053, +0.0099] | 0.598 | 0.628 |
| HomoGraphSAGE+PageItemFeat | -0.0092 | [-0.0175, -0.0012] | 0.030 | 0.092 |
| SASRec+Time | -0.0106 | [-0.0200, -0.0023] | 0.014 | 0.065 |
| HGNN_no_page_item | -0.0120 | [-0.0216, -0.0034] | 0.004 | 0.050 |
| HGNN_full | -0.0122 | [-0.0222, -0.0043] | 0.002 | 0.050 |
| LightGBM | -0.0124 | [-0.0207, -0.0048] | 0.004 | 0.061 |
| HGNN_no_page_item_no_SF | -0.0891 | [-0.1003, -0.0791] | 0.000 | 0.000 |
| Markov-3 | -0.1077 | [-0.1176, -0.0981] | 0.000 | – |
