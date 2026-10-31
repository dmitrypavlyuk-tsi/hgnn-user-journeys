# GA4 stage4: seed-aware comparison

Two-level bootstrap: 1000 resamples of 49,093 test sessions and, per model, of its seeds. Welch t-test on per-seed test Macro-F1.

| Model | Seeds | Macro-F1 mean ± s.d. | Per-seed | 95% CI (two-level) |
|---|---|---|---|---|
| HomoGraphSAGE_3layer_meanpool | 3 | 0.6589 ± 0.0028 | 0.656, 0.659, 0.662 | [0.6501, 0.6684] |
| HGNN_full_3layer_meanpool | 3 | 0.6583 ± 0.0050 | 0.653, 0.658, 0.663 | [0.6487, 0.6675] |
| HomoGraphSAGE_meanpool | 3 | 0.6563 ± 0.0068 | 0.650, 0.664, 0.655 | [0.6456, 0.6668] |
| GRU+Time+SF+PageItem | 3 | 0.6555 ± 0.0069 | 0.651, 0.663, 0.652 | [0.6450, 0.6665] |
| HGNN_full_meanpool | 3 | 0.6531 ± 0.0044 | 0.648, 0.654, 0.657 | [0.6435, 0.6626] |
| SASRec+Time+SF+PageItem | 3 | 0.6529 ± 0.0039 | 0.648, 0.655, 0.655 | [0.6441, 0.6619] |
| HGNN_full_attnpool | 3 | 0.6521 ± 0.0105 | 0.640, 0.658, 0.658 | [0.6386, 0.6645] |
| SASRec+Time | 3 | 0.6487 ± 0.0040 | 0.653, 0.646, 0.647 | [0.6408, 0.6568] |
| LightGBM | 3 | 0.6484 ± 0.0008 | 0.649, 0.648, 0.648 | [0.6403, 0.6559] |
| HGNN_full | 3 | 0.6417 ± 0.0053 | 0.636, 0.642, 0.647 | [0.6325, 0.6512] |
| HGNN_full_exp3join | 3 | 0.6404 ± 0.0052 | 0.635, 0.640, 0.646 | [0.6320, 0.6492] |
| HomoGraphSAGE+PageItemFeat | 3 | 0.6396 ± 0.0087 | 0.630, 0.647, 0.642 | [0.6284, 0.6510] |
| HGNN_no_page_item | 3 | 0.6382 ± 0.0095 | 0.627, 0.643, 0.644 | [0.6258, 0.6484] |
| SASRec+Time+SF | 3 | 0.6371 ± 0.0168 | 0.621, 0.655, 0.635 | [0.6206, 0.6550] |
| HGNN_no_page_item_no_SF | 3 | 0.5683 ± 0.0071 | 0.564, 0.565, 0.577 | [0.5599, 0.5768] |
| Markov-3 | 1 | 0.5531 ± 0.0000 | 0.553 | [0.5487, 0.5582] |

## Δ vs HGNN_full

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HomoGraphSAGE_3layer_meanpool | +0.0172 | [+0.0086, +0.0258] | 0.000 | 0.015 |
| HGNN_full_3layer_meanpool | +0.0166 | [+0.0070, +0.0255] | 0.000 | 0.017 |
| HomoGraphSAGE_meanpool | +0.0145 | [+0.0051, +0.0245] | 0.004 | 0.047 |
| GRU+Time+SF+PageItem | +0.0138 | [+0.0038, +0.0240] | 0.002 | 0.055 |
| HGNN_full_meanpool | +0.0114 | [+0.0025, +0.0201] | 0.022 | 0.048 |
| SASRec+Time+SF+PageItem | +0.0112 | [+0.0029, +0.0198] | 0.012 | 0.046 |
| HGNN_full_attnpool | +0.0104 | [-0.0024, +0.0225] | 0.106 | 0.226 |
| SASRec+Time | +0.0070 | [-0.0014, +0.0155] | 0.116 | 0.147 |
| LightGBM | +0.0066 | [-0.0010, +0.0141] | 0.082 | 0.160 |
| HGNN_full_exp3join | -0.0013 | [-0.0093, +0.0059] | 0.734 | 0.779 |
| HomoGraphSAGE+PageItemFeat | -0.0021 | [-0.0126, +0.0075] | 0.704 | 0.744 |
| HGNN_no_page_item | -0.0036 | [-0.0150, +0.0065] | 0.556 | 0.610 |
| SASRec+Time+SF | -0.0046 | [-0.0213, +0.0135] | 0.682 | 0.688 |
| HGNN_no_page_item_no_SF | -0.0734 | [-0.0837, -0.0625] | 0.000 | 0.000 |
| Markov-3 | -0.0886 | [-0.0986, -0.0793] | 0.000 | – |

## Δ vs GRU+Time+SF+PageItem

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HomoGraphSAGE_3layer_meanpool | +0.0034 | [-0.0055, +0.0116] | 0.414 | 0.488 |
| HGNN_full_3layer_meanpool | +0.0028 | [-0.0066, +0.0115] | 0.532 | 0.603 |
| HomoGraphSAGE_meanpool | +0.0008 | [-0.0103, +0.0105] | 0.810 | 0.897 |
| HGNN_full_meanpool | -0.0024 | [-0.0119, +0.0062] | 0.608 | 0.639 |
| SASRec+Time+SF+PageItem | -0.0026 | [-0.0118, +0.0059] | 0.598 | 0.612 |
| HGNN_full_attnpool | -0.0034 | [-0.0172, +0.0092] | 0.626 | 0.668 |
| SASRec+Time | -0.0068 | [-0.0174, +0.0043] | 0.230 | 0.231 |
| LightGBM | -0.0071 | [-0.0167, +0.0020] | 0.130 | 0.213 |
| HGNN_full | -0.0138 | [-0.0240, -0.0038] | 0.002 | 0.055 |
| HGNN_full_exp3join | -0.0151 | [-0.0251, -0.0049] | 0.004 | 0.043 |
| HomoGraphSAGE+PageItemFeat | -0.0159 | [-0.0280, -0.0039] | 0.008 | 0.072 |
| HGNN_no_page_item | -0.0173 | [-0.0310, -0.0054] | 0.000 | 0.069 |
| SASRec+Time+SF | -0.0184 | [-0.0360, +0.0006] | 0.056 | 0.189 |
| HGNN_no_page_item_no_SF | -0.0872 | [-0.1001, -0.0746] | 0.000 | 0.000 |
| Markov-3 | -0.1023 | [-0.1132, -0.0919] | 0.000 | – |

## Δ vs SASRec+Time+SF+PageItem

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HomoGraphSAGE_3layer_meanpool | +0.0060 | [-0.0003, +0.0130] | 0.066 | 0.101 |
| HGNN_full_3layer_meanpool | +0.0053 | [-0.0028, +0.0130] | 0.200 | 0.220 |
| HomoGraphSAGE_meanpool | +0.0033 | [-0.0051, +0.0116] | 0.452 | 0.513 |
| GRU+Time+SF+PageItem | +0.0026 | [-0.0059, +0.0118] | 0.598 | 0.612 |
| HGNN_full_meanpool | +0.0001 | [-0.0073, +0.0073] | 0.966 | 0.970 |
| HGNN_full_attnpool | -0.0008 | [-0.0132, +0.0098] | 0.912 | 0.907 |
| SASRec+Time | -0.0042 | [-0.0130, +0.0046] | 0.374 | 0.260 |
| LightGBM | -0.0046 | [-0.0123, +0.0026] | 0.230 | 0.172 |
| HGNN_full | -0.0112 | [-0.0198, -0.0029] | 0.012 | 0.046 |
| HGNN_full_exp3join | -0.0125 | [-0.0209, -0.0046] | 0.006 | 0.033 |
| HomoGraphSAGE+PageItemFeat | -0.0133 | [-0.0233, -0.0025] | 0.020 | 0.102 |
| HGNN_no_page_item | -0.0148 | [-0.0270, -0.0045] | 0.002 | 0.100 |
| SASRec+Time+SF | -0.0158 | [-0.0320, +0.0021] | 0.088 | 0.240 |
| HGNN_no_page_item_no_SF | -0.0846 | [-0.0956, -0.0739] | 0.000 | 0.000 |
| Markov-3 | -0.0998 | [-0.1090, -0.0908] | 0.000 | – |

## Δ vs HGNN_full_3layer_meanpool

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HomoGraphSAGE_3layer_meanpool | +0.0006 | [-0.0064, +0.0080] | 0.806 | 0.856 |
| HomoGraphSAGE_meanpool | -0.0020 | [-0.0109, +0.0074] | 0.670 | 0.703 |
| GRU+Time+SF+PageItem | -0.0028 | [-0.0115, +0.0066] | 0.532 | 0.603 |
| HGNN_full_meanpool | -0.0052 | [-0.0128, +0.0021] | 0.154 | 0.247 |
| SASRec+Time+SF+PageItem | -0.0053 | [-0.0130, +0.0028] | 0.200 | 0.220 |
| HGNN_full_attnpool | -0.0062 | [-0.0191, +0.0044] | 0.328 | 0.429 |
| SASRec+Time | -0.0095 | [-0.0185, +0.0004] | 0.054 | 0.063 |
| LightGBM | -0.0099 | [-0.0182, -0.0015] | 0.032 | 0.071 |
| HGNN_full | -0.0166 | [-0.0255, -0.0070] | 0.000 | 0.017 |
| HGNN_full_exp3join | -0.0179 | [-0.0269, -0.0080] | 0.000 | 0.013 |
| HomoGraphSAGE+PageItemFeat | -0.0187 | [-0.0297, -0.0073] | 0.000 | 0.045 |
| HGNN_no_page_item | -0.0201 | [-0.0332, -0.0094] | 0.000 | 0.047 |
| SASRec+Time+SF | -0.0212 | [-0.0383, -0.0023] | 0.024 | 0.152 |
| HGNN_no_page_item_no_SF | -0.0900 | [-0.1011, -0.0783] | 0.000 | 0.000 |
| Markov-3 | -0.1051 | [-0.1140, -0.0951] | 0.000 | – |

## Δ vs HomoGraphSAGE_3layer_meanpool

| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |
|---|---|---|---|---|
| HGNN_full_3layer_meanpool | -0.0006 | [-0.0080, +0.0064] | 0.806 | 0.856 |
| HomoGraphSAGE_meanpool | -0.0027 | [-0.0106, +0.0054] | 0.476 | 0.581 |
| GRU+Time+SF+PageItem | -0.0034 | [-0.0116, +0.0055] | 0.414 | 0.488 |
| HGNN_full_meanpool | -0.0059 | [-0.0134, +0.0006] | 0.078 | 0.135 |
| SASRec+Time+SF+PageItem | -0.0060 | [-0.0130, +0.0003] | 0.066 | 0.101 |
| HGNN_full_attnpool | -0.0068 | [-0.0190, +0.0032] | 0.248 | 0.379 |
| SASRec+Time | -0.0102 | [-0.0189, -0.0017] | 0.020 | 0.027 |
| LightGBM | -0.0106 | [-0.0179, -0.0034] | 0.002 | 0.016 |
| HGNN_full | -0.0172 | [-0.0258, -0.0086] | 0.000 | 0.015 |
| HGNN_full_exp3join | -0.0185 | [-0.0268, -0.0103] | 0.000 | 0.012 |
| HomoGraphSAGE+PageItemFeat | -0.0193 | [-0.0302, -0.0091] | 0.002 | 0.051 |
| HGNN_no_page_item | -0.0208 | [-0.0332, -0.0110] | 0.000 | 0.054 |
| SASRec+Time+SF | -0.0218 | [-0.0386, -0.0035] | 0.014 | 0.150 |
| HGNN_no_page_item_no_SF | -0.0906 | [-0.1017, -0.0803] | 0.000 | 0.001 |
| Markov-3 | -0.1058 | [-0.1149, -0.0971] | 0.000 | – |
