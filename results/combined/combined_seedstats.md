# Combined GA4 + Coveo: seed-aware results

Stage 5 = learning rate chosen on validation (primary); stage 4 = fixed lr 1e-3. CI = two-level bootstrap (test sessions and seeds); Welch p = seed-level t-test.

## Test Macro-F1 (mean ± s.d. over seeds), stage 5

| Model | GA4 | Coveo |
|---|---|---|
| SASRec+Time+SF+PageItem | 0.665 ± 0.003 (n=5) | 0.645 ± 0.007 (n=5) |
| HGNN_full_3layer_meanpool | 0.668 ± 0.005 (n=5) | 0.626 ± 0.014 (n=5) |
| HomoGraphSAGE_3layer_meanpool | 0.662 ± 0.005 (n=5) | 0.584 ± 0.047 (n=5) |
| GRU+Time+SF+PageItem | 0.664 ± 0.003 (n=5) | 0.496 ± 0.060 (n=5) |
| LightGBM | 0.648 ± 0.001 (n=5) | 0.581 ± 0.001 (n=3) |
| HGNN_full | 0.650 ± 0.004 (n=5) | 0.547 ± 0.023 (n=5) |
| Markov-3 | 0.553 ± 0.000 (n=1) | 0.529 ± 0.000 (n=1) |

## Key contrasts

| Contrast | Dataset | Stage | Pair | Δ | 95% CI (two-level) | Welch p | Verdict |
|---|---|---|---|---|---|---|---|
| Heterogeneity (pooled readout) | GA4 | stage5 | HGNN_full_3layer_meanpool − HomoGraphSAGE_3layer_meanpool | +0.0056 | [-0.0011, +0.0127] | 0.115 | n.s. |
| Best sequence − best graph | GA4 | stage5 | SASRec+Time+SF+PageItem − HGNN_full_3layer_meanpool | -0.0031 | [-0.0087, +0.0030] | 0.247 | n.s. |
| Original HGNN − pooled HGNN | GA4 | stage5 | HGNN_full − HGNN_full_3layer_meanpool | -0.0180 | [-0.0250, -0.0111] | 0.000 | sig. <0 |
| LightGBM − best model | GA4 | stage5 | LightGBM − HGNN_full_3layer_meanpool | -0.0194 | [-0.0264, -0.0119] | 0.001 | sig. <0 |
| GRU − SASRec (same inputs) | GA4 | stage5 | GRU+Time+SF+PageItem − SASRec+Time+SF+PageItem | -0.0010 | [-0.0062, +0.0041] | 0.625 | n.s. |
| Heterogeneity (pooled readout) | GA4 | stage4 | HGNN_full_3layer_meanpool − HomoGraphSAGE_3layer_meanpool | -0.0004 | [-0.0069, +0.0059] | 0.891 | n.s. |
| Best sequence − best graph | GA4 | stage4 | GRU+Time+SF+PageItem − HomoGraphSAGE_3layer_meanpool | -0.0018 | [-0.0093, +0.0052] | 0.612 | n.s. |
| Original HGNN − pooled HGNN | GA4 | stage4 | HGNN_full − HGNN_full_3layer_meanpool | -0.0174 | [-0.0253, -0.0100] | 0.001 | sig. <0 |
| LightGBM − best model | GA4 | stage4 | LightGBM − HomoGraphSAGE_3layer_meanpool | -0.0109 | [-0.0183, -0.0040] | 0.005 | sig. <0 |
| GRU − SASRec (same inputs) | GA4 | stage4 | GRU+Time+SF+PageItem − SASRec+Time+SF+PageItem | +0.0026 | [-0.0047, +0.0097] | 0.488 | n.s. |
| Heterogeneity (pooled readout) | Coveo | stage5 | HGNN_full_3layer_meanpool − HomoGraphSAGE_3layer_meanpool | +0.0423 | [+0.0057, +0.0811] | 0.116 | sig. >0 |
| Best sequence − best graph | Coveo | stage5 | SASRec+Time+SF+PageItem − HGNN_full_3layer_meanpool | +0.0188 | [+0.0072, +0.0327] | 0.037 | sig. >0 |
| Original HGNN − pooled HGNN | Coveo | stage5 | HGNN_full − HGNN_full_3layer_meanpool | -0.0795 | [-0.1008, -0.0571] | 0.000 | sig. <0 |
| LightGBM − best model | Coveo | stage5 | LightGBM − SASRec+Time+SF+PageItem | -0.0640 | [-0.0725, -0.0550] | 0.000 | sig. <0 |
| GRU − SASRec (same inputs) | Coveo | stage5 | GRU+Time+SF+PageItem − SASRec+Time+SF+PageItem | -0.1486 | [-0.1871, -0.0978] | 0.005 | sig. <0 |
| Heterogeneity (pooled readout) | Coveo | stage4 | HGNN_full_3layer_meanpool − HomoGraphSAGE_3layer_meanpool | +0.0453 | [+0.0105, +0.0736] | 0.035 | sig. >0 |
| Best sequence − best graph | Coveo | stage4 | SASRec+Time − HGNN_no_page_item | +0.0427 | [+0.0158, +0.0696] | 0.047 | sig. >0 |
| Original HGNN − pooled HGNN | Coveo | stage4 | HGNN_full − HGNN_full_3layer_meanpool | -0.0273 | [-0.0503, -0.0024] | 0.087 | sig. <0 |
| LightGBM − best model | Coveo | stage4 | LightGBM − SASRec+Time | -0.0562 | [-0.0640, -0.0491] | 0.000 | sig. <0 |
| GRU − SASRec (same inputs) | Coveo | stage4 | GRU+Time+SF+PageItem − SASRec+Time+SF+PageItem | -0.1053 | [-0.1464, -0.0667] | 0.011 | sig. <0 |
