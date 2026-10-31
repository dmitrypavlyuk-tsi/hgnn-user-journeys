# Combined GA4 + Coveo: seed-aware results

Stage 5 = learning rate chosen on validation (primary); stage 4 = fixed lr 1e-3. CI = two-level bootstrap (test sessions and seeds); Welch p = seed-level t-test.

## Test Macro-F1 (mean ± s.d. over seeds), stage 5

| Model | GA4 | Coveo |
|---|---|---|
| SASRec+Time+SF+PageItem | 0.665 ± 0.004 (n=3) | 0.645 ± 0.007 (n=5) |
| HGNN_full_3layer_meanpool | 0.667 ± 0.006 (n=3) | 0.626 ± 0.014 (n=5) |
| HomoGraphSAGE_3layer_meanpool | 0.661 ± 0.006 (n=3) | 0.584 ± 0.047 (n=5) |
| GRU+Time+SF+PageItem | 0.663 ± 0.004 (n=3) | 0.496 ± 0.060 (n=5) |
| LightGBM | 0.648 ± 0.001 (n=3) | 0.581 ± 0.001 (n=3) |
| HGNN_full | 0.649 ± 0.005 (n=3) | 0.547 ± 0.023 (n=5) |
| Markov-3 | 0.553 ± 0.000 (n=1) | 0.529 ± 0.000 (n=1) |

## Key contrasts

| Contrast | Dataset | Stage | Pair | Δ | 95% CI (two-level) | Welch p | Verdict |
|---|---|---|---|---|---|---|---|
| Heterogeneity (pooled readout) | GA4 | stage5 | HGNN_full_3layer_meanpool − HomoGraphSAGE_3layer_meanpool | +0.0063 | [-0.0023, +0.0152] | 0.248 | n.s. |
| Best sequence − best graph | GA4 | stage5 | SASRec+Time+SF+PageItem − HGNN_full_3layer_meanpool | -0.0021 | [-0.0098, +0.0058] | 0.629 | n.s. |
| Original HGNN − pooled HGNN | GA4 | stage5 | HGNN_full − HGNN_full_3layer_meanpool | -0.0184 | [-0.0274, -0.0098] | 0.013 | sig. <0 |
| LightGBM − best model | GA4 | stage5 | LightGBM − HGNN_full_3layer_meanpool | -0.0187 | [-0.0269, -0.0098] | 0.027 | sig. <0 |
| GRU − SASRec (same inputs) | GA4 | stage5 | GRU+Time+SF+PageItem − SASRec+Time+SF+PageItem | -0.0015 | [-0.0084, +0.0049] | 0.669 | n.s. |
| Heterogeneity (pooled readout) | GA4 | stage4 | HGNN_full_3layer_meanpool − HomoGraphSAGE_3layer_meanpool | -0.0006 | [-0.0080, +0.0064] | 0.856 | n.s. |
| Best sequence − best graph | GA4 | stage4 | GRU+Time+SF+PageItem − HomoGraphSAGE_3layer_meanpool | -0.0034 | [-0.0116, +0.0055] | 0.488 | n.s. |
| Original HGNN − pooled HGNN | GA4 | stage4 | HGNN_full − HGNN_full_3layer_meanpool | -0.0166 | [-0.0255, -0.0070] | 0.017 | sig. <0 |
| LightGBM − best model | GA4 | stage4 | LightGBM − HomoGraphSAGE_3layer_meanpool | -0.0106 | [-0.0179, -0.0034] | 0.016 | sig. <0 |
| GRU − SASRec (same inputs) | GA4 | stage4 | GRU+Time+SF+PageItem − SASRec+Time+SF+PageItem | +0.0026 | [-0.0059, +0.0118] | 0.612 | n.s. |
| Heterogeneity (pooled readout) | Coveo | stage5 | HGNN_full_3layer_meanpool − HomoGraphSAGE_3layer_meanpool | +0.0423 | [+0.0081, +0.0835] | 0.116 | sig. >0 |
| Best sequence − best graph | Coveo | stage5 | SASRec+Time+SF+PageItem − HGNN_full_3layer_meanpool | +0.0188 | [+0.0073, +0.0316] | 0.037 | sig. >0 |
| Original HGNN − pooled HGNN | Coveo | stage5 | HGNN_full − HGNN_full_3layer_meanpool | -0.0795 | [-0.1012, -0.0590] | 0.000 | sig. <0 |
| LightGBM − best model | Coveo | stage5 | LightGBM − SASRec+Time+SF+PageItem | -0.0640 | [-0.0725, -0.0556] | 0.000 | sig. <0 |
| GRU − SASRec (same inputs) | Coveo | stage5 | GRU+Time+SF+PageItem − SASRec+Time+SF+PageItem | -0.1486 | [-0.1895, -0.0957] | 0.005 | sig. <0 |
| Heterogeneity (pooled readout) | Coveo | stage4 | HGNN_full_3layer_meanpool − HomoGraphSAGE_3layer_meanpool | +0.0607 | [+0.0230, +0.0905] | 0.085 | sig. >0 |
| Best sequence − best graph | Coveo | stage4 | SASRec+Time+SF+PageItem − HGNN_full_3layer_meanpool | +0.0074 | [-0.0059, +0.0214] | 0.464 | n.s. |
| Original HGNN − pooled HGNN | Coveo | stage4 | HGNN_full − HGNN_full_3layer_meanpool | -0.0315 | [-0.0630, -0.0026] | 0.209 | sig. <0 |
| LightGBM − best model | Coveo | stage4 | LightGBM − SASRec+Time+SF+PageItem | -0.0094 | [-0.0173, -0.0022] | 0.007 | sig. <0 |
| GRU − SASRec (same inputs) | Coveo | stage4 | GRU+Time+SF+PageItem − SASRec+Time+SF+PageItem | -0.1103 | [-0.1611, -0.0798] | 0.055 | sig. <0 |
