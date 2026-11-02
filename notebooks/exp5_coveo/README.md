# Experiment 5: Controlled comparison on a second dataset (Coveo SIGIR eCom 2021)

Exp5 repeats the Exp4 comparison on a different retailer. It reuses the Exp4 library
(`src/exp4lib` in the repository root): the models, training settings, class weighting, early stopping, paired
session bootstrap and evaluation are identical. The only new part is the data stage, `src/exp4lib/coveo.py`.

## Data access

The dataset is free for research and education, but you have to register first. Fill in the form at
https://www.coveo.com/en/ailabs/sigir-ecom-data-challenge and accept the Terms and Conditions; using the
data implies accepting them. Upload the downloaded zip, unchanged, to Google Drive at
`MyDrive/exp5_coveo/raw/`. The notebook extracts only `browsing_train.csv` and `sku_to_content.csv`.

## How to run (standard Colab)

1. Upload `exp5_Coveo.ipynb`, choose a GPU runtime (L4/A100 recommended; High-RAM recommended).
2. Run Configuration and check that the version it prints matches the version the library cells print.
3. Run Setup, the library cells, then stages 1–4 in order. Every stage can be resumed.

## Mapping to the Exp4 protocol

| | Exp4 (GA4) | Exp5 (Coveo) |
|---|---|---|
| Event tokens | 9 GA4 commerce events | pageview, detail, add, remove, purchase |
| Page / Item | URL path / product in the event's items array | `hashed_url` / `product_sku_hash` |
| Item relations | view_item_of, add_to_cart_of, purchase_item_of | detail_of, add_of, remove_of, purchase_of |
| Labels | Buyer = purchase; Intent = cart/checkout events; Researcher = ≥3 product interactions | Buyer = purchase; Intent = add; Researcher = ≥3 detail events |
| Session context | 6 GA4 categoricals + is_new_user | start hour, start weekday (is_new_user = 0) |
| Item price | GA4 price | `price_bucket` (1–10) |
| Sample | all sessions, T_MAX = 43 | `N_SESSIONS` (300k) most recent sessions of a contiguous period; T_MAX = p95 of training session lengths |
| Undetermined prefix | no decisive event and <3 product interactions | no add/purchase and <3 detail events |

Splits use the same p70/p85 temporal rule on session end time. Items always belong to their own event, so
the Exp3 join leakage cannot occur here.

## Models (same set as Exp4, 5 seeds)

Markov-3, LightGBM and the 13 Exp4 neural models: SASRec+Time, SASRec+Time+SF, SASRec+Time+SF+PageItem,
GRU+Time+SF+PageItem, HGNN_full (2-layer, last-event), HGNN_no_page_item, HGNN_no_page_item_no_SF,
HGNN_full_meanpool, HGNN_full_attnpool, HGNN_full_3layer_meanpool, HomoGraphSAGE+PageItemFeat,
HomoGraphSAGE_meanpool, HomoGraphSAGE_3layer_meanpool. Not ported: HGNN_full_exp3join (the GA4
timestamp-based item join has no Coveo counterpart; Coveo items belong to their own event).

## Rough runtimes (not yet measured on Coveo)

| Stage | Estimate |
|---|---|
| 1. Data (unzip, DuckDB pass over 36M rows, tables) | 15–30 min |
| 2. Markov + LightGBM (30 Optuna trials) | 1–3 h on 8–12 vCPU; much slower on a 2-vCPU T4 runtime (set `LGBM_TUNE_FRAC=0.5`) |
| 3. 13 neural models × 5 seeds (fixed learning rate) | about 5–7 h on an L4 |
| 5. Learning-rate selection + 5 seeds at the chosen rate | about 6–8 h on an L4 |
| 4. Evaluation | under 5 min |

## Outputs (under `MyDrive/exp5_coveo/`)

`prepared/meta.json` holds the sample info, T_MAX and counts by split/class. Per-run results are in
`runs/runs/*.json` and `runs/preds/*.parquet`. The final results are `results/results.json`,
`results/summary.md` and `results/test_predictions_all_models.parquet`.
