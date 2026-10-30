# Experiment 4: Controlled comparison

This experiment re-runs the baselines from Exp2 and the HGNN from Exp3 so that **every model uses the same
data**: the Exp3 dataset (9 event types, `T_MAX = 43`, temporal p70/p85 split, 327,283 sessions,
1,277,287 prefixes). Every model is scored on the **same 211,277 test prefixes**. All models use prefix-row
class weights and early stopping on validation Macro-F1. The notebook also checks the Exp3 item join for
leakage and fixes it.

## Files

| File | Purpose |
|---|---|
| `exp4_ControlledComparison.ipynb` | Standalone Colab notebook; this is the only file you need to upload. |
| `../../src/exp4lib/*.py` | Library code. The notebook embeds these files as `%%writefile` cells. |
| `build_notebook.py` | Regenerates the notebook after you edit `src/exp4lib/`. Run it with `python build_notebook.py` (needs `nbformat`). |

## How to run (standard Colab)

1. Upload `exp4_ControlledComparison.ipynb` to Colab.
2. Go to *Runtime → Change runtime type* and choose a GPU. L4 or A100 is recommended; T4 also works. High-RAM is recommended but not required.
3. In the **Configuration** cell, set:
   - `BQ_PROJECT`: any Google Cloud project you can run queries in. A free BigQuery sandbox project is fine. The notebook only reads the public `ga4_obfuscated_sample_ecommerce` dataset.
   - `DRIVE_DIR`: where artifacts are stored. The default is `MyDrive/exp4_controlled`.
   - `SEEDS`: three seeds by default. Extend the list to `[42, 123, 456, 789, 1011]` to get five.
4. Run all cells. Colab asks for Google Drive access and then for Google Cloud access (stage 1 only).
5. If the runtime disconnects: reconnect and run the cells from **Configuration** down to the library import again, then re-run the stage you were in. Finished outputs are skipped. An interrupted neural run resumes from its last completed epoch.

To go from 3 to 5 seeds later, change `SEEDS` and re-run stages 2–4. Only the missing seeds are trained.

## Stages, outputs and rough runtimes

The runtimes below are estimates and have not been measured. The notebook logs the real time per run in
`runs/runs/*.json` and in `log.txt`, and stage 3 prints the estimated GPU time left after every run.

| Stage | What it does | Outputs (under `DRIVE_DIR`) | Rough runtime |
|---|---|---|---|
| 1. Data | BigQuery extraction; Exp3 count checks (PASS/FAIL per count); leakage check; fixed item join | `raw/*.parquet`, `prepared/{events,prefixes,sessions,items}.parquet`, `prepared/edges.npz`, `prepared/meta.json` (repro report and leakage counts) | 10–25 min (BigQuery download speed varies) |
| 2. Cheap baselines | Markov-3 (alpha selected on val); LightGBM features, Optuna tuning (30 trials), final fits per seed | `runs/runs/{Markov-3,LightGBM}__seed*.json`, `runs/preds/*.parquet`, `prepared/lgbm_features.parquet` | Markov: under 5 min. LightGBM: 1–3 h on 8–12 vCPU (L4/A100). On a 2-vCPU T4 runtime it is much slower; set `LGBM_TUNE_FRAC=0.5` and/or fewer trials. |
| 3. Neural models | 8 models × seeds, plus the optional leaky-join HGNN reference | `runs/runs/<model>__seed<s>.json` (history, val/test F1, runtime), `runs/preds/<model>__seed<s>.parquet` (test predictions + probabilities), `runs/ckpt/*__best.pt` | L4: roughly 20–40 min per HGNN-type run and 10–20 min per sequence-model run, so about 8–14 h for 3 seeds. A100 is faster; T4 is about 1.5–2× slower. |
| 4. Evaluation | Metrics, paired session bootstrap, prefix-length bins, undetermined subset | `results/results.json`, `results/summary.md`, `results/test_predictions_all_models.parquet` | 2–5 min |

Peak RAM in stages 1–2 should be about 6–9 GB, which fits the 12.7 GB standard tier. Use High-RAM if
stage 1 runs out of memory.

## Models

| Name | Description |
|---|---|
| `Markov-3` | Exp2 stepwise backoff over the last 3 events; alpha from {0, .05, .1, .2, .3, .5} selected on val. |
| `LightGBM` | Exp2 prefix features (counts/frequencies, entropy, key-event flags, timing, suffix bi/trigrams) plus the Exp3 session features (`prefix_len`, `unique_items`, `cart_flag`, `is_new_user`, 6 categoricals); Exp2 Optuna search space. |
| `SASRec+Time` | Exp2 architecture (2 layers, 2 heads, hidden 64, dropout 0.1). |
| `SASRec+Time+SF` | Adds the HGNN session vector (6 categorical embeddings + 4 numerics) before an HGNN-style MLP head. |
| `SASRec+Time+SF+PageItem` | Also adds page embeddings and mean item embeddings (item + price) to every event token. |
| `GRU+Time+SF+PageItem` | GRU4Rec-style 1-layer GRU with the same inputs as the previous model. |
| `HGNN_full` | Exp3 SessionHGNN (HeteroConv/SAGEConv over Event/Page/Item). This is the reference for the bootstrap. |
| `HGNN_no_page_item` | Exp3 ablation: event chain only, with session features. |
| `HGNN_no_page_item_no_SF` | New: event chain only, with no session features. |
| `HomoGraphSAGE+PageItemFeat` | Homogeneous GraphSAGE on the event chain; page and item enter as event features (separates "graph" from "heterogeneous"). |
| `HGNN_full_meanpool`, `HGNN_full_attnpool`, `HGNN_full_3layer_meanpool` | Stage 3b: HGNN_full with a pooled readout over all event nodes (mean or attention), optionally with 3 layers. |
| `HomoGraphSAGE_meanpool`, `HomoGraphSAGE_3layer_meanpool` | Stage 3c: homogeneous counterparts of the pooled HGNNs. They isolate heterogeneity once the readout is fixed. |
| `HGNN_full_exp3join` | Reference only: HGNN_full trained on the original Exp3 item join, to measure the leakage effect. |

Training settings follow the thesis for each model family. Graph models use Exp3 7.0: batch 256, Adam 1e-3,
up to 50 epochs, patience 7. Sequence models use Exp2 6.5: batch 512, Adam 1e-3, up to 30 epochs, patience 3.

## Differences from Exp2/Exp3 to keep in mind

- **Event order.** Events with an identical `(timestamp, bundle id)` are ordered by `event_name`, then by a row hash. Exp3 left these ties in BigQuery output order, which is not reproducible. Session, prefix and label counts do not depend on this order.
- **Item join (the fix).** Items come from the event's own `items` array. Exp3 matched them on `(event_timestamp, event_bundle_sequence_id)`. Stage 1 reports how many prefixes the Exp3 join affected.
- **SASRec padding.** SASRec uses right padding with a causal mask. This is equivalent to Exp2's left padding with mask-derived positions, and it avoids fully masked attention rows.
- **Undetermined subset.** "Product interactions" counts the events the models can see (`view_item`, `view_item_list`). `select_item` is excluded because it is not in the model stream.
