"""Regenerate exp4_ControlledComparison.ipynb from ../../src/exp4lib/*.py.

    python build_notebook.py

The notebook is standalone: each library module is embedded as a %%writefile
cell, so only the .ipynb has to be uploaded to Colab.
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "src", "exp4lib")
MODULES = ["__init__", "data", "coveo", "batching", "models", "train", "baselines", "evaluate"]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Experiment 4: Controlled Comparison on the Exp3 Dataset

Every model is trained on the **same Exp3 dataset and temporal split** (327,283 sessions,
1,277,287 prefixes) and scored on the **identical 211,277 test prefixes**, with prefix-row
class weights and early stopping on validation Macro-F1.

Stages (each is resumable; finished outputs on Google Drive are skipped):

1. **Data**: BigQuery extraction, exact reproduction of the Exp3 counts, item-join leakage check, fixed per-event item join.
2. **Cheap baselines**: Markov-3 with backoff, LightGBM (Exp2 features + Exp3 session features, Optuna).
3. **Neural models**: SASRec variants, GRU, HGNN variants, homogeneous GraphSAGE; one checkpoint per epoch, one result file per (model, seed).
4. **Evaluation**: Macro-F1 mean ± std, per-class F1, paired session-level bootstrap vs `HGNN_full`, prefix-length bins, undetermined-prefix subset, summary tables.

Set the configuration cell, then run top to bottom. After a disconnect, run the setup cells
again and continue from the stage you were in.

**Runtime:** set *Runtime → Change runtime type* to a GPU (L4 or A100 recommended, T4 works).
High-RAM is recommended but not required.

**Top-up to 5 seeds (all stages).** `SEEDS` now has 5 seeds. Every stage skips finished runs, so only seeds
789 and 1011 train. Recommended order: setup cells → stage 2 (2 extra LightGBM seeds, minutes) → stage 5
(selected learning rate) → stages 3/3b/3c (fixed learning rate) → stage 4. Fixed-rate stages copy a run
from `lr_selection/final/` instead of retraining when stage 5 chose the default rate for that model."""))

cells.append(md("## Configuration"))
cells.append(code("""NOTEBOOK_VERSION = "2026-11-01.1"   # must match exp4lib.__version__ printed after the library cells
print("Exp4 notebook version:", NOTEBOOK_VERSION)

# ---- edit these -------------------------------------------------------------
BQ_PROJECT = "your-gcp-project-id"   # any project you can run queries in (BigQuery sandbox is fine)
DRIVE_DIR = "/content/drive/MyDrive/exp4_controlled"   # all artifacts, checkpoints and results

SEEDS = [42, 123, 456, 789, 1011]    # 5 seeds in every stage (789/1011 top up the earlier 3-seed runs)
STRICT_REPRO = True                   # stop if the Exp3 counts are not reproduced exactly

NEURAL_MODELS = [                     # stage 3, trained seed by seed in this order
    "HGNN_full",
    "HGNN_no_page_item",
    "HGNN_no_page_item_no_SF",
    "HomoGraphSAGE+PageItemFeat",
    "SASRec+Time",
    "SASRec+Time+SF",
    "SASRec+Time+SF+PageItem",
    "GRU+Time+SF+PageItem",
]
STAGE_3B_MODELS = [                   # optional stage 3b: HGNN readout/depth variants
    "HGNN_full_meanpool",
    "HGNN_full_attnpool",
    "HGNN_full_3layer_meanpool",
]
STAGE_3B_SEEDS = SEEDS
STAGE_3C_MODELS = [                   # optional stage 3c: homogeneous graph with pooled readout
    "HomoGraphSAGE_3layer_meanpool",
    "HomoGraphSAGE_meanpool",
]
STAGE_3C_SEEDS = SEEDS
LR_GRID = [1e-3, 3e-4, 1e-4]           # optional stage 5: learning rate chosen on validation (seed SEEDS[0])
LR_MODELS = NEURAL_MODELS + STAGE_3B_MODELS + STAGE_3C_MODELS   # all 13 neural models
LR_SEEDS = SEEDS                       # seeds trained at the chosen learning rate
RUN_LEAKY_REFERENCE = True            # also train HGNN_full on the original (leaky) Exp3 item join
LEAKY_REFERENCE_SEEDS = SEEDS

LGBM_TRIALS = 30                      # Optuna trials (Exp2 used 50)
LGBM_TUNE_FRAC = 1.0                  # fraction of tuning-fit sessions; use 0.5 on a 2-vCPU T4 runtime
LGBM_TUNE_TIMEOUT_SEC = 3 * 3600      # hard cap on tuning time
N_BOOT = 1000
# -------------------------------------------------------------------------------"""))

cells.append(md("## Setup"))
cells.append(code("""!pip -q install torch_geometric optuna lightgbm
import os, sys, json, time, shutil, subprocess
import torch, psutil
from google.colab import drive
drive.mount("/content/drive")

RAW_DIR = os.path.join(DRIVE_DIR, "raw")
PREP_DIR = os.path.join(DRIVE_DIR, "prepared")
RUN_DIR = os.path.join(DRIVE_DIR, "runs")
RESULTS_DIR = os.path.join(DRIVE_DIR, "results")
for d in (RAW_DIR, PREP_DIR, RUN_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB, vCPU: {os.cpu_count()}")

LR_ROOT = os.path.join(DRIVE_DIR, "lr_selection")   # stage 5 output, also read by the fixed-rate stages
LOG_PATH = os.path.join(DRIVE_DIR, "log.txt")
def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\\n")"""))

cells.append(md("""### Library
The next cells write the Exp4 code to `/content/exp4lib/`. Re-run them after every runtime restart."""))
cells.append(code("os.makedirs('/content/exp4lib', exist_ok=True)"))
for m in MODULES:
    with open(os.path.join(LIB, m + ".py"), encoding="utf-8") as f:
        src = f.read()
    cells.append(code(f"%%writefile /content/exp4lib/{m}.py\n{src}"))
cells.append(code("""sys.path.insert(0, "/content")
import importlib, exp4lib
from exp4lib import data, batching, models, train, baselines, evaluate
importlib.reload(exp4lib)
for _m in (data, batching, models, train, baselines, evaluate):
    importlib.reload(_m)
print("exp4lib loaded, version", exp4lib.__version__)
assert exp4lib.__version__ == NOTEBOOK_VERSION, "Old library code in /content/exp4lib: re-run the %%writefile cells above."
"""))

cells.append(md("""## Stage 1: Extraction, Exp3 count reproduction, leakage check

* Runs the Exp3 queries (9 event types, plus `select_item` for the Researcher label only).
  The events query also returns each event's own `items` array. That is the fix for the Exp3 join, which
  matched items to events on `(event_timestamp, event_bundle_sequence_id)` only.
* Rebuilds sessions, labels, the `T_MAX = 43` filter and the p70/p85 split exactly as in Exp3, and checks every
  count against the Exp3 notebook.
* Leakage check: for every prefix it measures whether the Exp3 join pulled in items from events at positions `> t`
  (for the `unique_items` aggregate in Exp3 cell 98 and for the graph builder). All Exp4 models use the fixed join."""))
cells.append(code("""if not os.path.exists(os.path.join(PREP_DIR, "meta.json")):
    from google.colab import auth
    auth.authenticate_user()
    from google.cloud import bigquery
    client = bigquery.Client(project=BQ_PROJECT)
    t0 = time.time()
    data.extract_raw(client, RAW_DIR)
    meta = data.build_dataset(RAW_DIR, PREP_DIR, strict=STRICT_REPRO)
    log(f"Stage 1 done in {(time.time() - t0) / 60:.1f} min")
else:
    log("Stage 1 outputs found on Drive, skipping.")
with open(os.path.join(PREP_DIR, "meta.json")) as f:
    meta = json.load(f)
print("All Exp3 counts reproduced:", meta["repro_all_pass"])
for r in meta["repro_report"]:
    print(("PASS " if r["ok"] else "FAIL ") + r["check"])
print(json.dumps(meta["leakage"], indent=2))"""))

cells.append(md("## Stage 2: Markov-3 and LightGBM"))
cells.append(code("""import pandas as pd, gc
D = data.Exp4Data(PREP_DIR)
sessions_df = pd.read_parquet(os.path.join(PREP_DIR, "sessions.parquet"))
print({k: len(v) for k, v in D.rows.items()}, "class weights:", D.class_weights())

baselines.run_markov(D, sessions_df, RUN_DIR, n=3, log=log)

events_df = pd.read_parquet(os.path.join(PREP_DIR, "events.parquet"))
feats = baselines.build_lgbm_features(D, events_df, sessions_df,
                                      os.path.join(PREP_DIR, "lgbm_features.parquet"), log=log)
baselines.run_lightgbm(D, feats, sessions_df, RUN_DIR, seeds=SEEDS, n_trials=LGBM_TRIALS,
                       tune_frac=LGBM_TUNE_FRAC, timeout_sec=LGBM_TUNE_TIMEOUT_SEC, log=log)
del feats, events_df; gc.collect()"""))
cells.append(code("""# Interim results: cheap baselines
for p in sorted(os.listdir(os.path.join(RUN_DIR, "runs"))):
    if p.startswith(("Markov", "LightGBM__")):
        r = json.load(open(os.path.join(RUN_DIR, "runs", p)))
        print(f"{p[:-5]:28s} val={r['best_val_macro_f1']:.4f} test={r['test_macro_f1']:.4f}")"""))

cells.append(md("""## Stage 3: Neural models

The loop runs seed by seed (all models for the first seed, then the next seed), so a complete 3-seed table
exists before any 4th or 5th seed starts. Each epoch is checkpointed to Drive; an interrupted run resumes
from its last finished epoch. After each run it prints an estimate of the remaining GPU time."""))
cells.append(code("""if "D" not in globals():
    D = data.Exp4Data(PREP_DIR)

jobs = [(s, m) for s in SEEDS for m in NEURAL_MODELS]
if RUN_LEAKY_REFERENCE:
    jobs += [(s, "HGNN_full_exp3join") for s in LEAKY_REFERENCE_SEEDS]

def remaining_estimate():
    per_model = {}
    for s, m in jobs:
        p = os.path.join(RUN_DIR, "runs", f"{m}__seed{s}.json")
        if os.path.exists(p):
            per_model.setdefault(m, []).append(json.load(open(p))["runtime_sec"])
    left = 0.0
    for s, m in jobs:
        if not train.is_done(RUN_DIR, m, s):
            left += sum(per_model[m]) / len(per_model[m]) if m in per_model else float("nan")
    return left / 3600

t_stage = time.time()
for seed, name in jobs:
    if train.is_done(RUN_DIR, name, seed) or train.reuse_from_lr_final(RUN_DIR, LR_ROOT, name, seed, log=log):
        continue
    train.train_one(D, name, seed, RUN_DIR, DEVICE, log=log)
    torch.cuda.empty_cache()
    log(f"Estimated GPU time left: {remaining_estimate():.1f} h (nan = some models not timed yet)")
log(f"Stage 3 finished ({(time.time() - t_stage) / 3600:.1f} h in this session)")"""))

cells.append(md("""## Stage 3b (optional): HGNN readout and depth variants

Exp4 showed `HGNN_full` below the sequence models with equal inputs, with the gap widest on long prefixes.
Exp3 reads out only the last event node after 2 message-passing layers, so the readout sees at most the last
3 events of the chain (plus page/item neighbours). These variants test whether the readout and depth explain the gap.
Same data, fixed item join, training protocol and seeds as `HGNN_full`:

* `HGNN_full_meanpool`: [last event, mean over all event nodes] + session features
* `HGNN_full_attnpool`: [last event, gated attention pooling over all event nodes] + session features
* `HGNN_full_3layer_meanpool`: 3 HeteroConv layers + mean pooling

Uses the existing Drive artifacts; finished runs are skipped. Re-run stage 4 afterwards."""))
cells.append(code("""if "D" not in globals():
    D = data.Exp4Data(PREP_DIR)
jobs_3b = [(s, m) for s in STAGE_3B_SEEDS for m in STAGE_3B_MODELS]
t_stage = time.time()
for seed, name in jobs_3b:
    if train.is_done(RUN_DIR, name, seed) or train.reuse_from_lr_final(RUN_DIR, LR_ROOT, name, seed, log=log):
        continue
    train.train_one(D, name, seed, RUN_DIR, DEVICE, log=log)
    torch.cuda.empty_cache()
    left = [m for s, m in jobs_3b if not train.is_done(RUN_DIR, m, s)]
    done = [json.load(open(os.path.join(RUN_DIR, "runs", f"{m}__seed{s}.json")))["runtime_sec"]
            for s, m in jobs_3b if train.is_done(RUN_DIR, m, s)]
    log(f"Stage 3b: {len(left)} runs left, ~{len(left) * sum(done) / len(done) / 3600:.1f} h")
log(f"Stage 3b finished ({(time.time() - t_stage) / 3600:.1f} h in this session)")"""))

cells.append(md("""## Stage 3c (optional): homogeneous graph with pooled readout

Control for stage 3b. The homogeneous GraphSAGE gets exactly the same readout and depth as the pooled HGNNs
(mean pooling over all event nodes; 3 layers and 2 layers), but page and item enter as event-node features
instead of shared Page/Item nodes. Comparing `HomoGraphSAGE_3layer_meanpool` with `HGNN_full_3layer_meanpool`
isolates the effect of heterogeneity once the readout is fixed. Same data, protocol and seeds; finished runs
are skipped. Re-run stage 4 afterwards (it adds a bootstrap table vs `HomoGraphSAGE_3layer_meanpool`)."""))
cells.append(code("""if "D" not in globals():
    D = data.Exp4Data(PREP_DIR)
jobs_3c = [(s, m) for s in STAGE_3C_SEEDS for m in STAGE_3C_MODELS]
t_stage = time.time()
for seed, name in jobs_3c:
    if train.is_done(RUN_DIR, name, seed) or train.reuse_from_lr_final(RUN_DIR, LR_ROOT, name, seed, log=log):
        continue
    train.train_one(D, name, seed, RUN_DIR, DEVICE, log=log)
    torch.cuda.empty_cache()
    left = [m for s, m in jobs_3c if not train.is_done(RUN_DIR, m, s)]
    done = [json.load(open(os.path.join(RUN_DIR, "runs", f"{m}__seed{s}.json")))["runtime_sec"]
            for s, m in jobs_3c if train.is_done(RUN_DIR, m, s)]
    log(f"Stage 3c: {len(left)} runs left, ~{len(left) * sum(done) / len(done) / 3600:.1f} h")
log(f"Stage 3c finished ({(time.time() - t_stage) / 3600:.1f} h in this session)")"""))

cells.append(md("""## Stage 4: Evaluation

* Test Macro-F1 mean ± std over seeds and per-class F1.
* Paired bootstrap: 1,000 resamples of **test sessions** (all prefixes of a sampled session move together).
  Deltas vs `HGNN_full` for (a) the representative seed of each model (the seed closest to its mean) and
  (b) the seed-averaged Macro-F1. 95% percentile CIs. Further tables give deltas vs `GRU+Time+SF+PageItem` and vs `HomoGraphSAGE_3layer_meanpool` (heterogeneous vs homogeneous under the pooled readout).
* Macro-F1 by prefix-length bins (1, 2, 3, 4, 5, 6-7, 8-10, 11-15, 16-20, 21+).
* Undetermined prefixes: no add_to_cart/begin_checkout/add_shipping_info/add_payment_info/purchase and fewer
  than 3 observable product interactions (view_item + view_item_list) in the prefix.

Outputs in `results/`: `results.json`, `summary.md`, `test_predictions_all_models.parquet`.
Per-run predictions with class probabilities are in `runs/preds/`."""))
cells.append(code("""if "D" not in globals():
    D = data.Exp4Data(PREP_DIR)
out, md_text = evaluate.evaluate_all(D, RUN_DIR, RESULTS_DIR, n_boot=N_BOOT, log=log)
from IPython.display import Markdown, display
display(Markdown(md_text))"""))

cells.append(md("""## Stage 5 (optional): learning-rate selection

Robustness check of the training protocol. For each neural model, the learning rate is chosen from
`LR_GRID` on **validation** Macro-F1 with seed 42, then all seeds are trained at the chosen rate. Patience,
batch size, epochs, class weights and data are unchanged. Existing runs are reused: the default rate (1e-3)
comes from stage 3, so only the lower rates are trained. Results are written separately
(`lr_selection/final/`, `results_lr/`); the stage 4 results stay untouched. Resumable like every other stage."""))
cells.append(code("""if "D" not in globals():
    D = data.Exp4Data(PREP_DIR)
RESULTS_LR_DIR = os.path.join(DRIVE_DIR, "results_lr")
t_stage = time.time()
choice = train.lr_selection(D, LR_MODELS, RUN_DIR, LR_ROOT, DEVICE, lrs=LR_GRID,
                            select_seed=SEEDS[0], seeds=LR_SEEDS, log=log)
log(f"Stage 5 training finished ({(time.time() - t_stage) / 3600:.1f} h in this session)")
print(json.dumps(choice, indent=2))
out_lr, md_lr = evaluate.evaluate_all(D, os.path.join(LR_ROOT, "final"), RESULTS_LR_DIR, n_boot=N_BOOT,
                                      title=f"Controlled comparison ({D.meta.get('dataset', 'GA4')}), "
                                            "learning rate selected on validation: test results", log=log)
with open(os.path.join(RESULTS_LR_DIR, "lr_choice.json"), "w") as f:
    json.dump(choice, f, indent=2)
from IPython.display import Markdown, display
display(Markdown(md_lr))"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
out_path = os.path.join(HERE, "exp4_ControlledComparison.ipynb")
nbf.write(nb, out_path)
print("wrote", out_path)
