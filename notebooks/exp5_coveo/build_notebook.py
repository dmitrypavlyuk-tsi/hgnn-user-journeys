"""Regenerate exp5_Coveo.ipynb from ../../src/exp4lib/*.py.

    python build_notebook.py

Exp5 repeats the Exp4 comparison on a second dataset (Coveo SIGIR eCom 2021),
with the same model set except the GA4-specific leaky-join reference. The notebook is standalone: the shared library, including the Coveo data
module, is embedded as %%writefile cells.
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "src", "exp4lib")
MODULES = ["__init__", "data", "coveo", "batching", "models", "train", "baselines", "evaluate"]
with open(os.path.join(LIB, "__init__.py"), encoding="utf-8") as f:
    VERSION = f.read().split('__version__ = "')[1].split('"')[0]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Experiment 5: Controlled Comparison on a Second Dataset (Coveo SIGIR eCom 2021)

Exp5 repeats the Exp4 comparison on a different retailer, using the Coveo SIGIR eCom 2021 Data Challenge
browsing data. The task, labels, prefix protocol, models, training settings, class weighting, early stopping and
evaluation are the same as in Exp4. Only the data stage differs:

| | Exp4 (GA4 Merchandise Store) | Exp5 (Coveo) |
|---|---|---|
| Event tokens | 9 GA4 commerce events | pageview, detail, add, remove, purchase |
| Page | URL path | `hashed_url` |
| Item | product in the event's own `items` array | `product_sku_hash` of the product event |
| Item relations | view_item_of, add_to_cart_of, purchase_item_of | detail_of, add_of, remove_of, purchase_of |
| Labels | Buyer = purchase; Intent = cart/checkout; Researcher = ≥3 product interactions | Buyer = purchase; Intent = add; Researcher = ≥3 detail events |
| Session context | 6 GA4 categoricals + is_new_user | start hour, start weekday (no device/traffic fields; is_new_user = 0) |
| Sample | all sessions, T_MAX = 43 | ~300k most recent sessions of a contiguous period; T_MAX = p95 of training lengths |

Stages (each resumable; finished outputs on Google Drive are skipped):
1. **Data**: unzip, sample with DuckDB, build the prefix tables.
2. **Cheap baselines**: Markov-3 and LightGBM.
3. **Neural models**: the 13 Exp4 neural models (main models, ablations, readout/depth variants and homogeneous controls). Only `HGNN_full_exp3join` is not ported: it reproduces the GA4 timestamp-based item join, and Coveo items belong to their own event by construction.
4. **Evaluation**: as in Exp4, with bootstrap tables vs HGNN_full, the GRU and the homogeneous pooled graph.

**Before you start:** register at Coveo and download the dataset zip (research use only, see its Terms and Conditions),
then upload the zip unchanged to `MyDrive/exp5_coveo/raw/` on Google Drive.
**Runtime:** a GPU (L4/A100 recommended, T4 works). High-RAM recommended for stage 1.

**Top-up to 5 seeds (all stages).** `SEEDS` now has 5 seeds. Every stage skips finished runs, so only seeds
789 and 1011 train. Stage 5 already has 5 seeds. Recommended order: setup cells → stage 2 (2 extra LightGBM
seeds, minutes) → stage 5 (no training; copies the new LightGBM seeds and re-evaluates) → stage 3 (fixed
learning rate) → stage 4. Stage 3 copies a run from `lr_selection/final/` instead of retraining when stage 5
chose the default rate for that model (here `HGNN_full`)."""))

cells.append(md("## Configuration"))
cells.append(code(f'''NOTEBOOK_VERSION = "{VERSION}"   # must match exp4lib.__version__ printed after the library cells
print("Exp5 notebook version:", NOTEBOOK_VERSION)

# ---- edit these -------------------------------------------------------------
DRIVE_DIR = "/content/drive/MyDrive/exp5_coveo"   # put the Coveo zip into DRIVE_DIR/raw/
N_SESSIONS = 300_000                  # sessions sampled before the T_MAX filter
END_GAP_DAYS = 1.0                    # sample ends this many days before the last session start
SEEDS = [42, 123, 456, 789, 1011]    # 5 seeds in every stage (789/1011 top up the earlier 3-seed runs)

NEURAL_MODELS = [                     # first the 5 models of the original focused run, then the rest of Exp4
    "HGNN_full",
    "HGNN_full_3layer_meanpool",
    "HomoGraphSAGE_3layer_meanpool",
    "GRU+Time+SF+PageItem",
    "SASRec+Time+SF+PageItem",
    # added 2026-11-01 to match the GA4 model set (ablations, readout/depth variants, homogeneous controls)
    "SASRec+Time",
    "SASRec+Time+SF",
    "HGNN_no_page_item",
    "HGNN_no_page_item_no_SF",
    "HGNN_full_meanpool",
    "HGNN_full_attnpool",
    "HomoGraphSAGE+PageItemFeat",
    "HomoGraphSAGE_meanpool",
]
LR_GRID = [1e-3, 3e-4, 1e-4]           # optional stage 5: learning rate chosen on validation (seed SEEDS[0])
LR_MODELS = NEURAL_MODELS             # all 13 neural models
LR_SEEDS = SEEDS                       # seeds trained at the chosen learning rate
LGBM_TRIALS = 30
LGBM_TUNE_FRAC = 1.0                  # use 0.5 on a 2-vCPU T4 runtime
LGBM_TUNE_TIMEOUT_SEC = 3 * 3600
N_BOOT = 1000
# -------------------------------------------------------------------------------'''))

cells.append(md("## Setup"))
cells.append(code("""!pip -q install torch_geometric optuna lightgbm duckdb
import os, sys, json, time
import torch, psutil
from google.colab import drive
drive.mount("/content/drive")

RAW_DIR = os.path.join(DRIVE_DIR, "raw")
WORK_DIR = "/content/coveo_raw"           # unzipped CSVs on fast local disk (not Drive)
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
print("Files in raw/:", os.listdir(RAW_DIR))

LOG_PATH = os.path.join(DRIVE_DIR, "log.txt")
def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\\n")"""))

cells.append(md("""### Library
The next cells write the shared Exp4/Exp5 code to `/content/exp4lib/`. Re-run them after every runtime restart."""))
cells.append(code("os.makedirs('/content/exp4lib', exist_ok=True)"))
for m in MODULES:
    with open(os.path.join(LIB, m + ".py"), encoding="utf-8") as f:
        src = f.read()
    cells.append(code(f"%%writefile /content/exp4lib/{m}.py\n{src}"))
cells.append(code("""sys.path.insert(0, "/content")
import importlib, exp4lib
from exp4lib import data, coveo, batching, models, train, baselines, evaluate
importlib.reload(exp4lib)
for _m in (data, coveo, batching, models, train, baselines, evaluate):
    importlib.reload(_m)
print("exp4lib loaded, version", exp4lib.__version__)
assert exp4lib.__version__ == NOTEBOOK_VERSION, "Old library code in /content/exp4lib: re-run the %%writefile cells above."
"""))

cells.append(md("""## Stage 1: Data

Unzips `browsing_train.csv` and `sku_to_content.csv` to local disk. DuckDB then streams the 36M-row file and selects
the most recent `N_SESSIONS` sessions of a contiguous period. The notebook builds the Exp4 tables (events, prefixes,
sessions, items, edges) and prints the class and split counts. Expect about 10–20 min (unzip + one pass over the CSV)."""))
cells.append(code("""if not os.path.exists(os.path.join(PREP_DIR, "meta.json")):
    t0 = time.time()
    meta = coveo.build_dataset(RAW_DIR, WORK_DIR, PREP_DIR, n_sessions=N_SESSIONS, end_gap_days=END_GAP_DAYS)
    log(f"Stage 1 done in {(time.time() - t0) / 60:.1f} min")
else:
    log("Stage 1 outputs found on Drive, skipping.")
with open(os.path.join(PREP_DIR, "meta.json")) as f:
    meta = json.load(f)
print("T_MAX:", meta["T_MAX"], "| sample:", meta["sample"])
print(json.dumps(meta["counts"], indent=2))"""))

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
del feats, events_df; gc.collect()
for p in sorted(os.listdir(os.path.join(RUN_DIR, "runs"))):
    if p.startswith(("Markov", "LightGBM__")):
        r = json.load(open(os.path.join(RUN_DIR, "runs", p)))
        print(f"{p[:-5]:28s} val={r['best_val_macro_f1']:.4f} test={r['test_macro_f1']:.4f}")"""))

cells.append(md("""## Stage 3: Neural models

Seed by seed, epoch checkpoints on Drive, resumable. Prints the estimated GPU time left after each run."""))
cells.append(code("""if "D" not in globals():
    D = data.Exp4Data(PREP_DIR)
jobs = [(s, m) for s in SEEDS for m in NEURAL_MODELS]
t_stage = time.time()
for seed, name in jobs:
    if train.is_done(RUN_DIR, name, seed) or train.reuse_from_lr_final(RUN_DIR, LR_ROOT, name, seed, log=log):
        continue
    train.train_one(D, name, seed, RUN_DIR, DEVICE, log=log)
    torch.cuda.empty_cache()
    times = {}
    for s, m in jobs:
        p = os.path.join(RUN_DIR, "runs", f"{m}__seed{s}.json")
        if os.path.exists(p):
            times.setdefault(m, []).append(json.load(open(p))["runtime_sec"])
    left = sum(sum(times[m]) / len(times[m]) if m in times else float("nan")
               for s, m in jobs if not train.is_done(RUN_DIR, m, s))
    log(f"Estimated GPU time left: {left / 3600:.1f} h (nan = some models not timed yet)")
log(f"Stage 3 finished ({(time.time() - t_stage) / 3600:.1f} h in this session)")"""))

cells.append(md("""## Stage 4: Evaluation

Same metrics as Exp4: Macro-F1 mean ± std and per-class F1; paired bootstrap over test sessions (1,000 resamples)
vs `HGNN_full`, vs `GRU+Time+SF+PageItem` and vs `HomoGraphSAGE_3layer_meanpool`; prefix-length bins; undetermined
prefixes (no add/purchase and fewer than 3 detail events). Outputs: `results/results.json`, `results/summary.md`."""))
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
out_path = os.path.join(HERE, "exp5_Coveo.ipynb")
nbf.write(nb, out_path)
print("wrote", out_path, "| version", VERSION)
