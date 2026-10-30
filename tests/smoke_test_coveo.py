"""CPU smoke test of the Exp5 (Coveo) pipeline on synthetic data in the Coveo format.

    python tests/smoke_test_coveo.py [output_dir]

Creates a synthetic zip, then runs zip extraction, DuckDB sampling, data preparation,
Markov-3, LightGBM, the five focused neural models (1 epoch x 2 seeds) and the evaluation.
"""
import os, shutil, subprocess, sys
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from exp4lib import coveo, data, baselines, train, evaluate
from exp4lib.models import MODEL_SPECS

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_tmp_coveo")
shutil.rmtree(root, ignore_errors=True)
subprocess.run([sys.executable, os.path.join(HERE, "make_fake_coveo.py"), os.path.join(root, "raw")], check=True)

coveo.build_dataset(f"{root}/raw", f"{root}/work", f"{root}/prep", n_sessions=2500, end_gap_days=1)
D = data.Exp4Data(f"{root}/prep")
items = pd.read_parquet(f"{root}/prep/items.parquet")
for r in np.random.default_rng(0).choice(D.n, 200):
    st = D.sess_start[D.ev_sess[r]]
    assert items[(items.event_row >= st) & (items.event_row <= r)].item_id.nunique() == D.uniq["fixed"][r]
sessions = pd.read_parquet(f"{root}/prep/sessions.parquet")
events = pd.read_parquet(f"{root}/prep/events.parquet")
baselines.run_markov(D, sessions, f"{root}/runs")
feats = baselines.build_lgbm_features(D, events, sessions, f"{root}/prep/lgbm_features.parquet")
baselines.run_lightgbm(D, feats, sessions, f"{root}/runs", seeds=[42], n_trials=2, n_jobs=2)
for name in ["SASRec+Time+SF+PageItem", "GRU+Time+SF+PageItem", "HGNN_full",
             "HGNN_full_3layer_meanpool", "HomoGraphSAGE_3layer_meanpool"]:
    MODEL_SPECS[name]["train"] = dict(MODEL_SPECS[name]["train"], max_epochs=1, patience=1)
    for seed in (42, 123):
        train.train_one(D, name, seed, f"{root}/runs", torch.device("cpu"))
out, _ = evaluate.evaluate_all(D, f"{root}/runs", f"{root}/results", n_boot=50)
print("SMOKE TEST COVEO OK:", sorted(out["models"]), "| bootstrap refs:", list(out["bootstrap_vs"]))
