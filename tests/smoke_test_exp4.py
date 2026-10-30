"""CPU smoke test of the Exp4 pipeline on synthetic GA4-like data (no BigQuery needed).

    python tests/smoke_test_exp4.py [output_dir]

Builds a small synthetic event table in the raw-extract format, then runs data preparation,
Markov-3, LightGBM, every neural model for 2 epochs x 2 seeds, and the evaluation.
"""
import os, sys, shutil, json
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from exp4lib import data, baselines, train, evaluate
from exp4lib.models import MODEL_SPECS

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_tmp_exp4")
shutil.rmtree(root, ignore_errors=True)
raw, prep, runs, res = [os.path.join(root, d) for d in ("raw", "prep", "runs", "results")]
for d in (raw, prep, runs, res): os.makedirs(d)

rng = np.random.default_rng(0)
NAMES = ['page_view','view_search_results','view_item_list','view_item','add_to_cart',
         'begin_checkout','add_shipping_info','add_payment_info','purchase']
rows, items, sel = [], [], []
t0 = 1_604_000_000_000_000
for u in range(600):
    for s in range(rng.integers(1, 3)):
        sid = int(rng.integers(1e9)); L = int(rng.integers(1, 50)); ts = t0 + int(rng.integers(0, 9e13))
        buyer = rng.random() < 0.1
        for k in range(L):
            if rng.random() > 0.3: ts += int(rng.integers(1, 6e7))   # ~30% share timestamps
            nm = rng.choice(NAMES, p=[.5,.05,.15,.15,.06,.03,.02,.02,.02] if buyer else [.6,.05,.15,.14,.03,.01,.01,.005,.005])
            raw_row = len(rows)
            rows.append(dict(event_timestamp=ts, event_bundle_sequence_id=int(rng.integers(0, 3)), event_name=nm,
                user_pseudo_id=f"u{u}", ga_session_id=sid, ga_session_number=int(s+1),
                page_location=f"https://shop.example/p{rng.integers(0, 30)}/" if rng.random()>.01 else None,
                device_category=rng.choice(['desktop','mobile']), country=rng.choice(['US','UK','DE',None]),
                region='r', traffic_source=rng.choice(['google','(direct)']), traffic_medium='organic',
                traffic_campaign='(none)', row_hash=int(rng.integers(-2**62, 2**62))))
            if nm in ('view_item','add_to_cart','purchase'):
                for _ in range(rng.integers(1, 4)):
                    items.append(dict(raw_row=raw_row, item_id=f"I{rng.integers(0, 80)}", price=float(rng.integers(1, 90))))
            if rng.random() < .03: sel.append(dict(user_pseudo_id=f"u{u}", ga_session_id=sid, event_timestamp=ts, event_name='select_item'))
ev = pd.DataFrame(rows); ev["raw_row"] = np.arange(len(ev))
ev.to_parquet(f"{raw}/raw_events.parquet"); pd.DataFrame(items).to_parquet(f"{raw}/raw_items.parquet")
pd.DataFrame(sel).to_parquet(f"{raw}/raw_select_item.parquet")

meta = data.build_dataset(raw, prep, strict=False)
D = data.Exp4Data(prep)
print("prefixes", D.n, {k: len(v) for k, v in D.rows.items()})
sessions = pd.read_parquet(f"{prep}/sessions.parquet"); events = pd.read_parquet(f"{prep}/events.parquet")

# check: fixed unique_items equals a brute-force recount
it = pd.read_parquet(f"{prep}/items.parquet")
for r in rng.choice(D.n, 200):
    s = D.ev_sess[r]; st = D.sess_start[s]
    brute = it[(it.event_row >= st) & (it.event_row <= r)].item_id.nunique()
    assert brute == D.uniq["fixed"][r], (r, brute, D.uniq["fixed"][r])
print("unique_items brute-force check OK")

baselines.run_markov(D, sessions, runs)
feats = baselines.build_lgbm_features(D, events, sessions, f"{prep}/lgbm_features.parquet")
print("lgbm features", feats.shape)
baselines.run_lightgbm(D, feats, sessions, runs, seeds=[42], n_trials=2, n_jobs=2)
for name, spec in MODEL_SPECS.items():
    spec["train"] = dict(spec["train"], max_epochs=2, patience=1)
    train.train_one(D, name, 42, runs, torch.device("cpu"))
    train.train_one(D, name, 123, runs, torch.device("cpu"))
out, md = evaluate.evaluate_all(D, runs, res, n_boot=50)
print("SMOKE TEST EXP4 OK:", sorted(out["models"]))
