"""Markov-3 with stepwise backoff (Exp2 Section 4) and LightGBM on Exp2-style
prefix features plus the Exp3 session features (Exp2 Section 5), both on the
Exp3 dataset. Prefix row i == event row i (see data.py)."""
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


EXP2_LABEL_ORDER = [3, 2, 1, 0]  # Exp2 LABELS = Buyer, Intent, Researcher, Browser (argmax tie order)
ALPHA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def _mf1(y, p):
    return float(f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0))


def _save_preds(path, rows, prob):
    df = pd.DataFrame({"row": rows, "pred": prob.argmax(1).astype(np.int8)})
    for c in range(4):
        df[f"p{c}"] = prob[:, c].astype(np.float32)
    df.to_parquet(path, index=False)


# --------------------------------------------------------------------------
# Markov-3
# --------------------------------------------------------------------------
def run_markov(D, sessions_df, run_dir, n=3, log=print):
    tag = f"Markov-{n}__seed0"
    if os.path.exists(os.path.join(run_dir, "runs", tag + ".json")):
        log(f"[{tag}] already done, skipping")
        return
    t0 = time.time()
    tok = D.ev_name
    pos = D.ev_pos
    codes = {}
    for k in range(1, n + 1):
        c = np.zeros(D.n, dtype=np.int64)
        for j in range(k):
            sh = np.r_[np.zeros(j, dtype=np.int64), tok[:D.n - j]] if j else tok
            c = c * 16 + sh
        c[pos < k] = -1  # state needs t >= k
        codes[k] = c

    tr = D.rows["train"]
    tables = {}
    for k in range(1, n + 1):
        m = codes[k][tr] >= 0
        df = pd.DataFrame({"s": codes[k][tr][m], "y": D.y[tr][m]})
        tab = df.groupby(["s", "y"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
        tables[k] = tab
    # prior from train session labels
    ss = sessions_df[sessions_df["split"] == "train"]["y"].map({"Browser": 0, "Researcher": 1, "Intent": 2, "Buyer": 3})
    prior = np.bincount(ss, minlength=4) / len(ss)

    def dists(rows):
        emp = np.full((len(rows), 4), np.nan)
        for k in range(1, n + 1):  # higher order overwrites lower when seen
            tab = tables[k]
            c = codes[k][rows]
            hit = np.isin(c, tab.index.to_numpy())
            cnt = tab.reindex(c[hit]).to_numpy(np.float64)
            emp[hit] = cnt / cnt.sum(1, keepdims=True)
        return emp

    def predict(emp, alpha):
        d = np.where(np.isnan(emp[:, :1]), prior[None, :], (1 - alpha) * np.nan_to_num(emp) + alpha * prior[None, :])
        ordered = d[:, EXP2_LABEL_ORDER]
        return d, np.array(EXP2_LABEL_ORDER)[ordered.argmax(1)]

    val = D.rows["val"]
    emp_val = dists(val)
    scores = {a: _mf1(D.y[val], predict(emp_val, a)[1]) for a in ALPHA_GRID}
    best_alpha = max(scores, key=scores.get)  # first best in grid order
    test = D.rows["test"]
    prob, pred = predict(dists(test), best_alpha)
    test_f1 = _mf1(D.y[test], pred)
    os.makedirs(os.path.join(run_dir, "preds"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "runs"), exist_ok=True)
    df = pd.DataFrame({"row": test, "pred": pred.astype(np.int8)})
    for c in range(4):
        df[f"p{c}"] = prob[:, c].astype(np.float32)
    df.to_parquet(os.path.join(run_dir, "preds", tag + ".parquet"), index=False)
    info = {"model": f"Markov-{n}", "seed": 0, "alpha_val_scores": scores, "best_alpha": best_alpha,
            "best_val_macro_f1": scores[best_alpha], "test_macro_f1": test_f1,
            "n_states": {k: int(len(t)) for k, t in tables.items()},
            "runtime_sec": round(time.time() - t0, 1)}
    with open(os.path.join(run_dir, "runs", tag + ".json"), "w") as f:
        json.dump(info, f, indent=2)
    log(f"[{tag}] alpha={best_alpha} val={scores[best_alpha]:.4f} test={test_f1:.4f}")


# --------------------------------------------------------------------------
# LightGBM features
# --------------------------------------------------------------------------
TOP_K = 50
NGRAM_COVERAGE = 0.95
CONTEXT_FIELDS_C = ["country", "region", "traffic_campaign"]


def _select_top_m(counter, coverage):
    total = sum(counter.values())
    out, cum = [], 0
    for ng, c in counter.most_common():
        cum += c
        out.append(ng)
        if cum >= total * coverage:
            break
    return out


def build_lgbm_features(D, events_df, sessions_df, out_path, log=print):
    """Exp2 Section 5.2 features computed for every prefix row at once."""
    if os.path.exists(out_path):
        log("LightGBM feature table exists, loading.")
        return pd.read_parquet(out_path)
    t0 = time.time()
    name = events_df["event_name"].to_numpy()
    s = events_df["session_idx"].to_numpy()
    ts_sec = events_df["event_timestamp"].to_numpy(np.int64) / 1e6
    tpos = D.prefix_len.astype(np.float32)
    vocab = sorted(pd.unique(name[D.rows["train"]]))
    F = {"prefix_len": tpos}

    grp = pd.Series(s)
    freqs = []
    for tokname in vocab:
        cnt = pd.Series((name == tokname).astype(np.int32)).groupby(grp).cumsum().to_numpy().astype(np.float32)
        F[f"count_{tokname}"] = cnt
        F[f"freq_{tokname}"] = cnt / tpos
        freqs.append(cnt / tpos)
    fr = np.stack(freqs, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        F["entropy"] = np.where(fr > 0, -fr * np.log(fr), 0).sum(1).astype(np.float32)
    for e in D.key_events:
        F[f"has_{e}"] = pd.Series((name == e).astype(np.int8)).groupby(grp).cummax().to_numpy().astype(np.float32)

    dt = events_df["dt_sec"].to_numpy(np.float64)
    first = D.prefix_len == 1
    posdt = np.where((~first) & (dt > 0), dt, 0.0)
    npos = pd.Series(((~first) & (dt > 0)).astype(np.int32)).groupby(grp).cumsum().to_numpy()
    sdt = pd.Series(posdt).groupby(grp).cumsum().to_numpy()
    F["avg_dt"] = np.where(npos > 0, sdt / np.maximum(npos, 1), 0.0).astype(np.float32)
    start_ts = pd.Series(ts_sec).groupby(grp).transform("first").to_numpy()
    F["time_since_start"] = (ts_sec - start_ts).astype(np.float32)
    max_dt = D.meta["dt_stats"]["MAX_DT_sec"]
    for e in D.key_events:
        last = pd.Series(np.where(name == e, ts_sec, np.nan)).groupby(grp).ffill().to_numpy()
        occ = ~np.isnan(last)
        F[f"has_{e}_occurred"] = occ.astype(np.float32)
        F[f"time_since_last_{e}"] = np.where(occ, ts_sec - np.nan_to_num(last), max_dt).astype(np.float32)

    # suffix n-grams, vocab from train (top-M covering 95% of occurrences)
    tok_id = pd.Series(name).map({v: i for i, v in enumerate(vocab)}).fillna(len(vocab)).to_numpy(np.int64)
    V = len(vocab) + 1
    prev1 = np.r_[-1, tok_id[:-1]]
    prev2 = np.r_[-1, -1, tok_id[:-2]]
    bg = np.where(D.prefix_len >= 2, prev1 * V + tok_id, -1)
    tg = np.where(D.prefix_len >= 3, (prev2 * V + prev1) * V + tok_id, -1)
    tr = D.rows["train"]
    for nm, codes in (("bigram", bg), ("trigram", tg)):
        c_tr = codes[tr]
        top = _select_top_m(Counter(c_tr[c_tr >= 0].tolist()), NGRAM_COVERAGE)
        idx = {c: i for i, c in enumerate(top)}
        mapped = pd.Series(codes).map(idx).fillna(len(top)).to_numpy(np.int64)
        mapped[codes < 0] = -1
        for i in range(len(top) + 1):  # last index = <OTHER_NGRAM>
            F[f"{nm}_{i}"] = (mapped == i).astype(np.uint8)

    # Exp3 session features
    F["unique_items"] = D.uniq["fixed"]
    F["cart_flag"] = D.cart
    F["is_new_user"] = D.sess_new[D.ev_sess]
    feats = pd.DataFrame(F)

    # context categoricals (Exp2 encoding: MISSING, top-K for high-cardinality, train-only)
    ss = sessions_df.set_index("session_idx")
    tr_sess = ss["split"].eq("train")
    for col in D.session_cat_cols:
        vals = ss["ctx_" + col].fillna("MISSING").astype(str)
        if col in CONTEXT_FIELDS_C:
            top = set(vals[tr_sess & (vals != "MISSING")].value_counts().iloc[:TOP_K].index) | {"MISSING"}
            vals = vals.where(vals.isin(top), "<OTHER>")
        known = sorted(set(vals[tr_sess]))
        fallback = "MISSING" if "MISSING" in known else vals[tr_sess].value_counts().idxmax()
        vals = vals.where(vals.isin(known), fallback)
        enc = {v: i for i, v in enumerate(known)}
        feats["ctx_" + col] = vals.map(enc).to_numpy(np.int32)[D.ev_sess]
    feats.to_parquet(out_path, index=False)
    log(f"LightGBM features: {feats.shape} in {time.time() - t0:.0f}s")
    return feats


def run_lightgbm(D, feats, sessions_df, run_dir, seeds, n_trials=30, tune_frac=1.0,
                 n_jobs=None, timeout_sec=None, log=print):
    import lightgbm as lgb
    import optuna

    n_jobs = n_jobs or os.cpu_count()
    cat_cols = ["ctx_" + c for c in D.session_cat_cols]
    cols = list(feats.columns)
    os.makedirs(os.path.join(run_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "preds"), exist_ok=True)
    w_cls = D.class_weights()
    params_path = os.path.join(run_dir, "runs", "lgbm_best_params.json")

    if not os.path.exists(params_path):
        t0 = time.time()
        tr_s = sessions_df[sessions_df["split"] == "train"].sort_values("session_end_ts")
        cut = int(len(tr_s) * 0.8)
        fit_s = tr_s.iloc[:cut]["session_idx"].to_numpy()
        if tune_frac < 1.0:
            fit_s = np.random.default_rng(42).choice(fit_s, int(len(fit_s) * tune_frac), replace=False)
        hold_s = tr_s.iloc[cut:]["session_idx"].to_numpy()
        tr = D.rows["train"]
        fit_rows = tr[np.isin(D.ev_sess[tr], fit_s)]
        hold_rows = tr[np.isin(D.ev_sess[tr], hold_s)]
        Xf, yf, wf = feats.iloc[fit_rows], D.y[fit_rows], w_cls[D.y[fit_rows]]
        Xh, yh = feats.iloc[hold_rows], D.y[hold_rows]
        log(f"Optuna tuning: fit={len(fit_rows):,} holdout={len(hold_rows):,} prefixes, {n_trials} trials")

        def objective(trial):
            p = {
                "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
                "max_depth": trial.suggest_categorical("max_depth", [-1, 8, 16]),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 50, 100]),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            m = lgb.LGBMClassifier(**p, objective="multiclass", random_state=42, n_jobs=n_jobs, verbose=-1)
            m.fit(Xf, yf, sample_weight=wf, categorical_feature=cat_cols, eval_set=[(Xh, yh)],
                  eval_metric="multi_logloss", callbacks=[lgb.early_stopping(30, verbose=False)])
            f = _mf1(yh, m.predict(Xh))
            log(f"  trial {trial.number}: macro_f1={f:.4f}")
            return f

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=timeout_sec)
        best = dict(study.best_params)
        with open(params_path, "w") as f:
            json.dump({"params": best, "tune_macro_f1": study.best_value,
                       "n_trials_completed": len(study.trials), "tune_frac": tune_frac,
                       "runtime_sec": round(time.time() - t0, 1)}, f, indent=2)
        del Xf, Xh
    with open(params_path) as f:
        tuned = json.load(f)
    best = tuned["params"]

    tr, va, te = D.rows["train"], D.rows["val"], D.rows["test"]
    for seed in seeds:
        tag = f"LightGBM__seed{seed}"
        if os.path.exists(os.path.join(run_dir, "runs", tag + ".json")):
            log(f"[{tag}] already done, skipping")
            continue
        t0 = time.time()
        m = lgb.LGBMClassifier(**best, objective="multiclass", random_state=seed, n_jobs=n_jobs, verbose=-1)
        m.fit(feats.iloc[tr], D.y[tr], sample_weight=w_cls[D.y[tr]], categorical_feature=cat_cols)
        val_f1 = _mf1(D.y[va], m.predict(feats.iloc[va]))
        prob = m.predict_proba(feats.iloc[te])
        test_f1 = _mf1(D.y[te], prob.argmax(1))
        _save_preds(os.path.join(run_dir, "preds", tag + ".parquet"), te, prob)
        info = {"model": "LightGBM", "seed": seed, "params": best, "best_val_macro_f1": val_f1,
                "test_macro_f1": test_f1, "n_features": len(cols),
                "tuning_runtime_sec": tuned["runtime_sec"], "runtime_sec": round(time.time() - t0, 1)}
        with open(os.path.join(run_dir, "runs", tag + ".json"), "w") as f:
            json.dump(info, f, indent=2)
        log(f"[{tag}] val={val_f1:.4f} test={test_f1:.4f} ({(time.time() - t0) / 60:.1f} min)")
