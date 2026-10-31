"""Seed-aware significance tests (CPU only), computed from the combined prediction file
written by evaluate.evaluate_all (`test_predictions_all_models.parquet`).

The paired session bootstrap in evaluate.py resamples test sessions but keeps every
trained model fixed, so it measures test-set noise only. When a model's Macro-F1 varies
a lot from seed to seed, that understates the uncertainty of a comparison. Here:

* Two-level bootstrap: each resample draws test sessions with replacement (shared by all
  models, so comparisons stay paired over the test set) and, independently for every
  model, draws its seeds with replacement. A model's score in a resample is the mean
  Macro-F1 of its drawn seeds on the drawn sessions.
* Seed-level Welch t-test on the per-seed test Macro-F1 values (each seed = one sample).
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

from .evaluate import f1_from_cm

COL = re.compile(r"^(?P<model>.+)__seed(?P<seed>\d+)$")


def load_predictions(path):
    df = pd.read_parquet(path)
    runs = {}
    for c in df.columns:
        m = COL.match(c)
        if m:
            runs.setdefault(m["model"], []).append(c)
    sess, _ = pd.factorize(df["session_id"])
    return df["y_true"].to_numpy(np.int64), sess, df, runs


def seed_bootstrap(path, references, n_boot=1000, boot_seed=2026, models=None):
    y, sess, df, runs = load_predictions(path)
    if models is not None:
        runs = {m: c for m, c in runs.items() if m in models}
    S = int(sess.max()) + 1
    # per (model, seed): session-level confusion counts, flattened 4x4
    cms = {m: np.stack([np.bincount(sess * 16 + y * 4 + df[c].to_numpy(np.int64), minlength=S * 16)
                        .reshape(S, 16).astype(np.float32) for c in cols]) for m, cols in runs.items()}
    rng = np.random.default_rng(boot_seed)
    boot = {m: np.empty(n_boot) for m in runs}
    chunk = 50
    for c0 in range(0, n_boot, chunk):
        k = min(chunk, n_boot - c0)
        W = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(k)]).astype(np.float32)
        for m, cm in cms.items():
            n = cm.shape[0]
            per_seed = f1_from_cm((np.einsum("ks,nsc->knc", W, cm)).reshape(k, n, 4, 4))[0]  # (k, n)
            pick = rng.integers(0, n, size=(k, n))                                            # resample seeds
            boot[m][c0:c0 + k] = np.take_along_axis(per_seed, pick, axis=1).mean(1)

    full = {m: f1_from_cm(cm.sum(1).reshape(-1, 4, 4))[0] for m, cm in cms.items()}  # per-seed test F1
    out = {"n_boot": n_boot, "n_test_sessions": S,
           "models": {m: {"seeds": [int(COL.match(c)["seed"]) for c in runs[m]],
                          "per_seed_macro_f1": [float(v) for v in full[m]],
                          "mean": float(full[m].mean()),
                          "std": float(full[m].std(ddof=1)) if len(full[m]) > 1 else 0.0,
                          "ci95_two_level": [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))]}
                      for m in runs},
           "comparisons": {}}
    for ref in references:
        if ref not in runs:
            continue
        table = {}
        for m in runs:
            if m == ref:
                continue
            d = boot[m] - boot[ref]
            a, b = full[m], full[ref]
            welch = stats.ttest_ind(a, b, equal_var=False) if len(a) > 1 and len(b) > 1 else None
            table[m] = {"delta": float(a.mean() - b.mean()),
                        "ci95_two_level": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "p_two_sided_two_level": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                        "welch_t": None if welch is None else float(welch.statistic),
                        "welch_p": None if welch is None or np.isnan(welch.pvalue) else float(welch.pvalue)}
        out["comparisons"][ref] = table
    return out


def to_markdown(out, title):
    L = [f"# {title}", "",
         f"Two-level bootstrap: {out['n_boot']} resamples of {out['n_test_sessions']:,} test sessions "
         "and, per model, of its seeds. Welch t-test on per-seed test Macro-F1.", "",
         "| Model | Seeds | Macro-F1 mean ± s.d. | Per-seed | 95% CI (two-level) |", "|---|---|---|---|---|"]
    for m, r in sorted(out["models"].items(), key=lambda kv: -kv[1]["mean"]):
        L.append(f"| {m} | {len(r['seeds'])} | {r['mean']:.4f} ± {r['std']:.4f} | "
                 f"{', '.join(f'{v:.3f}' for v in r['per_seed_macro_f1'])} | "
                 f"[{r['ci95_two_level'][0]:.4f}, {r['ci95_two_level'][1]:.4f}] |")
    for ref, tab in out["comparisons"].items():
        L += ["", f"## Δ vs {ref}", "",
              "| Model | Δ | 95% CI (two-level) | p (two-level) | Welch p (seeds) |", "|---|---|---|---|---|"]
        for m, r in sorted(tab.items(), key=lambda kv: -kv[1]["delta"]):
            wp = "–" if r["welch_p"] is None else f"{r['welch_p']:.3f}"
            L.append(f"| {m} | {r['delta']:+.4f} | [{r['ci95_two_level'][0]:+.4f}, {r['ci95_two_level'][1]:+.4f}] | "
                     f"{r['p_two_sided_two_level']:.3f} | {wp} |")
    return "\n".join(L) + "\n"


def run(path, out_dir, references, title, n_boot=1000):
    out = seed_bootstrap(path, references, n_boot=n_boot)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "seed_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)
    md = to_markdown(out, title)
    with open(os.path.join(out_dir, "seed_bootstrap.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return out, md
