"""Stage 4: metrics, paired session-level bootstrap, prefix-length bins,
undetermined subset, summary tables."""
import glob
import json
import os

import numpy as np
import pandas as pd

from .data import CLASS_ORDER

BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 15), (16, 20), (21, 10**6)]
BIN_LABELS = ["1", "2", "3", "4", "5", "6-7", "8-10", "11-15", "16-20", "21+"]
REFERENCE = "HGNN_full"
SECOND_REFERENCES = [  # extra paired-bootstrap baselines
    "GRU+Time+SF+PageItem",            # best sequence model (stage 3b question)
    "HomoGraphSAGE_3layer_meanpool",   # hetero vs homo under the pooled readout (stage 3c)
]


def f1_from_cm(cm):
    """Macro-F1 (and per-class F1) from confusion matrices [..., 4, 4] (rows=true)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(-2) - tp
    fn = cm.sum(-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    return f1.mean(-1), f1


def cm_of(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return np.bincount(y * 4 + p, minlength=16).reshape(4, 4).astype(np.float64)


def load_runs(run_dir):
    """{model: {seed: (info, preds_df)}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "runs", "*__seed*.json"))):
        tag = os.path.basename(path)[:-5]
        pp = os.path.join(run_dir, "preds", tag + ".parquet")
        if not os.path.exists(pp):
            continue
        with open(path) as f:
            info = json.load(f)
        out.setdefault(info["model"], {})[int(info["seed"])] = (info, pd.read_parquet(pp))
    return out


def evaluate_all(D, run_dir, out_dir, n_boot=1000, boot_seed=2026, reference=REFERENCE,
                 extra_references=SECOND_REFERENCES, title=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    runs = load_runs(run_dir)
    if not runs:
        raise RuntimeError("No finished runs found.")
    test = D.rows["test"]
    y = D.y[test]
    plen = D.prefix_len[test]
    und = D.undetermined[test]
    sid_codes, sid_uni = pd.factorize(D.session_id[test])
    S = len(sid_uni)

    # ---- per model / per seed metrics -------------------------------------------------
    results = {}
    preds_by = {}
    for model, seeds in runs.items():
        per_seed = {}
        for seed, (info, df) in sorted(seeds.items()):
            assert np.array_equal(df["row"].to_numpy(), test), f"{model} seed {seed}: test rows differ"
            p = df["pred"].to_numpy(np.int64)
            preds_by[(model, seed)] = p
            mf, pc = f1_from_cm(cm_of(y, p))
            bins = {}
            for (lo, hi), lab in zip(BINS, BIN_LABELS):
                bins[lab] = f1_from_cm(cm_of(y, p, (plen >= lo) & (plen <= hi)))[0]
            per_seed[seed] = {
                "macro_f1": mf, "per_class_f1": dict(zip(CLASS_ORDER, pc.tolist())),
                "bins": bins, "undetermined_macro_f1": f1_from_cm(cm_of(y, p, und))[0],
                "determined_macro_f1": f1_from_cm(cm_of(y, p, ~und))[0],
                "best_val_macro_f1": info.get("best_val_macro_f1"),
                "best_epoch": info.get("best_epoch"), "runtime_sec": info.get("runtime_sec"),
            }
        mfs = np.array([v["macro_f1"] for v in per_seed.values()])
        mean = float(mfs.mean())
        rep = min(per_seed, key=lambda s: abs(per_seed[s]["macro_f1"] - mean))

        def agg(getter):
            vals = np.array([getter(v) for v in per_seed.values()], dtype=float)
            return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}

        results[model] = {
            "n_seeds": len(per_seed), "seeds": sorted(per_seed), "representative_seed": int(rep),
            "macro_f1": agg(lambda v: v["macro_f1"]),
            "per_class_f1": {c: agg(lambda v, c=c: v["per_class_f1"][c]) for c in CLASS_ORDER},
            "bins": {b: agg(lambda v, b=b: v["bins"][b]) for b in BIN_LABELS},
            "undetermined_macro_f1": agg(lambda v: v["undetermined_macro_f1"]),
            "determined_macro_f1": agg(lambda v: v["determined_macro_f1"]),
            "runtime_sec_total": float(sum(v["runtime_sec"] or 0 for v in per_seed.values())),
            "per_seed": per_seed,
        }

    # ---- paired bootstrap over test sessions ------------------------------------------
    keys = list(preds_by)
    sess_cm = {}
    for k in keys:
        idx = sid_codes * 16 + y * 4 + preds_by[k]
        sess_cm[k] = np.bincount(idx, minlength=S * 16).reshape(S, 16).astype(np.float32)
    boot = {k: np.empty(n_boot) for k in keys}
    rng = np.random.default_rng(boot_seed)
    chunk = 100
    for c0 in range(0, n_boot, chunk):
        m = min(chunk, n_boot - c0)
        W = rng.multinomial(S, np.full(S, 1.0 / S), size=m).astype(np.float32)  # resampled session counts
        for k in keys:
            cm = (W @ sess_cm[k]).reshape(m, 4, 4)
            boot[k][c0:c0 + m] = f1_from_cm(cm)[0]
    log(f"Bootstrap done: {n_boot} resamples of {S:,} test sessions")

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    def deltas_vs(refname):
        table = {}
        if refname not in results:
            return table
        ref = results[refname]
        ref_rep = boot[(refname, ref["representative_seed"])]
        ref_mean = np.mean([boot[(refname, s)] for s in ref["seeds"]], axis=0)
        for model, r in results.items():
            if model == refname:
                continue
            d_rep = boot[(model, r["representative_seed"])] - ref_rep
            d_avg = np.mean([boot[(model, s)] for s in r["seeds"]], axis=0) - ref_mean
            pt_rep = r["per_seed"][r["representative_seed"]]["macro_f1"] - ref["per_seed"][ref["representative_seed"]]["macro_f1"]
            pt_avg = r["macro_f1"]["mean"] - ref["macro_f1"]["mean"]
            table[model] = {
                "delta_rep_seed": pt_rep, "ci95_rep_seed": ci(d_rep),
                "p_delta_ge_0_rep": float((d_rep >= 0).mean()),
                "delta_seed_avg": pt_avg, "ci95_seed_avg": ci(d_avg),
                "p_delta_ge_0_avg": float((d_avg >= 0).mean()),
            }
        return table

    boot_table = deltas_vs(reference)
    boot_all = {r: deltas_vs(r) for r in [reference] + list(extra_references) if r in results}
    for model, r in results.items():
        r["macro_f1_ci95_rep_seed"] = ci(boot[(model, r["representative_seed"])])

    dataset = D.meta.get("dataset", "GA4 obfuscated e-commerce sample")
    out = {"title": title or f"Controlled comparison ({dataset}): test results", "dataset": dataset,
           "reference_model": reference, "n_boot": n_boot, "n_test_sessions": int(S),
           "n_test_prefixes": int(len(test)), "n_undetermined_test_prefixes": int(und.sum()),
           "models": results, "bootstrap_vs_reference": boot_table, "bootstrap_vs": boot_all,
           "leakage": D.meta.get("leakage"), "repro_all_pass": D.meta.get("repro_all_pass")}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---- combined predictions parquet --------------------------------------------------
    comb = pd.DataFrame({"session_id": D.session_id[test], "prefix_len": plen, "y_true": y,
                         "undetermined": und})
    for (model, seed), p in preds_by.items():
        comb[f"{model}__seed{seed}"] = p.astype(np.int8)
    comb.to_parquet(os.path.join(out_dir, "test_predictions_all_models.parquet"), index=False)

    md = summary_markdown(out)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return out, md


def _fmt(a):
    return f"{a['mean']:.4f} ± {a['std']:.4f}"


def summary_markdown(out):
    res = out["models"]
    order = sorted(res, key=lambda m: -res[m]["macro_f1"]["mean"])
    L = [f"# {out.get('title', 'Controlled comparison: test results')}", "",
         f"Dataset: {out.get('dataset', 'GA4 obfuscated e-commerce sample')}.", "",
         f"Test set: {out['n_test_prefixes']:,} prefixes from {out['n_test_sessions']:,} sessions. "
         f"Mean ± std over seeds. Bootstrap: {out['n_boot']} resamples of test sessions, "
         f"deltas vs `{out['reference_model']}`.", "",
         "## Macro-F1 and per-class F1", "",
         "| Model | Seeds | Macro-F1 | Browser | Researcher | Intent | Buyer | Undetermined | Runtime (min) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = res[m]
        pc = r["per_class_f1"]
        L.append(f"| {m} | {r['n_seeds']} | {_fmt(r['macro_f1'])} | {pc['Browser']['mean']:.4f} | "
                 f"{pc['Researcher']['mean']:.4f} | {pc['Intent']['mean']:.4f} | {pc['Buyer']['mean']:.4f} | "
                 f"{_fmt(r['undetermined_macro_f1'])} | {r['runtime_sec_total'] / 60:.1f} |")
    tables = out.get("bootstrap_vs") or {out["reference_model"]: out["bootstrap_vs_reference"]}
    for refname, tab in tables.items():
        if not tab:
            continue
        L += ["", f"## Paired bootstrap: delta Macro-F1 vs {refname}", "",
              "| Model | Δ (rep. seed) | 95% CI | Δ (seed-avg) | 95% CI | P(Δ ≥ 0), seed-avg |",
              "|---|---|---|---|---|---|"]
        for m in order:
            b = tab.get(m)
            if b is None:
                continue
            L.append(f"| {m} | {b['delta_rep_seed']:+.4f} | [{b['ci95_rep_seed'][0]:+.4f}, {b['ci95_rep_seed'][1]:+.4f}] | "
                     f"{b['delta_seed_avg']:+.4f} | [{b['ci95_seed_avg'][0]:+.4f}, {b['ci95_seed_avg'][1]:+.4f}] | "
                     f"{b['p_delta_ge_0_avg']:.3f} |")
    L += ["", "## Macro-F1 by prefix length (mean over seeds)", "",
          "| Model | " + " | ".join(BIN_LABELS) + " |", "|---|" + "---|" * len(BIN_LABELS)]
    for m in order:
        L.append(f"| {m} | " + " | ".join(f"{res[m]['bins'][b]['mean']:.3f}" for b in BIN_LABELS) + " |")
    lk = out.get("leakage")
    if lk:
        L += ["", "## Leakage check (Exp3 item join on timestamp + bundle id)", "",
              f"- Events sharing a timestamp with another event of the same session: "
              f"{lk['events_sharing_timestamp_within_session']:.1%}"]
        for k in ("unique_items_sees_future_items", "unique_items_value_differs",
                  "graph_has_future_item_edges", "graph_has_future_add_to_cart_or_purchase_items"):
            v = lk[k]
            L.append(f"- `{k}`: {v['all']:,} of {lk['prefixes_total']:,} prefixes "
                     f"(test: {v['by_split']['test']:,})")
        L.append(f"- Exp3 graph edges not from the item's own event: {lk['exp3_edges_not_from_own_event']:,}; "
                 f"with the wrong relation type: {lk['exp3_edges_with_wrong_relation_type']:,}")
    return "\n".join(L) + "\n"
