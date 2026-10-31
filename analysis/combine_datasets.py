"""Combine the seed-aware results of both datasets into one table and a set of key contrasts.

    python analysis/combine_datasets.py

Reads results/exp4/seedstats_stage{4,5}/ and results/exp5/seedstats_stage{4,5}/ and writes
results/combined/combined_seedstats.{md,json}. Stage 5 (learning rate chosen on validation)
is the primary result; stage 4 (fixed lr = 1e-3) is shown for comparison.
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATASETS = {"GA4": "exp4", "Coveo": "exp5"}
SEQ = ["SASRec+Time+SF+PageItem", "GRU+Time+SF+PageItem", "SASRec+Time+SF", "SASRec+Time"]
GRAPH = ["HGNN_full_3layer_meanpool", "HGNN_full_meanpool", "HGNN_full_attnpool", "HGNN_full",
         "HomoGraphSAGE_3layer_meanpool", "HomoGraphSAGE_meanpool", "HomoGraphSAGE+PageItemFeat",
         "HGNN_no_page_item", "HGNN_no_page_item_no_SF"]
SHARED = ["SASRec+Time+SF+PageItem", "HGNN_full_3layer_meanpool", "HomoGraphSAGE_3layer_meanpool",
          "GRU+Time+SF+PageItem", "LightGBM", "HGNN_full", "Markov-3"]


def load(exp, stage):
    with open(os.path.join(ROOT, "results", exp, f"seedstats_{stage}", "seed_bootstrap.json")) as f:
        return json.load(f)


def contrast(res, a, b):
    """a - b, from the comparison table that uses b as reference (or the mirrored one)."""
    c = res["comparisons"]
    if b in c and a in c[b]:
        r = c[b][a]
        return r["delta"], r["ci95_two_level"], r["welch_p"]
    r = c[a][b]
    return -r["delta"], [-r["ci95_two_level"][1], -r["ci95_two_level"][0]], r["welch_p"]


def best(res, pool):
    return max((m for m in pool if m in res["models"]), key=lambda m: res["models"][m]["mean"])


def verdict(ci):
    return "sig. >0" if ci[0] > 0 else ("sig. <0" if ci[1] < 0 else "n.s.")


out = {"models": {}, "contrasts": {}}
L = ["# Combined GA4 + Coveo: seed-aware results", "",
     "Stage 5 = learning rate chosen on validation (primary); stage 4 = fixed lr 1e-3. "
     "CI = two-level bootstrap (test sessions and seeds); Welch p = seed-level t-test.", "",
     "## Test Macro-F1 (mean ± s.d. over seeds), stage 5", "",
     "| Model | GA4 | Coveo |", "|---|---|---|"]
res = {(d, st): load(e, st) for d, e in DATASETS.items() for st in ("stage4", "stage5")}
for m in SHARED:
    cells = []
    for d in DATASETS:
        r = res[(d, "stage5")]["models"].get(m)
        cells.append("–" if r is None else f"{r['mean']:.3f} ± {r['std']:.3f} (n={len(r['seeds'])})")
        out["models"].setdefault(m, {})[d] = r
    L.append(f"| {m} | {cells[0]} | {cells[1]} |")

L += ["", "## Key contrasts", "",
      "| Contrast | Dataset | Stage | Pair | Δ | 95% CI (two-level) | Welch p | Verdict |",
      "|---|---|---|---|---|---|---|---|"]
for d in DATASETS:
    for st in ("stage5", "stage4"):
        r = res[(d, st)]
        bs, bg = best(r, SEQ), best(r, GRAPH)
        top = best(r, list(r["models"]))
        pairs = [("Heterogeneity (pooled readout)", "HGNN_full_3layer_meanpool", "HomoGraphSAGE_3layer_meanpool"),
                 ("Best sequence − best graph", bs, bg),
                 ("Original HGNN − pooled HGNN", "HGNN_full", "HGNN_full_3layer_meanpool"),
                 ("LightGBM − best model", "LightGBM", top),
                 ("GRU − SASRec (same inputs)", "GRU+Time+SF+PageItem", "SASRec+Time+SF+PageItem")]
        for name, a, b in pairs:
            if a == b:
                continue
            delta, ci, wp = contrast(r, a, b)
            out["contrasts"].setdefault(name, {}).setdefault(d, {})[st] = {
                "a": a, "b": b, "delta": delta, "ci95_two_level": ci, "welch_p": wp, "verdict": verdict(ci)}
            wps = "–" if wp is None else f"{wp:.3f}"
            L.append(f"| {name} | {d} | {st} | {a} − {b} | {delta:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {wps} | {verdict(ci)} |")

os.makedirs(os.path.join(ROOT, "results", "combined"), exist_ok=True)
with open(os.path.join(ROOT, "results", "combined", "combined_seedstats.json"), "w") as f:
    json.dump(out, f, indent=2)
with open(os.path.join(ROOT, "results", "combined", "combined_seedstats.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
