"""Smoke test of the seed-aware bootstrap on synthetic predictions with known structure.

    python tests/smoke_test_seedstats.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from exp4lib import seedstats  # noqa: E402

rng = np.random.default_rng(0)
n_sess, per = 4000, 5
sid = np.repeat(np.arange(n_sess), per)
y = np.repeat(rng.choice(4, n_sess, p=[.7, .15, .1, .05]), per)


def predict(acc):
    wrong = rng.random(len(y)) > acc
    return np.where(wrong, rng.integers(0, 4, len(y)), y).astype(np.int8)


df = pd.DataFrame({"session_id": sid.astype(str), "prefix_len": np.tile(np.arange(1, per + 1), n_sess),
                   "y_true": y, "undetermined": False})
for s, acc in zip([42, 123, 456], [.80, .80, .80]):           # stable
    df[f"Stable__seed{s}"] = predict(acc)
for s, acc in zip([42, 123, 456], [.92, .66, .82]):           # same mean, unstable seeds
    df[f"Unstable__seed{s}"] = predict(acc)
for s, acc in zip([42, 123, 456], [.88, .88, .88]):           # clearly better
    df[f"Better__seed{s}"] = predict(acc)

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "test_predictions_all_models.parquet")
    df.to_parquet(p, index=False)
    out, md = seedstats.run(p, d, ["Stable"], "synthetic", n_boot=400)
print(md)
cmp = out["comparisons"]["Stable"]
w_unstable = np.diff(cmp["Unstable"]["ci95_two_level"])[0]
w_better = np.diff(cmp["Better"]["ci95_two_level"])[0]
assert cmp["Better"]["ci95_two_level"][0] > 0, "clearly better model must be significant"
assert cmp["Unstable"]["ci95_two_level"][0] < 0 < cmp["Unstable"]["ci95_two_level"][1], "unstable model must not be significant"
assert w_unstable > 3 * w_better, "seed instability must widen the interval"
print(f"SMOKE TEST SEEDSTATS OK | CI width unstable={w_unstable:.3f} vs stable-seed model={w_better:.3f}")
