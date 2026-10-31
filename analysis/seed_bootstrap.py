"""Seed-aware significance analysis (CPU only).

    python analysis/seed_bootstrap.py <test_predictions_all_models.parquet> <out_dir> \
        [--ref HGNN_full --ref GRU+Time+SF+PageItem ...] [--title "..."] [--n-boot 1000]

Reads the combined prediction file written by stage 4 / stage 5 (`results*/
test_predictions_all_models.parquet`) and writes seed_bootstrap.json and seed_bootstrap.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from exp4lib import seedstats  # noqa: E402

DEFAULT_REFS = ["HGNN_full", "GRU+Time+SF+PageItem", "SASRec+Time+SF+PageItem",
                "HGNN_full_3layer_meanpool", "HomoGraphSAGE_3layer_meanpool"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("out_dir")
    ap.add_argument("--ref", action="append", help="reference model (repeatable)")
    ap.add_argument("--title", default="Seed-aware comparison")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    _, md = seedstats.run(a.predictions, a.out_dir, a.ref or DEFAULT_REFS, a.title, n_boot=a.n_boot)
    print(md)
