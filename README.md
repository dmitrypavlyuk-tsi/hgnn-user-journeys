# hgnn-user-journeys

A controlled comparison of heterogeneous graph neural networks (HGNNs), sequence models and gradient
boosting for **early purchase-intent prediction** from e-commerce clickstreams. Each prefix of an ongoing
session (its first *t* events) is classified into one of four outcome classes: Browser, Researcher, Intent
or Buyer. Every model is trained on the same data and scored on exactly the same test prefixes. A paired
session-level bootstrap tests the differences, and an audit checks the feature joins for look-ahead leakage.

This repository holds the code, notebooks and aggregate results. It extends the master's
thesis experiments (Exp0–Exp3, https://github.com/dgizdevans/master) with:

- **Exp4**: a controlled re-run on the GA4 dataset with 16 model configurations, readout/depth ablations and a
  leakage audit.
- **Exp5**: the same focused comparison on a second retailer (Coveo SIGIR eCom 2021).

## Repository layout

```
src/exp4lib/            shared library: data preparation (GA4, Coveo), batching, models, training, baselines, evaluation
notebooks/exp4_controlled/
    exp4_ControlledComparison.ipynb   standalone Colab notebook (GA4 / BigQuery)
    build_notebook.py                 regenerates the notebook from src/exp4lib
notebooks/exp5_coveo/
    exp5_Coveo.ipynb                  standalone Colab notebook (Coveo)
    build_notebook.py
results/exp4/           aggregate results: summary.md, results.json
tests/                  CPU smoke tests on synthetic data (no real data needed)
```

The notebooks are standalone: they embed `src/exp4lib` as `%%writefile` cells, so you only need to upload
the `.ipynb` to Colab. After you edit `src/exp4lib`, run `python notebooks/<exp>/build_notebook.py`.

## Data sources

No data is included in this repository.

| Dataset | Access | Used in |
|---|---|---|
| Google Analytics 4 obfuscated e-commerce sample (Google Merchandise Store, Nov 2020 – Jan 2021) | Public BigQuery dataset `bigquery-public-data.ga4_obfuscated_sample_ecommerce`. You need a Google Cloud project to run queries (a free BigQuery sandbox project is enough). | Exp4 |
| Coveo SIGIR eCom 2021 Data Challenge | Free for research and education after registering at https://www.coveo.com/en/ailabs/sigir-ecom-data-challenge and accepting the Terms and Conditions. Do not redistribute the data. | Exp5 |

## How to run (Google Colab)

1. Upload the notebook to Colab and choose a GPU runtime (L4/A100 recommended, T4 works).
2. Set the configuration cell. For Exp4, set `BQ_PROJECT` to your Google Cloud project. For Exp5, first upload
   the Coveo zip unchanged to `MyDrive/exp5_coveo/raw/`.
3. Run the cells in order. Artifacts, checkpoints and results are written to Google Drive, and every stage
   can be resumed: finished outputs are skipped, and an interrupted training run resumes from its last epoch.

Stages:

| Stage | Exp4 (GA4) | Exp5 (Coveo) |
|---|---|---|
| 1 Data | BigQuery extraction, exact reproduction of the Exp3 counts, item-join leakage audit | Unzip, DuckDB sampling (≈300k sessions), prefix tables |
| 2 Cheap baselines | Markov-3, LightGBM (Optuna) | same |
| 3 Neural models | 8 models + leaky-join reference | 5 models (focused subset) |
| 3b / 3c (Exp4 only) | HGNN readout/depth variants; homogeneous graph with pooled readout | – |
| 4 Evaluation | Macro-F1 ± s.d., per-class F1, paired session bootstrap, prefix-length bins, undetermined prefixes | same |

Each notebook folder has a README with more detail and rough runtimes.

### Local smoke tests (CPU, synthetic data)

```bash
pip install -r requirements.txt
python tests/smoke_test_exp4.py
python tests/smoke_test_coveo.py
```

## Results

Aggregate Exp4 test results (Macro-F1 mean ± s.d. over seeds, per-class F1, paired bootstrap tables,
prefix-length bins) are in `results/exp4/summary.md` and `results/exp4/results.json`. Exp5 (Coveo) is
pending.

## How to cite

Please cite the software using `CITATION.cff`. A paper describing this work is in preparation.

## License

The code is released under the MIT License (see `LICENSE`). The datasets are not included and remain
subject to their own terms: the Google BigQuery public dataset terms, and the Coveo SIGIR eCom 2021 Terms
and Conditions.
