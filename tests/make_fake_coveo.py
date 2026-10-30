"""Synthetic data in the exact Coveo SIGIR eCom 2021 schema (column layout as in the public sample files),
zipped like the original release. Used by smoke_test_coveo.py; contains no real data."""
import os, sys, zipfile, hashlib, numpy as np, pandas as pd
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
h = lambda s: hashlib.sha256(str(s).encode()).hexdigest()
rows = []
t0 = 1_550_000_000_000
for s in range(3000):
    sid = h(f"s{s}"); ts = t0 + int(rng.integers(0, 40 * 86_400_000)); L = int(rng.integers(1, 60))
    buyer = rng.random() < 0.05
    for k in range(L):
        if rng.random() > 0.3: ts += int(rng.integers(500, 120_000))
        r = rng.random()
        if r < 0.45:
            rows.append((sid, "pageview", "", "", ts, h(f"u{rng.integers(0, 200)}")))
        else:
            act = rng.choice(["detail", "add", "remove", "purchase"], p=[.8, .1, .04, .06] if buyer else [.9, .07, .025, .005])
            sku = h(f"p{rng.integers(0, 300)}")
            rows.append((sid, "event_product", act, sku, ts, h(f"u{rng.integers(0, 200)}")))
            if act == "detail" and rng.random() < .7:  # PDP: detail + pageview at the same timestamp
                rows.append((sid, "pageview", "", "", ts, rows[-1][5]))
df = pd.DataFrame(rows, columns=["session_id_hash", "event_type", "product_action", "product_sku_hash",
                                 "server_timestamp_epoch_ms", "hashed_url"]).sample(frac=1, random_state=0)
df.to_csv(os.path.join(out, "browsing_train.csv"), index=False)
skus = pd.DataFrame({"product_sku_hash": [h(f"p{i}") for i in range(300)]})
skus["description_vector"] = "[0.1, 0.2]"; skus["category_hash"] = "a/b"; skus["image_vector"] = "[1.0]"
skus["price_bucket"] = rng.integers(1, 11, len(skus)).astype(float); skus.loc[::17, "price_bucket"] = np.nan
skus.to_csv(os.path.join(out, "sku_to_content.csv"), index=False)
with zipfile.ZipFile(os.path.join(out, "SIGIR-ecom-data-challenge.zip"), "w") as z:
    for f in ("browsing_train.csv", "sku_to_content.csv"):
        z.write(os.path.join(out, f), "train/" + f)
for f in ("browsing_train.csv", "sku_to_content.csv"): os.remove(os.path.join(out, f))
print(len(df), "rows")
