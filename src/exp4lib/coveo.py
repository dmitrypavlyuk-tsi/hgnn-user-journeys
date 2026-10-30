"""Exp5 data stage: the Coveo SIGIR eCom 2021 Data Challenge dataset, prepared into
exactly the same tables as data.build_dataset() so that every Exp4 model, the
training loop and the evaluation can be reused unchanged.

Mapping to the Exp4 protocol
----------------------------
* Event token: product_action (detail / add / remove / purchase), or "pageview"
  for page views without a product.
* Page = hashed_url; Item = product_sku_hash (one item per product event, so the
  item join is per-event by construction: no timestamp join, no leakage).
* Labels (full session, priority order): Buyer = purchase; Intent = add (no
  purchase); Researcher = >= 3 detail events; Browser = the rest.
* Session context (prefix-safe, known at the first event): start hour and start
  weekday. Coveo has no device/traffic fields; timestamps are shifted by whole
  weeks, which keeps intra-week patterns.
* Sample: the N most recent sessions that start at least `end_gap_days` before
  the last session start (a contiguous period). T_MAX = p95 of training session
  lengths, then the Exp3 p70/p85 temporal split by session end.
"""
import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd

from .data import CLASS_ORDER, LABEL_MAP

EVENT_TOKENS = ["add", "detail", "pageview", "purchase", "remove"]
ITEM_RELATIONS = ["detail_of", "add_of", "remove_of", "purchase_of"]
REL_OF_TOKEN = {"detail": 0, "add": 1, "remove": 2, "purchase": 3}
KEY_EVENTS = ["purchase", "add", "remove"]
SESSION_CAT_COLS = ["start_hour", "start_weekday"]
NEEDED_FILES = ["browsing_train.csv", "sku_to_content.csv"]


def locate_raw(raw_dir, work_dir):
    """Return paths to browsing_train.csv and sku_to_content.csv, extracting them from a
    zip in raw_dir (to fast local disk) if they are not already available."""
    os.makedirs(work_dir, exist_ok=True)
    found = {}
    for name in NEEDED_FILES:
        for d in (work_dir, raw_dir):
            hits = glob.glob(os.path.join(d, "**", name), recursive=True)
            if hits:
                found[name] = hits[0]
                break
    missing = [n for n in NEEDED_FILES if n not in found]
    if missing:
        zips = sorted(glob.glob(os.path.join(raw_dir, "*.zip")))
        if not zips:
            raise FileNotFoundError(f"No {missing} and no .zip in {raw_dir}")
        with zipfile.ZipFile(zips[0]) as z:
            for member in z.namelist():
                base = os.path.basename(member)
                if base in missing:
                    print(f"Extracting {member} from {os.path.basename(zips[0])} ...")
                    target = os.path.join(work_dir, base)
                    with z.open(member) as src, open(target, "wb") as dst:
                        while True:
                            buf = src.read(1 << 24)
                            if not buf:
                                break
                            dst.write(buf)
                    found[base] = target
    missing = [n for n in NEEDED_FILES if n not in found]
    if missing:
        raise FileNotFoundError(f"Could not find {missing} in {raw_dir} or its zip")
    return found["browsing_train.csv"], found["sku_to_content.csv"]


def _browsing_view(con, path):
    con.execute(f"""
        CREATE OR REPLACE VIEW browsing AS
        SELECT * FROM read_csv('{path}', header=true, columns={{
            'session_id_hash': 'VARCHAR', 'event_type': 'VARCHAR', 'product_action': 'VARCHAR',
            'product_sku_hash': 'VARCHAR', 'server_timestamp_epoch_ms': 'BIGINT',
            'hashed_url': 'VARCHAR'}})
    """)


def extract_sample(browsing_csv, sku_csv, n_sessions, end_gap_days=1.0):
    """Select a contiguous sample of sessions with DuckDB (streams the 36M-row file)."""
    import duckdb
    con = duckdb.connect()
    _browsing_view(con, browsing_csv.replace("\\", "/"))
    con.execute("""
        CREATE TEMP TABLE sess AS
        SELECT session_id_hash, MIN(server_timestamp_epoch_ms) AS start_ms, COUNT(*) AS n
        FROM browsing GROUP BY 1
    """)
    total_sessions, max_start = con.execute("SELECT COUNT(*), MAX(start_ms) FROM sess").fetchone()
    cutoff = int(max_start - end_gap_days * 86_400_000)
    con.execute(f"""
        CREATE TEMP TABLE chosen AS
        SELECT session_id_hash FROM sess WHERE start_ms <= {cutoff}
        ORDER BY start_ms DESC, session_id_hash LIMIT {int(n_sessions)}
    """)
    ev = con.execute("""
        SELECT b.session_id_hash, b.event_type, b.product_action, b.product_sku_hash,
               b.server_timestamp_epoch_ms AS ts_ms, b.hashed_url,
               hash(b.session_id_hash, b.event_type, b.product_action, b.product_sku_hash,
                    b.server_timestamp_epoch_ms, b.hashed_url) AS row_hash
        FROM browsing b JOIN chosen c USING (session_id_hash)
    """).df()
    sku = con.execute(f"""
        SELECT product_sku_hash, category_hash, price_bucket
        FROM read_csv('{sku_csv.replace(chr(92), "/")}', header=true, columns={{
            'product_sku_hash': 'VARCHAR', 'description_vector': 'VARCHAR',
            'category_hash': 'VARCHAR', 'image_vector': 'VARCHAR', 'price_bucket': 'DOUBLE'}})
        WHERE product_sku_hash IN (SELECT DISTINCT product_sku_hash FROM browsing
                                   JOIN chosen USING (session_id_hash))
    """).df()
    info = {"total_sessions_in_file": int(total_sessions), "sampled_sessions": int(ev["session_id_hash"].nunique()),
            "sample_cutoff_start_ms": cutoff, "end_gap_days": end_gap_days}
    return ev, sku, info


def build_dataset(raw_dir, work_dir, out_dir, n_sessions=300_000, end_gap_days=1.0):
    browsing_csv, sku_csv = locate_raw(raw_dir, work_dir)
    ev, sku, info = extract_sample(browsing_csv, sku_csv, n_sessions, end_gap_days)
    print(f"Sampled {info['sampled_sessions']:,} of {info['total_sessions_in_file']:,} sessions, "
          f"{len(ev):,} events")

    act = ev["product_action"].fillna("").str.strip()
    ev["event_name"] = np.where(act == "", "pageview", act)
    ev["session_id"] = ev["session_id_hash"]
    ev = ev.sort_values(["session_id", "ts_ms", "event_name", "row_hash"], kind="mergesort").reset_index(drop=True)
    ev["event_index"] = ev.groupby("session_id").cumcount() + 1

    sess = ev.groupby("session_id").agg(session_length=("event_index", "max"),
                                        session_start_ts=("ts_ms", "min"),
                                        session_end_ts=("ts_ms", "max")).reset_index()
    flags = pd.DataFrame({"session_id": ev["session_id"], "buy": ev["event_name"].eq("purchase"),
                          "add": ev["event_name"].eq("add"),
                          "det": ev["event_name"].eq("detail").astype(np.int32)}
                         ).groupby("session_id").agg(buy=("buy", "any"), add=("add", "any"), det=("det", "sum"))
    y = np.where(flags["buy"], "Buyer", np.where(flags["add"], "Intent",
                 np.where(flags["det"] >= 3, "Researcher", "Browser")))
    sess = sess.merge(pd.DataFrame({"session_id": flags.index, "y": y}), on="session_id")

    def split_of(frame):
        p70 = int(round(frame["session_end_ts"].quantile(0.70)))
        p85 = int(round(frame["session_end_ts"].quantile(0.85)))
        return np.where(frame["session_end_ts"] <= p70, "train",
                        np.where(frame["session_end_ts"] <= p85, "val", "test")), p70, p85

    prov, _, _ = split_of(sess)
    t_max = int(np.percentile(sess.loc[prov == "train", "session_length"], 95))
    n_before = len(sess)
    sess = sess[sess["session_length"] <= t_max].reset_index(drop=True)
    sess["split"], p70, p85 = split_of(sess)
    print(f"T_MAX = {t_max} (p95 of train lengths); kept {len(sess):,} of {n_before:,} sessions")

    keep = set(sess["session_id"])
    ev = ev[ev["session_id"].isin(keep)].reset_index(drop=True)
    sess = sess.sort_values("session_id").reset_index(drop=True)
    sess["session_idx"] = np.arange(len(sess), dtype=np.int32)
    ev = ev.merge(sess[["session_id", "session_idx", "split", "y"]], on="session_id", how="left")
    sess["start"] = np.r_[0, np.cumsum(sess["session_length"].to_numpy())[:-1]].astype(np.int64)
    n = len(ev)
    assert n == int(sess["session_length"].sum())
    train = ev["split"].eq("train").to_numpy()

    # time features (seconds), Exp3-style z-score and Exp2-style clipped [0, 1]
    dt = ev.groupby("session_idx")["ts_ms"].diff().fillna(0).to_numpy(np.float64) / 1000.0
    dt_mean, dt_std = float(dt[train].mean()), float(pd.Series(dt[train]).std())
    pos = dt[train & (ev["event_index"].to_numpy() > 1)]
    max_dt = float(np.percentile(pos[pos > 0], 99))

    tr = ev[train]
    event_vocab = {k: i + 2 for i, k in enumerate(sorted(tr["event_name"].unique()))}
    event_vocab.update({"<PAD>": 0, "<UNK>": 1})
    page_vocab = {k: i + 2 for i, k in enumerate(sorted(tr["hashed_url"].dropna().unique()))}
    page_vocab.update({"<PAD>": 0, "<OOV>": 1})

    # items: one per product event that carries a SKU
    is_item = ev["event_name"].isin(list(REL_OF_TOKEN)) & ev["product_sku_hash"].notna()
    items = pd.DataFrame({"event_row": np.flatnonzero(is_item.to_numpy()).astype(np.int64),
                          "item_id": ev.loc[is_item, "product_sku_hash"].to_numpy(),
                          "rel": ev.loc[is_item, "event_name"].map(REL_OF_TOKEN).to_numpy(np.int8)})
    item_train = train[items["event_row"].to_numpy()]
    item_vocab = {k: i + 2 for i, k in enumerate(sorted(items.loc[item_train, "item_id"].unique()))}
    item_vocab.update({"<PAD>": 0, "<OOV>": 1})
    items["item_idx"] = items["item_id"].map(item_vocab).fillna(1).astype(np.int32)
    price = items["item_id"].map(sku.set_index("product_sku_hash")["price_bucket"])
    p_mean, p_std = float(price[item_train].mean()), float(price[item_train].std())
    items["price_norm"] = (price.fillna(0) - p_mean) / p_std
    ip = items[item_train].groupby("item_idx")["price_norm"].mean()
    price_lookup = np.zeros(len(item_vocab), dtype=np.float32)
    price_lookup[ip.index.to_numpy()] = ip.to_numpy(np.float32)
    price_lookup[[0, 1]] = 0.0

    # session context: start hour / weekday (UTC of shifted timestamps)
    start_dt = pd.to_datetime(sess["session_start_ts"], unit="ms", utc=True)
    sess["ctx_start_hour"] = start_dt.dt.hour.astype(str).to_numpy()
    sess["ctx_start_weekday"] = start_dt.dt.weekday.astype(str).to_numpy()
    cat_vocabs = {}
    tr_sess = sess["split"].eq("train").to_numpy()
    for col in SESSION_CAT_COLS:
        vals = sess["ctx_" + col]
        vocab = {v: i + 1 for i, v in enumerate(sorted(set(vals[tr_sess])))}
        cat_vocabs[col] = vocab
        sess[col + "_idx"] = vals.map(vocab).fillna(0).astype(np.int32).to_numpy()
    sess["is_new_user"] = np.int8(0)  # not available in Coveo; constant

    # prefix-level arrays (prefix row == event row)
    sess_of_row = ev["session_idx"].to_numpy(np.int64)
    df = pd.DataFrame({"s": sess_of_row[items["event_row"].to_numpy()], "item": items["item_id"].to_numpy(),
                       "row": items["event_row"].to_numpy()})
    first = df.groupby(["s", "item"])["row"].min().to_numpy()
    cum = np.cumsum(np.bincount(first, minlength=n))
    uniq = (cum - np.r_[0, cum][sess["start"].to_numpy()[sess_of_row]]).astype(np.int32)
    grp = ev["session_idx"]
    en = ev["event_name"]
    cart = en.eq("add").astype(np.int8).groupby(grp).cummax().to_numpy(np.int8)
    key_seen = en.isin(["add", "purchase"]).astype(np.int8).groupby(grp).cummax().to_numpy() > 0
    det_cnt = en.eq("detail").astype(np.int32).groupby(grp).cumsum().to_numpy()
    undetermined = (~key_seen) & (det_cnt < 3)
    split_arr = ev["split"].to_numpy()
    y_arr = ev["y"].map(LABEL_MAP).to_numpy(np.int8)
    false = np.zeros(n, dtype=bool)

    prefix = pd.DataFrame({"session_idx": sess_of_row, "session_id": ev["session_id"].to_numpy(),
                           "prefix_len": ev["event_index"].to_numpy(np.int16), "y": y_arr, "split": split_arr,
                           "unique_items": uniq, "unique_items_exp3": uniq, "cart_flag": cart,
                           "undetermined": undetermined, "leak_agg": false, "leak_graph": false})
    events_out = pd.DataFrame({
        "session_idx": sess_of_row, "event_index": ev["event_index"].to_numpy(np.int16),
        "event_name": en.to_numpy(),
        "event_name_idx": en.map(event_vocab).fillna(1).astype(np.int16).to_numpy(),
        "page_key_idx": ev["hashed_url"].map(page_vocab).fillna(1).astype(np.int32).to_numpy(),
        "dt_norm": ((dt - dt_mean) / dt_std).astype(np.float32), "dt_sec": dt.astype(np.float32),
        "dt01": (np.minimum(dt, max_dt) / max_dt).astype(np.float32),
        "event_timestamp": ev["ts_ms"].to_numpy(np.int64) * 1000,  # microseconds, as in Exp4
    })
    counts = np.bincount(items["event_row"].to_numpy(), minlength=n)
    ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)

    os.makedirs(out_dir, exist_ok=True)
    events_out.to_parquet(os.path.join(out_dir, "events.parquet"), index=False)
    prefix.to_parquet(os.path.join(out_dir, "prefixes.parquet"), index=False)
    sess.to_parquet(os.path.join(out_dir, "sessions.parquet"), index=False)
    items[["event_row", "item_idx", "item_id", "rel"]].to_parquet(os.path.join(out_dir, "items.parquet"), index=False)
    item_arr, rel_arr = items["item_idx"].to_numpy(np.int32), items["rel"].to_numpy(np.int8)
    np.savez_compressed(os.path.join(out_dir, "edges.npz"), fixed_ptr=ptr, fixed_item=item_arr, fixed_rel=rel_arr,
                        exp3_ptr=ptr, exp3_item=item_arr, exp3_rel=rel_arr, price_lookup=price_lookup)

    def by(frame, col):
        return {k: int(v) for k, v in frame[col].value_counts().items()}

    meta = {
        "dataset": "Coveo SIGIR eCom 2021", "T_MAX": t_max, "LABEL_MAP": LABEL_MAP, "CLASS_ORDER": CLASS_ORDER,
        "session_cat_cols": SESSION_CAT_COLS, "key_events": KEY_EVENTS, "item_relations": ITEM_RELATIONS,
        "split_boundaries": {"p70": p70, "p85": p85},
        "dt_stats": {"dt_mean_sec": dt_mean, "dt_std_sec": dt_std, "MAX_DT_sec": max_dt},
        "price_stats": {"price_bucket_mean": p_mean, "price_bucket_std": p_std},
        "vocab_sizes": {"event": len(event_vocab), "page": len(page_vocab), "item": len(item_vocab)},
        "event_vocab": event_vocab, "cat_vocab_sizes": {c: len(v) for c, v in cat_vocabs.items()},
        "sample": info,
        "counts": {"sessions": int(len(sess)), "sessions_by_split": by(sess, "split"),
                   "sessions_by_class": by(sess, "y"), "prefixes": int(n),
                   "prefixes_by_split": {s: int((split_arr == s).sum()) for s in ("train", "val", "test")},
                   "test_prefixes_by_class": {c: int(((split_arr == "test") & (y_arr == LABEL_MAP[c])).sum())
                                              for c in CLASS_ORDER},
                   "undetermined_test_prefixes": int(undetermined[split_arr == "test"].sum())},
        "repro_all_pass": None, "leakage": None,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=int)
    with open(os.path.join(out_dir, "cat_vocabs.json"), "w") as f:
        json.dump(cat_vocabs, f)
    print(json.dumps(meta["counts"], indent=2))
    return meta
