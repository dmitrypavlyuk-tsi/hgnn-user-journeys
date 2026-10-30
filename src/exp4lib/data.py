"""Exp4 data stage: BigQuery extraction, Exp3-identical sessions/labels/split,
prefix tables, the item-join leakage check and the fixed per-event item join.

Conventions
-----------
* Events are sorted by (session_id, event_timestamp, event_bundle_sequence_id,
  event_name, row_hash). Exp3 sorted by the first three keys only and relied on
  BigQuery output order for ties, which is not reproducible; event_name is the
  Exp2 tie-breaker and row_hash makes the order fully deterministic.
* Because every session has length <= T_MAX, prefix (s, t) <-> event row
  start[s] + t - 1. All prefix-level arrays are therefore aligned with the
  event table: prefix row i == event row i.
* Labels: Browser=0, Researcher=1, Intent=2, Buyer=3 (Exp3 LABEL_MAP).
"""
import json
import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd

T_MAX = 43
LABEL_MAP = {"Browser": 0, "Researcher": 1, "Intent": 2, "Buyer": 3}
CLASS_ORDER = ["Browser", "Researcher", "Intent", "Buyer"]
SESSION_CAT_COLS = ["device_category", "traffic_source", "traffic_medium",
                    "traffic_campaign", "country", "region"]
ITEM_EVENTS = ["view_item", "add_to_cart", "purchase"]
ITEM_REL = {"view_item": 0, "add_to_cart": 1, "purchase": 2}
REL_NAMES = ["view_item_of", "add_to_cart_of", "purchase_item_of"]
KEY_EVENTS = ["purchase", "add_to_cart", "begin_checkout",
              "add_payment_info", "add_shipping_info"]
PRODUCT_EVENTS_OBSERVABLE = ["view_item", "view_item_list"]

# Expected counts from the Exp3 notebook (Sections 0-2)
EXP3_EXPECTED = {
    "items_base_rows": 3_431_675,
    "select_item_rows": 31_007,
    "events_base_rows": 1_899_352,
    "sessions_before_tmax": 334_037,
    "sessions": 327_283,
    "sessions_by_split": {"train": 229_098, "val": 49_092, "test": 49_093},
    "session_labels": {"Browser": 293_699, "Researcher": 19_048, "Intent": 12_241, "Buyer": 2_295},
    "prefixes": 1_277_287,
    "prefixes_by_split": {"train": 875_203, "val": 190_807, "test": 211_277},
    "test_prefixes_by_class": {"Browser": 131_685, "Buyer": 9_848, "Intent": 29_747, "Researcher": 39_997},
}

_SESSION_ID_SQL = "(SELECT MAX(value.int_value) FROM UNNEST(event_params) WHERE key = 'ga_session_id')"

EVENTS_QUERY = f"""
SELECT
    e.event_timestamp,
    e.event_bundle_sequence_id,
    e.event_name,
    e.user_pseudo_id,
    {_SESSION_ID_SQL} AS ga_session_id,
    (SELECT MAX(value.int_value) FROM UNNEST(event_params) WHERE key = 'ga_session_number') AS ga_session_number,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_location,
    e.device.category       AS device_category,
    e.geo.country           AS country,
    e.geo.region            AS region,
    e.traffic_source.source AS traffic_source,
    e.traffic_source.medium AS traffic_medium,
    e.traffic_source.name   AS traffic_campaign,
    FARM_FINGERPRINT(TO_JSON_STRING(e)) AS row_hash,
    -- items carried on the event row itself: this is the fix for the Exp3 join
    IF(e.event_name IN ('view_item', 'add_to_cart', 'purchase'),
       ARRAY(SELECT AS STRUCT it.item_id, it.price FROM UNNEST(e.items) AS it
             WHERE it.item_id IS NOT NULL),
       []) AS items
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS e
WHERE e.event_name IN (
    'page_view', 'view_search_results', 'view_item_list', 'view_item',
    'add_to_cart', 'begin_checkout',
    'add_shipping_info', 'add_payment_info', 'purchase'
)
  AND e.user_pseudo_id IS NOT NULL
  AND {_SESSION_ID_SQL} IS NOT NULL
  AND e.event_name IS NOT NULL
"""

SELECT_ITEM_QUERY = f"""
SELECT
    user_pseudo_id,
    {_SESSION_ID_SQL} AS ga_session_id,
    event_timestamp,
    event_name
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE event_name = 'select_item'
  AND user_pseudo_id IS NOT NULL
  AND {_SESSION_ID_SQL} IS NOT NULL
"""


# --------------------------------------------------------------------------
# Section 0: extraction
# --------------------------------------------------------------------------
def _query_df(client, sql):
    job = client.query(sql)
    try:
        return job.to_dataframe(create_bqstorage_client=True)
    except Exception as e:  # sandbox projects may not have the Storage Read API
        print(f"  (BigQuery Storage API unavailable: {type(e).__name__}; using REST download)")
        return job.result().to_dataframe(create_bqstorage_client=False)


def extract_raw(client, out_dir):
    """Run the two queries and save raw parquet files (events with items flattened out)."""
    ev_path = os.path.join(out_dir, "raw_events.parquet")
    it_path = os.path.join(out_dir, "raw_items.parquet")
    si_path = os.path.join(out_dir, "raw_select_item.parquet")
    if all(os.path.exists(p) for p in (ev_path, it_path, si_path)):
        print("Raw extracts already on Drive, skipping BigQuery.")
        return

    print("Querying events (with per-event items arrays)...")
    ev = _query_df(client, EVENTS_QUERY)
    print(f"  events: {ev.shape}")

    # Flatten items, keeping the event's raw row id as the join key
    ev = ev.reset_index(drop=True)
    ev["raw_row"] = np.arange(len(ev), dtype=np.int64)
    has = ev["items"].map(lambda a: a is not None and len(a) > 0).to_numpy()
    rows, ids, prices = [], [], []
    for r, arr in zip(ev.loc[has, "raw_row"].to_numpy(), ev.loc[has, "items"].to_numpy()):
        for it in arr:
            rows.append(r)
            ids.append(it["item_id"])
            prices.append(it["price"])
    items = pd.DataFrame({"raw_row": np.asarray(rows, dtype=np.int64),
                          "item_id": ids,
                          "price": pd.array(prices, dtype="Float64").astype("float64")})
    ev = ev.drop(columns=["items"])
    print(f"  items (flattened): {items.shape}")

    print("Querying select_item (labels only)...")
    si = _query_df(client, SELECT_ITEM_QUERY)
    print(f"  select_item: {si.shape}")

    ev.to_parquet(ev_path, index=False)
    items.to_parquet(it_path, index=False)
    si.to_parquet(si_path, index=False)


# --------------------------------------------------------------------------
# Section 1-2: sessions, labels, split, vocabularies, prefix tables
# --------------------------------------------------------------------------
def _normalize_page(url):
    if url is None or (isinstance(url, float) and np.isnan(url)):
        return None
    try:
        return urlparse(url).path.rstrip("/") or "/"
    except Exception:
        return None


def _check(name, got, expected, report):
    ok = got == expected
    report.append({"check": name, "got": got, "expected": expected, "ok": bool(ok)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected}")


def build_dataset(raw_dir, out_dir, strict=True):
    """Build all Exp4 tables. Writes to out_dir and returns the repro report."""
    ev = pd.read_parquet(os.path.join(raw_dir, "raw_events.parquet"))
    items = pd.read_parquet(os.path.join(raw_dir, "raw_items.parquet"))
    si = pd.read_parquet(os.path.join(raw_dir, "raw_select_item.parquet"))
    report = []
    print("Reproduction checks against Exp3:")
    _check("events_base_rows", int(len(ev)), EXP3_EXPECTED["events_base_rows"], report)
    _check("items_base_rows", int(len(items)), EXP3_EXPECTED["items_base_rows"], report)
    _check("select_item_rows", int(len(si)), EXP3_EXPECTED["select_item_rows"], report)

    # 1.0 session id + deterministic sort
    ev["session_id"] = ev["user_pseudo_id"] + "-" + ev["ga_session_id"].astype(str)
    ev = ev.sort_values(["session_id", "event_timestamp", "event_bundle_sequence_id",
                         "event_name", "row_hash"], kind="mergesort").reset_index(drop=True)
    ev["event_index"] = ev.groupby("session_id").cumcount() + 1

    # 1.1 session aggregates
    sess = ev.groupby("session_id").agg(
        session_length=("event_index", "max"),
        session_start_ts=("event_timestamp", "min"),
        session_end_ts=("event_timestamp", "max"),
    ).reset_index()

    # 1.2 labels (vectorised version of Exp3 assign_label)
    en = ev["event_name"]
    flags = pd.DataFrame({
        "session_id": ev["session_id"],
        "buy": en.eq("purchase"),
        "intent": en.isin(["add_to_cart", "begin_checkout", "add_payment_info", "add_shipping_info"]),
        "prod": en.isin(["view_item", "view_item_list"]),
    }).groupby("session_id").agg(buy=("buy", "any"), intent=("intent", "any"), prod=("prod", "sum"))
    si["session_id"] = si["user_pseudo_id"] + "-" + si["ga_session_id"].astype(str)
    flags["prod"] = flags["prod"] + si.groupby("session_id").size().reindex(flags.index, fill_value=0)
    y = np.where(flags["buy"], "Buyer",
                 np.where(flags["intent"], "Intent",
                          np.where(flags["prod"] >= 3, "Researcher", "Browser")))
    sess = sess.merge(pd.DataFrame({"session_id": flags.index, "y": y}), on="session_id", how="left")
    _check("sessions_before_tmax", int(len(sess)), EXP3_EXPECTED["sessions_before_tmax"], report)

    # 1.4 T_MAX filter
    sess = sess[sess["session_length"] <= T_MAX].reset_index(drop=True)
    _check("sessions", int(len(sess)), EXP3_EXPECTED["sessions"], report)
    _check("session_labels", {k: int(v) for k, v in sess["y"].value_counts().items()},
           EXP3_EXPECTED["session_labels"], report)

    # 1.5 temporal split (identical to Exp3)
    p70 = int(round(sess["session_end_ts"].quantile(0.70)))
    p85 = int(round(sess["session_end_ts"].quantile(0.85)))
    sess["split"] = np.where(sess["session_end_ts"] <= p70, "train",
                             np.where(sess["session_end_ts"] <= p85, "val", "test"))
    _check("sessions_by_split", {k: int(v) for k, v in sess["split"].value_counts().items()},
           EXP3_EXPECTED["sessions_by_split"], report)

    # 2.0 filter events to kept sessions
    keep = set(sess["session_id"])
    ev = ev[ev["session_id"].isin(keep)].reset_index(drop=True)
    sess = sess.sort_values("session_id").reset_index(drop=True)  # same order as groupby
    sess["session_idx"] = np.arange(len(sess), dtype=np.int32)
    ev = ev.merge(sess[["session_id", "session_idx", "split", "y"]], on="session_id", how="left")
    # merge keeps left order; events are already sorted by session_id
    starts = np.r_[0, np.cumsum(sess["session_length"].to_numpy())[:-1]].astype(np.int64)
    sess["start"] = starts
    assert len(ev) == int(sess["session_length"].sum())
    assert (ev["session_idx"].to_numpy()[starts] == sess["session_idx"].to_numpy()).all()

    n_pref = len(ev)
    _check("prefixes", int(n_pref), EXP3_EXPECTED["prefixes"], report)
    _check("prefixes_by_split", {k: int(v) for k, v in ev["split"].value_counts().items()},
           EXP3_EXPECTED["prefixes_by_split"], report)
    tst = ev[ev["split"] == "test"]["y"].value_counts()
    _check("test_prefixes_by_class", {k: int(v) for k, v in tst.items()},
           EXP3_EXPECTED["test_prefixes_by_class"], report)

    train_mask = ev["split"].eq("train").to_numpy()

    # 2.1 page_key
    uniq_loc = ev["page_location"].dropna().unique()
    loc_map = {u: _normalize_page(u) for u in uniq_loc}
    ev["page_key"] = ev["page_location"].map(loc_map)

    # 2.2 delta_t_prev (microseconds, z-scored with train stats) + Exp2-style seconds
    dt_us = ev.groupby("session_idx")["event_timestamp"].diff().fillna(0).to_numpy(np.float64)
    dt_mean = float(dt_us[train_mask].mean())
    dt_std = float(pd.Series(dt_us[train_mask]).std())
    ev["dt_norm"] = ((dt_us - dt_mean) / dt_std).astype(np.float32)
    dt_sec = dt_us / 1e6
    pos_tr = dt_sec[train_mask & (ev["event_index"].to_numpy() > 1)]
    max_dt = float(np.percentile(pos_tr[pos_tr > 0], 99))  # Exp2 MAX_DT
    ev["dt_sec"] = dt_sec.astype(np.float32)
    ev["dt01"] = (np.minimum(dt_sec, max_dt) / max_dt).astype(np.float32)

    # 2.4 vocabularies (train only)
    tr_ev = ev[train_mask]
    event_vocab = {n: i + 2 for i, n in enumerate(sorted(tr_ev["event_name"].unique()))}
    event_vocab.update({"<PAD>": 0, "<UNK>": 1})
    page_vocab = {k: i + 2 for i, k in enumerate(sorted(tr_ev["page_key"].dropna().unique()))}
    page_vocab.update({"<PAD>": 0, "<OOV>": 1})
    ev["event_name_idx"] = ev["event_name"].map(event_vocab).fillna(1).astype(np.int16)
    ev["page_key_idx"] = ev["page_key"].map(page_vocab).fillna(1).astype(np.int32)

    # items: attach to the filtered/sorted event rows via raw_row (the fixed join)
    raw2row = pd.Series(np.arange(n_pref, dtype=np.int64), index=ev["raw_row"].to_numpy())
    items = items[items["raw_row"].isin(raw2row.index)].copy()
    items["event_row"] = raw2row.reindex(items["raw_row"].to_numpy()).to_numpy()
    items = items.sort_values(["event_row", "raw_row"], kind="mergesort").reset_index(drop=True)
    items_split = ev["split"].to_numpy()[items["event_row"].to_numpy()]
    tr_items = items[items_split == "train"]
    item_vocab = {k: i + 2 for i, k in enumerate(sorted(tr_items["item_id"].unique()))}
    item_vocab.update({"<PAD>": 0, "<OOV>": 1})
    items["item_idx"] = items["item_id"].map(item_vocab).fillna(1).astype(np.int32)

    # 2.3 price normalisation and train-only item node price (Exp3 3.1)
    price_mean = float(tr_items["price"].mean())
    price_std = float(tr_items["price"].std())
    items["price_norm"] = (items["price"].fillna(0) - price_mean) / price_std
    item_price = (items[items_split == "train"].groupby("item_idx")["price_norm"].mean())
    price_lookup = np.zeros(len(item_vocab), dtype=np.float32)  # OOV/PAD -> 0.0
    price_lookup[item_price.index.to_numpy()] = item_price.to_numpy(np.float32)
    for k in ("<PAD>", "<OOV>"):
        price_lookup[item_vocab[k]] = 0.0

    ev_name_arr = ev["event_name"].to_numpy()
    items["rel"] = pd.Series(ev_name_arr[items["event_row"].to_numpy()]).map(ITEM_REL).to_numpy(np.int8)

    # 5.0 session-constant features (first non-null per session, as Exp3 groupby.first())
    sf = ev.groupby("session_idx")[SESSION_CAT_COLS + ["ga_session_number"]].first()
    gsn = pd.to_numeric(sf["ga_session_number"].reindex(sess["session_idx"]), errors="coerce").fillna(-1).to_numpy()
    sess["is_new_user"] = (gsn == 1).astype(np.int8)
    cat_vocabs = {}
    train_sess = sess["split"].eq("train").to_numpy()
    for col in SESSION_CAT_COLS:
        vals = sf[col].reindex(sess["session_idx"]).fillna("<NA>").astype(str).to_numpy()
        vocab = {v: i + 1 for i, v in enumerate(sorted(set(vals[train_sess])))}  # 0 = OOV
        cat_vocabs[col] = vocab
        sess[col + "_idx"] = pd.Series(vals).map(vocab).fillna(0).astype(np.int32).to_numpy()

    # Exp2-style session context strings for LightGBM
    for col in SESSION_CAT_COLS:
        sess["ctx_" + col] = sf[col].reindex(sess["session_idx"]).to_numpy()

    # ------------------------------------------------------------------
    # Item edges: fixed (own event) and Exp3-style (timestamp/bundle join)
    # ------------------------------------------------------------------
    pos = ev["event_index"].to_numpy(np.int16)
    sess_of_row = ev["session_idx"].to_numpy(np.int32)
    key_cols = ["session_idx", "event_timestamp", "event_bundle_sequence_id"]
    evk = ev[key_cols].copy()
    evk["event_row"] = np.arange(n_pref, dtype=np.int64)
    evk["is_item_event"] = ev["event_name"].isin(ITEM_EVENTS).to_numpy()

    it = items[["event_row", "item_idx", "item_id", "rel"]].copy()
    it["item_row"] = np.arange(len(it), dtype=np.int64)
    it = it.join(evk.set_index("event_row")[key_cols], on="event_row")

    # Exp3 graph: every prefix item-event links to every item row with the same key,
    # routed by the PREFIX event's name
    g_ev = evk[evk["is_item_event"]].rename(columns={"event_row": "src_row"})
    exp3_edges = it.merge(g_ev[key_cols + ["src_row"]], on=key_cols, how="inner")
    exp3_edges["src_rel"] = pd.Series(ev_name_arr[exp3_edges["src_row"].to_numpy()]).map(ITEM_REL).to_numpy(np.int8)
    exp3_edges = exp3_edges.sort_values(["src_row", "item_row"], kind="mergesort").reset_index(drop=True)

    # Exp3 aggregates: any prefix event (any name) with the same key pulls item rows in
    any_ev = evk.rename(columns={"event_row": "src_row"})
    agg_links = it.merge(any_ev[key_cols + ["src_row"]], on=key_cols, how="inner")

    def csr(src_rows, n):
        counts = np.bincount(src_rows, minlength=n)
        return np.r_[0, np.cumsum(counts)].astype(np.int64)

    fixed_ptr = csr(items["event_row"].to_numpy(), n_pref)
    exp3_ptr = csr(exp3_edges["src_row"].to_numpy(), n_pref)

    # --- prefix-level unique_items, fixed vs Exp3 ---
    def unique_items_curve(item_ids, eff_row):
        """unique_items at every prefix row given the row at which each item row becomes visible."""
        df = pd.DataFrame({"s": sess_of_row[eff_row], "item": item_ids, "row": eff_row})
        first = df.groupby(["s", "item"])["row"].min().to_numpy()
        inc = np.bincount(first, minlength=n_pref)
        cum = np.cumsum(inc)
        start_of_row = sess["start"].to_numpy()[sess_of_row]
        before = np.r_[0, cum][start_of_row]
        return (cum - before).astype(np.int32)

    uniq_fixed = unique_items_curve(items["item_id"].to_numpy(), items["event_row"].to_numpy())
    first_vis = agg_links.groupby("item_row")["src_row"].min()
    uniq_exp3 = unique_items_curve(it["item_id"].to_numpy()[first_vis.index.to_numpy()],
                                   first_vis.to_numpy())

    # --- leakage accounting ---
    def covered(lo_rows, hi_rows):
        """prefix rows r with lo <= r < hi for any interval (all within one session)."""
        m = lo_rows < hi_rows
        diff = np.zeros(n_pref + 1, dtype=np.int64)
        np.add.at(diff, lo_rows[m], 1)
        np.add.at(diff, hi_rows[m], -1)
        return np.cumsum(diff)[:-1] > 0

    leak_agg = covered(agg_links["src_row"].to_numpy(), agg_links["event_row"].to_numpy())
    leak_graph = covered(exp3_edges["src_row"].to_numpy(), exp3_edges["event_row"].to_numpy())
    signal = exp3_edges[ev_name_arr[exp3_edges["event_row"].to_numpy()] != "view_item"]
    leak_graph_signal = covered(signal["src_row"].to_numpy(), signal["event_row"].to_numpy())
    misrouted_edges = int((exp3_edges["src_rel"].to_numpy() != exp3_edges["rel"].to_numpy()).sum())
    dup_edges = int(len(exp3_edges) - (exp3_edges["src_row"] == exp3_edges["event_row"]).sum())

    # shared-timestamp statistic quoted in the brief
    grp = ev.groupby(["session_idx", "event_timestamp"])["event_index"].transform("size")
    share_ts = float((grp > 1).mean())

    split_arr = ev["split"].to_numpy()
    y_arr = ev["y"].map(LABEL_MAP).to_numpy(np.int8)
    leak = {
        "events_sharing_timestamp_within_session": round(share_ts, 4),
        "item_rows": int(len(items)),
        "exp3_graph_edges": int(len(exp3_edges)),
        "fixed_graph_edges": int(len(items)),
        "exp3_edges_not_from_own_event": dup_edges,
        "exp3_edges_with_wrong_relation_type": misrouted_edges,
        "prefixes_total": int(n_pref),
    }
    for name, mask in [("unique_items_sees_future_items", leak_agg),
                       ("unique_items_value_differs", uniq_exp3 != uniq_fixed),
                       ("graph_has_future_item_edges", leak_graph),
                       ("graph_has_future_add_to_cart_or_purchase_items", leak_graph_signal)]:
        leak[name] = {
            "all": int(mask.sum()),
            "by_split": {s: int(mask[split_arr == s].sum()) for s in ("train", "val", "test")},
            "test_by_class": {c: int(mask[(split_arr == "test") & (y_arr == LABEL_MAP[c])].sum())
                              for c in CLASS_ORDER},
        }
    print("\nLeakage check (Exp3 timestamp/bundle join):")
    print(json.dumps(leak, indent=2))

    # ------------------------------------------------------------------
    # Prefix-level arrays (prefix row == event row)
    # ------------------------------------------------------------------
    en_arr = ev["event_name"]
    cart_flag = en_arr.eq("add_to_cart").astype(np.int8).groupby(ev["session_idx"]).cummax().to_numpy(np.int8)
    key_seen = en_arr.isin(KEY_EVENTS).astype(np.int8).groupby(ev["session_idx"]).cummax().to_numpy() > 0
    prod_cnt = en_arr.isin(PRODUCT_EVENTS_OBSERVABLE).astype(np.int32).groupby(ev["session_idx"]).cumsum().to_numpy()
    undetermined = (~key_seen) & (prod_cnt < 3)

    prefix = pd.DataFrame({
        "session_idx": sess_of_row,
        "session_id": ev["session_id"].to_numpy(),
        "prefix_len": pos,
        "y": y_arr,
        "split": split_arr,
        "unique_items": uniq_fixed,
        "unique_items_exp3": uniq_exp3,
        "cart_flag": cart_flag,
        "undetermined": undetermined,
        "leak_agg": leak_agg,
        "leak_graph": leak_graph,
    })

    events_out = pd.DataFrame({
        "session_idx": sess_of_row,
        "event_index": pos,
        "event_name": ev["event_name"].to_numpy(),
        "event_name_idx": ev["event_name_idx"].to_numpy(),
        "page_key_idx": ev["page_key_idx"].to_numpy(),
        "dt_norm": ev["dt_norm"].to_numpy(),
        "dt_sec": ev["dt_sec"].to_numpy(),
        "dt01": ev["dt01"].to_numpy(),
        "event_timestamp": ev["event_timestamp"].to_numpy(),
    })
    sess_out = sess.drop(columns=[]).copy()

    os.makedirs(out_dir, exist_ok=True)
    events_out.to_parquet(os.path.join(out_dir, "events.parquet"), index=False)
    prefix.to_parquet(os.path.join(out_dir, "prefixes.parquet"), index=False)
    sess_out.to_parquet(os.path.join(out_dir, "sessions.parquet"), index=False)
    items[["event_row", "item_idx", "item_id", "rel"]].to_parquet(os.path.join(out_dir, "items.parquet"), index=False)
    np.savez_compressed(
        os.path.join(out_dir, "edges.npz"),
        fixed_ptr=fixed_ptr,
        fixed_item=items["item_idx"].to_numpy(np.int32),
        fixed_rel=items["rel"].to_numpy(np.int8),
        exp3_ptr=exp3_ptr,
        exp3_item=exp3_edges["item_idx"].to_numpy(np.int32),
        exp3_rel=exp3_edges["src_rel"].to_numpy(np.int8),
        price_lookup=price_lookup,
    )
    meta = {
        "T_MAX": T_MAX, "LABEL_MAP": LABEL_MAP, "CLASS_ORDER": CLASS_ORDER,
        "split_boundaries": {"p70": p70, "p85": p85},
        "dt_stats": {"dt_mean_us": dt_mean, "dt_std_us": dt_std, "MAX_DT_sec": max_dt},
        "price_stats": {"price_mean": price_mean, "price_std": price_std},
        "vocab_sizes": {"event": len(event_vocab), "page": len(page_vocab), "item": len(item_vocab)},
        "event_vocab": event_vocab,
        "cat_vocab_sizes": {c: len(v) for c, v in cat_vocabs.items()},
        "repro_report": report,
        "repro_all_pass": all(r["ok"] for r in report),
        "leakage": leak,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=int)
    with open(os.path.join(out_dir, "cat_vocabs.json"), "w") as f:
        json.dump(cat_vocabs, f)

    if strict and not meta["repro_all_pass"]:
        bad = [r["check"] for r in report if not r["ok"]]
        raise AssertionError(f"Exp3 counts not reproduced: {bad}. Set STRICT_REPRO=False to continue anyway.")
    return meta


# --------------------------------------------------------------------------
# Loading for the model stages
# --------------------------------------------------------------------------
class Exp4Data:
    """Light numpy view of the prepared tables, shared by every model."""

    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        ev = pd.read_parquet(os.path.join(data_dir, "events.parquet"))
        pf = pd.read_parquet(os.path.join(data_dir, "prefixes.parquet"))
        ss = pd.read_parquet(os.path.join(data_dir, "sessions.parquet"))
        e = np.load(os.path.join(data_dir, "edges.npz"))

        self.n = len(ev)
        self.ev_name = ev["event_name_idx"].to_numpy(np.int64)
        self.ev_page = ev["page_key_idx"].to_numpy(np.int64)
        self.ev_dtn = ev["dt_norm"].to_numpy(np.float32)
        self.ev_dt01 = ev["dt01"].to_numpy(np.float32)
        self.ev_pos = ev["event_index"].to_numpy(np.int64)
        self.ev_sess = ev["session_idx"].to_numpy(np.int64)
        self.sess_start = ss["start"].to_numpy(np.int64)
        # dataset-specific settings (defaults = Exp3/Exp4 GA4 dataset)
        self.session_cat_cols = self.meta.get("session_cat_cols", SESSION_CAT_COLS)
        self.key_events = self.meta.get("key_events", KEY_EVENTS)
        self.item_relations = self.meta.get("item_relations", REL_NAMES)
        self.t_max = int(self.meta.get("T_MAX", T_MAX))
        self.cat_cols = [c + "_idx" for c in self.session_cat_cols]
        self.sess_cat = ss[self.cat_cols].to_numpy(np.int64)
        self.sess_new = ss["is_new_user"].to_numpy(np.float32)

        self.y = pf["y"].to_numpy(np.int64)
        self.split = pf["split"].to_numpy()
        self.session_id = pf["session_id"].to_numpy()
        self.prefix_len = pf["prefix_len"].to_numpy(np.int64)
        self.undetermined = pf["undetermined"].to_numpy(bool)
        self.uniq = {"fixed": pf["unique_items"].to_numpy(np.float32),
                     "exp3": pf["unique_items_exp3"].to_numpy(np.float32)}
        self.cart = pf["cart_flag"].to_numpy(np.float32)
        self.edges = {
            "fixed": (e["fixed_ptr"], e["fixed_item"].astype(np.int64), e["fixed_rel"].astype(np.int64)),
            "exp3": (e["exp3_ptr"], e["exp3_item"].astype(np.int64), e["exp3_rel"].astype(np.int64)),
        }
        self.price_lookup = e["price_lookup"].astype(np.float32)
        self.vocab = self.meta["vocab_sizes"]
        self.cat_vocab_sizes = [self.meta["cat_vocab_sizes"][c] for c in self.session_cat_cols]
        self.rows = {s: np.flatnonzero(self.split == s) for s in ("train", "val", "test")}

    def class_weights(self):
        """Prefix-row balanced weights: total / (n_classes * count), Exp3 7.0."""
        ytr = self.y[self.rows["train"]]
        counts = np.bincount(ytr, minlength=4)
        return (len(ytr) / (4 * counts)).astype(np.float32)

    def session_numeric(self, rows, join="fixed"):
        """[prefix_len, unique_items, cart_flag, is_new_user], raw values as in Exp3."""
        return np.stack([self.prefix_len[rows].astype(np.float32),
                         self.uniq[join][rows],
                         self.cart[rows],
                         self.sess_new[self.ev_sess[rows]]], axis=1)
