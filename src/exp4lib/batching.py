"""Numpy batch builders. A batch is a set of prefix rows; prefix row r is the
prefix that ends at event row r, i.e. events sess_start[s] .. r."""
import numpy as np
import torch


def _excl_cumsum(a):
    out = np.zeros_like(a)
    np.cumsum(a[:-1], out=out[1:])
    return out


def _ranges(starts, counts):
    """Concatenate arange(s, s+c) for every (s, c)."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    return np.repeat(starts - _excl_cumsum(counts), counts) + np.arange(total)


def _session_tensors(D, rows, join, device):
    sess = D.ev_sess[rows]
    return {
        "cat": torch.from_numpy(D.sess_cat[sess]).to(device),
        "num": torch.from_numpy(D.session_numeric(rows, join)).to(device),
        "y": torch.from_numpy(D.y[rows]).to(device),
    }


def _prefix_events(D, rows):
    t = D.prefix_len[rows]
    start = D.sess_start[D.ev_sess[rows]]
    erows = _ranges(start, t)
    owner = np.repeat(np.arange(len(rows)), t)
    offs = np.arange(len(erows)) - np.repeat(_excl_cumsum(t), t)
    return t, erows, owner, offs


def _item_edges(D, erows, join):
    ptr, item, rel = D.edges[join]
    cnt = ptr[erows + 1] - ptr[erows]
    eidx = _ranges(ptr[erows], cnt)
    src = np.repeat(np.arange(len(erows)), cnt)  # index into erows
    return src, item[eidx], rel[eidx]


def graph_batch(D, rows, device, join="fixed", with_page_item=True):
    """Heterogeneous prefix graphs, identical in content to Exp3 build_prefix_subgraph.

    Event nodes: one per prefix event (event_name_idx, position 1..t, delta_t_prev_norm).
    Page/Item nodes: deduplicated within each prefix. Edges: next_event, on_page,
    view_item_of / add_to_cart_of / purchase_item_of (reverse edges added in the model).
    """
    t, erows, owner, offs = _prefix_events(D, rows)
    n_ev = len(erows)
    b = {
        "event_name": torch.from_numpy(D.ev_name[erows]).to(device),
        "event_num": torch.from_numpy(np.stack([(offs + 1).astype(np.float32),
                                                D.ev_dtn[erows]], axis=1)).to(device),
        "readout": torch.from_numpy(np.cumsum(t) - 1).to(device),
        "event_graph": torch.from_numpy(owner).to(device),  # prefix id of every event node (pooling)
    }
    nxt = np.flatnonzero(offs < np.repeat(t, t) - 1)
    b["next_event"] = torch.from_numpy(np.stack([nxt, nxt + 1])).to(device)
    b.update(_session_tensors(D, rows, join, device))
    if not with_page_item:
        return b

    P = int(D.vocab["page"])
    pkey = owner * P + D.ev_page[erows]
    upk, pinv = np.unique(pkey, return_inverse=True)
    b["page_idx"] = torch.from_numpy(upk % P).to(device)
    b["on_page"] = torch.from_numpy(np.stack([np.arange(n_ev), pinv.reshape(-1)])).to(device)

    src, item, rel = _item_edges(D, erows, join)
    I = int(D.vocab["item"])
    ikey = owner[src] * I + item
    uik, iinv = np.unique(ikey, return_inverse=True)
    item_idx = uik % I
    iinv = iinv.reshape(-1)
    b["item_idx"] = torch.from_numpy(item_idx).to(device)
    b["item_price"] = torch.from_numpy(D.price_lookup[item_idx]).to(device)
    for r, name in enumerate(D.item_relations):
        m = rel == r
        b[name] = torch.from_numpy(np.stack([src[m], iinv[m]])).to(device)
    return b


def event_feature_batch(D, rows, device, join="fixed"):
    """Event chain with page/item as node features (homogeneous GraphSAGE)."""
    t, erows, owner, offs = _prefix_events(D, rows)
    b = {
        "event_name": torch.from_numpy(D.ev_name[erows]).to(device),
        "event_num": torch.from_numpy(np.stack([(offs + 1).astype(np.float32),
                                                D.ev_dtn[erows]], axis=1)).to(device),
        "page_idx": torch.from_numpy(D.ev_page[erows]).to(device),
        "readout": torch.from_numpy(np.cumsum(t) - 1).to(device),
        "event_graph": torch.from_numpy(owner).to(device),
    }
    nxt = np.flatnonzero(offs < np.repeat(t, t) - 1)
    b["next_event"] = torch.from_numpy(np.stack([nxt, nxt + 1])).to(device)
    src, item, _ = _item_edges(D, erows, join)
    b["item_tok"] = torch.from_numpy(src).to(device)
    b["item_idx"] = torch.from_numpy(item).to(device)
    b["item_price"] = torch.from_numpy(D.price_lookup[item]).to(device)
    b["n_tok"] = len(erows)
    b.update(_session_tensors(D, rows, join, device))
    return b


def seq_batch(D, rows, device, join="fixed"):
    """Right-padded sequences (length = longest prefix in the batch).

    Right padding + causal attention is equivalent to the Exp2 left-padded SASRec
    (positions 0..t-1, readout at the last real token) and avoids fully-masked rows.
    """
    t, erows, owner, offs = _prefix_events(D, rows)
    B, L = len(rows), int(t.max())
    flat = owner * L + offs
    tok = np.zeros(B * L, dtype=np.int64); tok[flat] = D.ev_name[erows]
    page = np.zeros(B * L, dtype=np.int64); page[flat] = D.ev_page[erows]
    dt = np.zeros(B * L, dtype=np.float32); dt[flat] = D.ev_dt01[erows]
    src, item, _ = _item_edges(D, erows, join)
    b = {
        "tok": torch.from_numpy(tok.reshape(B, L)).to(device),
        "page": torch.from_numpy(page.reshape(B, L)).to(device),
        "dt": torch.from_numpy(dt.reshape(B, L)).to(device),
        "len": torch.from_numpy(t).to(device),
        "item_tok": torch.from_numpy(flat[src]).to(device),
        "item_idx": torch.from_numpy(item).to(device),
        "item_price": torch.from_numpy(D.price_lookup[item]).to(device),
    }
    b.update(_session_tensors(D, rows, join, device))
    return b
