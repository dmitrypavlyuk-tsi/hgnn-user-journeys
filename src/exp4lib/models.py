"""Exp4 neural models. Graph models follow Exp3 Section 6 / 10.4, sequence models
follow Exp2 Section 6.6. All take the dict batches from batching.py."""
import functools

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter, softmax

from . import batching

DEFAULT_ITEM_RELATIONS = ["view_item_of", "add_to_cart_of", "purchase_item_of"]  # Exp3/Exp4


def hetero_edge_types(item_relations=DEFAULT_ITEM_RELATIONS):
    """Edge types in the Exp3 order (this order fixes parameter initialisation)."""
    types = [("event", "next_event", "event"),
             ("event", "on_page", "page"), ("page", "rev_on_page", "event")]
    for rel in item_relations:
        types += [("event", rel, "item"), ("item", "rev_" + rel, "event")]
    return types


HETERO_EDGE_TYPES = hetero_edge_types()


class SessionVector(nn.Module):
    """Exp3 session features: one hidden_dim embedding per categorical + raw numerics."""

    def __init__(self, cat_vocab_sizes, n_numeric, hidden_dim):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s + 1, hidden_dim) for s in cat_vocab_sizes])
        self.dim = len(cat_vocab_sizes) * hidden_dim + n_numeric

    def forward(self, b):
        return torch.cat([e(b["cat"][:, i]) for i, e in enumerate(self.embs)] + [b["num"]], dim=-1)


def mlp_head(in_dim, hidden_dim, n_classes, dropout):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden_dim, n_classes))


class ItemTokenVector(nn.Module):
    """Mean over a token's items of (item embedding + price projection); zero if none."""

    def __init__(self, item_vocab, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(item_vocab, hidden_dim, padding_idx=0)
        self.price = nn.Linear(1, hidden_dim)

    def forward(self, b, n_tok):
        v = self.emb(b["item_idx"]) + self.price(b["item_price"].unsqueeze(-1))
        out = torch.zeros(n_tok, v.shape[1], device=v.device, dtype=v.dtype)
        out.index_add_(0, b["item_tok"], v)
        cnt = torch.zeros(n_tok, device=v.device, dtype=v.dtype)
        cnt.index_add_(0, b["item_tok"], torch.ones_like(b["item_price"]))
        return out / cnt.clamp(min=1).unsqueeze(-1)


# --------------------------------------------------------------------------
# Graph models
# --------------------------------------------------------------------------
class SessionHGNN(nn.Module):
    """Exp3 SessionHGNN (HeteroConv + SAGEConv, last-event readout).

    Exp4 stage 3b options: n_layers (Exp3 = 2) and readout:
      "last" - Exp3: embedding of the last event node only
      "mean" - [last event, mean over all event nodes of the prefix]
      "attn" - [last event, gated attention pooling over all event nodes]
    """

    def __init__(self, vocab, cat_vocab_sizes, hidden_dim=64, n_classes=4, dropout=0.3,
                 n_layers=2, readout="last", item_relations=DEFAULT_ITEM_RELATIONS):
        super().__init__()
        H = hidden_dim
        self.H = H
        self.readout = readout
        self.item_relations = list(item_relations)
        edge_types = hetero_edge_types(self.item_relations)
        self.event_emb = nn.Embedding(vocab["event"], H, padding_idx=0)
        self.page_emb = nn.Embedding(vocab["page"], H, padding_idx=0)
        self.item_emb = nn.Embedding(vocab["item"], H, padding_idx=0)
        self.event_numeric_proj = nn.Linear(2, H)
        self.item_numeric_proj = nn.Linear(1, H)
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((H, H), H) for et in edge_types}, aggr="sum")
            for _ in range(n_layers)])
        if readout == "attn":
            self.gate = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 1))
        graph_dim = H if readout == "last" else 2 * H
        self.session = SessionVector(cat_vocab_sizes, 4, H)
        self.head = mlp_head(graph_dim + self.session.dim, H, n_classes, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, b):
        x_event = self.event_emb(b["event_name"]) + self.event_numeric_proj(b["event_num"])
        x_page = self.page_emb(b["page_idx"])
        x_item = self.item_emb(b["item_idx"]) + self.item_numeric_proj(b["item_price"].unsqueeze(-1))
        x = {"event": x_event, "page": x_page, "item": x_item}
        ei = {("event", "next_event", "event"): b["next_event"]}
        for rel, dst in [("on_page", "page")] + [(r, "item") for r in self.item_relations]:
            ei[("event", rel, dst)] = b[rel]
            ei[(dst, "rev_" + rel, "event")] = b[rel].flip(0)
        for conv in self.convs:
            x = conv(x, ei)
            x = {k: self.dropout(v.relu()) for k, v in x.items()}
        xe = x["event"]
        h = xe[b["readout"]]
        if self.readout != "last":
            g, B = b["event_graph"], h.shape[0]
            if self.readout == "mean":
                pooled = scatter(xe, g, dim=0, dim_size=B, reduce="mean")
            else:
                a = softmax(self.gate(xe), g, num_nodes=B)
                pooled = scatter(a * xe, g, dim=0, dim_size=B, reduce="sum")
            h = torch.cat([h, pooled], dim=-1)
        return self.head(torch.cat([h, self.session(b)], dim=-1))


class EventOnlyHGNN(nn.Module):
    """Exp3 10.4 ablation: Event nodes + next_event only. use_session=False removes
    the session vector from the head entirely (new Exp4 variant)."""

    def __init__(self, vocab, cat_vocab_sizes, hidden_dim=64, n_classes=4, dropout=0.3, use_session=True):
        super().__init__()
        H = hidden_dim
        et = ("event", "next_event", "event")
        self.event_emb = nn.Embedding(vocab["event"], H, padding_idx=0)
        self.event_numeric_proj = nn.Linear(2, H)
        self.conv1 = HeteroConv({et: SAGEConv((H, H), H)}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((H, H), H)}, aggr="sum")
        self.use_session = use_session
        self.session = SessionVector(cat_vocab_sizes, 4, H) if use_session else None
        self.head = mlp_head(H + (self.session.dim if use_session else 0), H, n_classes, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, b):
        x = {"event": self.event_emb(b["event_name"]) + self.event_numeric_proj(b["event_num"])}
        ei = {("event", "next_event", "event"): b["next_event"]}
        x = self.conv1(x, ei)
        x = {k: self.dropout(v.relu()) for k, v in x.items()}
        x = self.conv2(x, ei)
        x = {k: self.dropout(v.relu()) for k, v in x.items()}
        h = x["event"][b["readout"]]
        if self.use_session:
            h = torch.cat([h, self.session(b)], dim=-1)
        return self.head(h)


class HomoGraphSAGE(nn.Module):
    """Homogeneous GraphSAGE on the event chain. Page and item identity enter as
    node features of the event (no shared Page/Item nodes, no extra relations).
    n_layers / readout as in SessionHGNN (stage 3c: pooled-readout control)."""

    def __init__(self, vocab, cat_vocab_sizes, hidden_dim=64, n_classes=4, dropout=0.3,
                 n_layers=2, readout="last"):
        super().__init__()
        H = hidden_dim
        self.readout = readout
        self.event_emb = nn.Embedding(vocab["event"], H, padding_idx=0)
        self.event_numeric_proj = nn.Linear(2, H)
        self.page_emb = nn.Embedding(vocab["page"], H, padding_idx=0)
        self.items = ItemTokenVector(vocab["item"], H)
        self.convs = nn.ModuleList([SAGEConv(H, H) for _ in range(n_layers)])
        self.session = SessionVector(cat_vocab_sizes, 4, H)
        graph_dim = H if readout == "last" else 2 * H
        self.head = mlp_head(graph_dim + self.session.dim, H, n_classes, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, b):
        x = (self.event_emb(b["event_name"]) + self.event_numeric_proj(b["event_num"])
             + self.page_emb(b["page_idx"]) + self.items(b, b["n_tok"]))
        for conv in self.convs:
            x = self.dropout(conv(x, b["next_event"]).relu())
        h = x[b["readout"]]
        if self.readout == "mean":
            pooled = scatter(x, b["event_graph"], dim=0, dim_size=h.shape[0], reduce="mean")
            h = torch.cat([h, pooled], dim=-1)
        return self.head(torch.cat([h, self.session(b)], dim=-1))


# --------------------------------------------------------------------------
# Sequence models
# --------------------------------------------------------------------------
class SeqInput(nn.Module):
    """Per-token input: event + time (+ page + items)."""

    def __init__(self, vocab, H, use_page_item, use_pos, max_len=43):  # max_len = T_MAX
        super().__init__()
        self.token_emb = nn.Embedding(vocab["event"], H, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, H) if use_pos else None
        self.time_proj = nn.Linear(1, H)
        self.use_page_item = use_page_item
        if use_page_item:
            self.page_emb = nn.Embedding(vocab["page"], H, padding_idx=0)
            self.items = ItemTokenVector(vocab["item"], H)

    def forward(self, b):
        B, L = b["tok"].shape
        x = self.token_emb(b["tok"]) + self.time_proj(b["dt"].unsqueeze(-1))
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(L, device=x.device)).unsqueeze(0)
        if self.use_page_item:
            x = x + self.page_emb(b["page"]) + self.items(b, B * L).view(B, L, -1)
        return x


class SASRec(nn.Module):
    """Exp2 SASRec+Time (2 layers, 2 heads, causal attention, last-token readout).
    use_session adds the Exp3 session vector before an MLP head (as in the HGNN);
    use_page_item adds page and item embeddings to every event token."""

    def __init__(self, vocab, cat_vocab_sizes, hidden_dim=64, n_classes=4, dropout=0.1,
                 num_heads=2, num_layers=2, use_session=False, use_page_item=False, max_len=43):
        super().__init__()
        H = hidden_dim
        self.inp = SeqInput(vocab, H, use_page_item, use_pos=True, max_len=max_len)
        self.layer_norm = nn.LayerNorm(H)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=H, nhead=num_heads, dim_feedforward=4 * H,
                                           dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.use_session = use_session
        if use_session:
            self.session = SessionVector(cat_vocab_sizes, 4, H)
            self.head = mlp_head(H + self.session.dim, H, n_classes, dropout)
        else:
            self.head = nn.Linear(H, n_classes)

    def forward(self, b):
        x = self.dropout(self.layer_norm(self.inp(b)))
        L = x.shape[1]
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.transformer(x, mask=causal)
        h = x[torch.arange(x.shape[0], device=x.device), b["len"] - 1]
        if self.use_session:
            h = torch.cat([h, self.session(b)], dim=-1)
        return self.head(h)


class GRUClassifier(nn.Module):
    """GRU4Rec-style encoder with the same inputs as SASRec+Time+SF+PageItem."""

    def __init__(self, vocab, cat_vocab_sizes, hidden_dim=64, n_classes=4, dropout=0.1):
        super().__init__()
        H = hidden_dim
        self.inp = SeqInput(vocab, H, use_page_item=True, use_pos=False)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(H, H, num_layers=1, batch_first=True)
        self.session = SessionVector(cat_vocab_sizes, 4, H)
        self.head = mlp_head(H + self.session.dim, H, n_classes, dropout)

    def forward(self, b):
        x, _ = self.gru(self.dropout(self.inp(b)))  # causal: output at t-1 sees tokens < t only
        h = x[torch.arange(x.shape[0], device=x.device), b["len"] - 1]
        return self.head(torch.cat([h, self.session(b)], dim=-1))


# --------------------------------------------------------------------------
# Registry. Training settings are those of the thesis for each family:
# graph models = Exp3 7.0, sequence models = Exp2 6.5.
# --------------------------------------------------------------------------
GRAPH_TRAIN = {"batch_size": 256, "lr": 1e-3, "max_epochs": 50, "patience": 7}
SEQ_TRAIN = {"batch_size": 512, "lr": 1e-3, "max_epochs": 30, "patience": 3}


def _pb(builder, **kw):
    return functools.partial(builder, **kw)


MODEL_SPECS = {
    "SASRec+Time": dict(
        make=lambda D: SASRec(D.vocab, D.cat_vocab_sizes, max_len=D.t_max),
        batch=_pb(batching.seq_batch), train=SEQ_TRAIN),
    "SASRec+Time+SF": dict(
        make=lambda D: SASRec(D.vocab, D.cat_vocab_sizes, use_session=True, max_len=D.t_max),
        batch=_pb(batching.seq_batch), train=SEQ_TRAIN),
    "SASRec+Time+SF+PageItem": dict(
        make=lambda D: SASRec(D.vocab, D.cat_vocab_sizes, use_session=True, use_page_item=True,
                              max_len=D.t_max),
        batch=_pb(batching.seq_batch), train=SEQ_TRAIN),
    "GRU+Time+SF+PageItem": dict(
        make=lambda D: GRUClassifier(D.vocab, D.cat_vocab_sizes),
        batch=_pb(batching.seq_batch), train=SEQ_TRAIN),
    "HGNN_full": dict(
        make=lambda D: SessionHGNN(D.vocab, D.cat_vocab_sizes, item_relations=D.item_relations),
        batch=_pb(batching.graph_batch), train=GRAPH_TRAIN),
    "HGNN_no_page_item": dict(
        make=lambda D: EventOnlyHGNN(D.vocab, D.cat_vocab_sizes),
        batch=_pb(batching.graph_batch, with_page_item=False), train=GRAPH_TRAIN),
    "HGNN_no_page_item_no_SF": dict(
        make=lambda D: EventOnlyHGNN(D.vocab, D.cat_vocab_sizes, use_session=False),
        batch=_pb(batching.graph_batch, with_page_item=False), train=GRAPH_TRAIN),
    "HomoGraphSAGE+PageItemFeat": dict(
        make=lambda D: HomoGraphSAGE(D.vocab, D.cat_vocab_sizes),
        batch=_pb(batching.event_feature_batch), train=GRAPH_TRAIN),
    # Stage 3b: readout / depth variants of HGNN_full (same protocol, fixed join)
    "HGNN_full_meanpool": dict(
        make=lambda D: SessionHGNN(D.vocab, D.cat_vocab_sizes, readout="mean",
                                   item_relations=D.item_relations),
        batch=_pb(batching.graph_batch), train=GRAPH_TRAIN),
    "HGNN_full_attnpool": dict(
        make=lambda D: SessionHGNN(D.vocab, D.cat_vocab_sizes, readout="attn",
                                   item_relations=D.item_relations),
        batch=_pb(batching.graph_batch), train=GRAPH_TRAIN),
    "HGNN_full_3layer_meanpool": dict(
        make=lambda D: SessionHGNN(D.vocab, D.cat_vocab_sizes, n_layers=3, readout="mean",
                                   item_relations=D.item_relations),
        batch=_pb(batching.graph_batch), train=GRAPH_TRAIN),
    # Stage 3c: homogeneous counterparts of the pooled HGNNs (heterogeneity control)
    "HomoGraphSAGE_meanpool": dict(
        make=lambda D: HomoGraphSAGE(D.vocab, D.cat_vocab_sizes, readout="mean"),
        batch=_pb(batching.event_feature_batch), train=GRAPH_TRAIN),
    "HomoGraphSAGE_3layer_meanpool": dict(
        make=lambda D: HomoGraphSAGE(D.vocab, D.cat_vocab_sizes, n_layers=3, readout="mean"),
        batch=_pb(batching.event_feature_batch), train=GRAPH_TRAIN),
    # Reference only: HGNN full trained/evaluated on the original Exp3 item join
    "HGNN_full_exp3join": dict(
        make=lambda D: SessionHGNN(D.vocab, D.cat_vocab_sizes, item_relations=D.item_relations),
        batch=_pb(batching.graph_batch, join="exp3"), train=GRAPH_TRAIN),
}
