"""Resumable multi-seed training with early stopping on val Macro-F1.

For every (model, seed) the following files live in <run_dir>:
  ckpt/<model>__seed<seed>.pt      epoch-level checkpoint (deleted when the run is done)
  preds/<model>__seed<seed>.parquet test predictions + class probabilities
  runs/<model>__seed<seed>.json    history, best epoch, val/test Macro-F1, runtime
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .models import MODEL_SPECS

EVAL_BATCH = 2048


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_tag(model_name, seed):
    return f"{model_name}__seed{seed}"


def is_done(run_dir, model_name, seed):
    return os.path.exists(os.path.join(run_dir, "runs", run_tag(model_name, seed) + ".json"))


@torch.no_grad()
def predict(model, D, rows, build, device, batch_size=EVAL_BATCH):
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        b = build(D, rows[i:i + batch_size], device)
        out.append(torch.softmax(model(b).float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0))


def _as_rng_state(s):
    """Saved RNG state (tensor on any device, numpy array, list, bytes) -> CPU ByteTensor."""
    if isinstance(s, (bytes, bytearray)):
        s = list(s)
    if isinstance(s, torch.Tensor):
        return s.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(np.asarray(s)).to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng(ck, log, tag):
    """Best effort only: data order comes from (seed, epoch), so a failed restore merely
    changes dropout noise after the resume point. Never fatal."""
    try:
        torch.set_rng_state(_as_rng_state(ck["torch_rng"]))
    except Exception as e:
        log(f"[{tag}] warning: CPU RNG state not restored ({type(e).__name__}: {e})")
    states = ck.get("cuda_rng")
    if not torch.cuda.is_available() or not states:
        return
    try:
        states = [_as_rng_state(s) for s in states][:torch.cuda.device_count()]
        torch.cuda.set_rng_state_all(states)  # module attribute, looked up at call time
    except Exception as e:
        log(f"[{tag}] warning: CUDA RNG state not restored ({type(e).__name__}: {e})")


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_one(D, model_name, seed, run_dir, device, log=print, max_train_rows=None):
    """Train one (model, seed). Resumes from the epoch checkpoint if present."""
    spec = MODEL_SPECS[model_name]
    cfg = spec["train"]
    build = spec["batch"]
    tag = run_tag(model_name, seed)
    for sub in ("ckpt", "preds", "runs"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt", tag + ".pt")
    if is_done(run_dir, model_name, seed):
        log(f"[{tag}] already done, skipping")
        return

    set_global_seed(seed)
    model = spec["make"](D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(D.class_weights()).to(device))

    train_rows = D.rows["train"]
    if max_train_rows is not None:  # smoke tests only
        train_rows = train_rows[:max_train_rows]
    val_rows = D.rows["val"]

    state = {"epoch": 0, "best_val": -1.0, "best_epoch": 0, "patience": 0,
             "history": [], "best_state": None, "elapsed": 0.0}
    if os.path.exists(ckpt_path):
        # load on CPU: RNG states must be CPU ByteTensors; model/optimizer move to device on load
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        _restore_rng(ck, log, tag)
        log(f"[{tag}] resumed after epoch {state['epoch']}")

    while state["epoch"] < cfg["max_epochs"] and state["patience"] < cfg["patience"]:
        epoch = state["epoch"] + 1
        t0 = time.time()
        model.train()
        perm = np.random.default_rng([seed, epoch]).permutation(train_rows)
        bs = cfg["batch_size"]
        loss_sum = 0.0
        for i in range(0, len(perm), bs):
            rows = perm[i:i + bs]
            b = build(D, rows, device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(b), b["y"])
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(rows)
        val_prob = predict(model, D, val_rows, build, device)
        val_f1 = macro_f1(D.y[val_rows], val_prob.argmax(1))
        dt = time.time() - t0
        state["elapsed"] += dt
        state["history"].append({"epoch": epoch, "train_loss": loss_sum / len(perm),
                                 "val_macro_f1": val_f1, "seconds": round(dt, 1)})
        if val_f1 > state["best_val"]:
            state.update(best_val=val_f1, best_epoch=epoch, patience=0,
                         best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            mark = " *"
        else:
            state["patience"] += 1
            mark = ""
        state["epoch"] = epoch
        log(f"[{tag}] epoch {epoch:2d} loss={loss_sum / len(perm):.4f} val_f1={val_f1:.4f}{mark} ({dt:.0f}s)")
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                     ckpt_path)

    model.load_state_dict(state["best_state"])
    t0 = time.time()
    test_rows = D.rows["test"]
    prob = predict(model, D, test_rows, build, device)
    pred = prob.argmax(1)
    test_f1 = macro_f1(D.y[test_rows], pred)
    state["elapsed"] += time.time() - t0

    df = pd.DataFrame({"row": test_rows, "pred": pred.astype(np.int8)})
    for c in range(4):
        df[f"p{c}"] = prob[:, c].astype(np.float32)
    df.to_parquet(os.path.join(run_dir, "preds", tag + ".parquet"), index=False)
    torch.save(state["best_state"], os.path.join(run_dir, "ckpt", tag + "__best.pt"))
    info = {"model": model_name, "seed": seed, "best_epoch": state["best_epoch"],
            "best_val_macro_f1": state["best_val"], "test_macro_f1": test_f1,
            "epochs_run": state["epoch"], "runtime_sec": round(state["elapsed"], 1),
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_config": cfg, "history": state["history"]}
    with open(os.path.join(run_dir, "runs", tag + ".json"), "w") as f:
        json.dump(info, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    log(f"[{tag}] done: best_epoch={state['best_epoch']} val={state['best_val']:.4f} "
        f"test={test_f1:.4f} ({state['elapsed'] / 60:.1f} min)")
    return info
