"""Training, evaluation, parameter counting, result saving and plotting.

Main entry points
-----------------
* ``ExperimentConfig``  - every hyper-parameter of one run.
* ``run_experiment``    - train one model, evaluate on val/test, save everything
                          (``results/metrics.csv``, ``results/runs/<run_id>.json``,
                          figures). Finished runs are skipped, so a Kaggle session
                          that stops can simply be restarted.
* ``count_parameters``  - number of TRAINABLE parameters.
* ``smoke_test``        - synthetic-data self check (shapes, equations, masking).

Command line (from the repository root)::

    python -m src.train --smoke-test
    python -m src.train --cell wmc --epochs 5 --train-fraction 0.2
"""

import argparse
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .cells import CELL_LABELS, CELL_REGISTRY
from .data import (BatchLoader, DataBundle, DataConfig, build_bundle,
                   build_bundle_from_dataframes, compute_class_weights,
                   make_loaders, make_synthetic_dataframes)
from .model import TextClassifier, initialize_weights

METRICS_FILE = "metrics.csv"
OPTIMIZER_CHOICES = ("adam", "sgd", "rmsprop")


# --------------------------------------------------------------------------- #
# Reproducibility and device
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + GPU).

    Note: some CUDA kernels (e.g. embedding backward) are non-deterministic, so two
    GPU runs with the same seed can differ slightly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    """Hyper-parameters of one run (all saved as columns of metrics.csv)."""

    cell: str = "lstm"                 # lstm | cifg | wmc
    group: str = "default"             # free label, e.g. 'hp_study' or 'final'
    embed_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.3
    pooling: str = "last"              # last | mean | max
    optimizer: str = "adam"            # adam | sgd | rmsprop
    lr: float = 1e-3
    momentum: float = 0.9              # used by sgd only
    weight_decay: float = 0.0
    init: str = "xavier"               # xavier | orthogonal | default
    forget_bias: float = 1.0           # used when init is xavier / orthogonal
    batch_size: int = 64
    epochs: int = 6                    # maximum number of epochs
    patience: int = 2                  # early stopping on validation macro-F1
    grad_clip: float = 1.0
    use_class_weights: bool = False
    train_fraction: float = 1.0        # stratified fraction of the training split
    seed: int = 0                      # training seed (the data split seed is fixed)

    @property
    def run_id(self) -> str:
        return (f"{self.cell}_{self.optimizer}_lr{self.lr:g}_{self.init}_bs{self.batch_size}"
                f"_h{self.hidden_size}_e{self.embed_dim}_L{self.num_layers}"
                f"_bi{int(self.bidirectional)}_do{self.dropout:g}_{self.pooling}"
                f"_cw{int(self.use_class_weights)}_ep{self.epochs}"
                f"_tf{self.train_fraction:g}_s{self.seed}")


def config_from_row(row: Dict, **overrides) -> ExperimentConfig:
    """Rebuild an ``ExperimentConfig`` from a metrics.csv row (with optional overrides)."""
    kwargs = {}
    for f in fields(ExperimentConfig):
        if f.name in row and not pd.isna(row[f.name]):
            kwargs[f.name] = f.type(row[f.name])
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def build_model(cfg: ExperimentConfig, vocab_size: int, num_classes: int) -> TextClassifier:
    """Create and initialise the classifier described by ``cfg``."""
    model = TextClassifier(
        vocab_size=vocab_size, num_classes=num_classes, cell_type=cfg.cell,
        embed_dim=cfg.embed_dim, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional, dropout=cfg.dropout, pooling=cfg.pooling,
    )
    initialize_weights(model, cfg.init, cfg.forget_bias)
    return model


def build_optimizer(params, name: str, lr: float, momentum: float = 0.9,
                    weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Adam, SGD (with momentum) or RMSprop."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be one of {OPTIMIZER_CHOICES}")


def count_parameters(model: nn.Module, only_recurrent: bool = False) -> int:
    """Number of TRAINABLE parameters (optionally only the recurrent stack ``model.rnn``)."""
    module = model.rnn if only_recurrent else model
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def evaluate_predictions(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> Dict:
    """Accuracy, macro-F1, per-class F1 and confusion matrix."""
    ids = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=ids, average="macro", zero_division=0)),
        "per_class_f1": f1_score(labels, preds, labels=ids, average=None,
                                 zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds, labels=ids).tolist(),
    }


def run_epoch(model: nn.Module, loader: BatchLoader, criterion: nn.Module, device: torch.device,
              num_classes: int, optimizer: Optional[torch.optim.Optimizer] = None,
              grad_clip: Optional[float] = None) -> Dict:
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for x, _, y in loader:
            x = x.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_preds.append(logits.argmax(dim=1).detach().cpu())
            all_labels.append(y.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    result = evaluate_predictions(labels, preds, num_classes)
    result.update(loss=total_loss / max(total, 1), preds=preds, labels=labels)
    return result


def fit(model: nn.Module, train_loader: BatchLoader, val_loader: BatchLoader,
        criterion: nn.Module, optimizer: torch.optim.Optimizer, cfg: ExperimentConfig,
        device: torch.device, num_classes: int, verbose: bool = True):
    """Train with early stopping on validation macro-F1; restore the best weights.

    Returns ``(history, best_epoch, train_time_seconds)``. The training time includes
    the per-epoch validation passes. Train loss / accuracy are running values
    measured with dropout active.
    """
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc",
                               "train_macro_f1", "val_macro_f1")}
    best_f1, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0

    _sync(device)
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, num_classes,
                       optimizer, cfg.grad_clip)
        va = run_epoch(model, val_loader, criterion, device, num_classes)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_acc"].append(va["accuracy"])
        history["train_macro_f1"].append(tr["macro_f1"])
        history["val_macro_f1"].append(va["macro_f1"])
        if verbose:
            print(f"  epoch {epoch:02d}/{cfg.epochs} | train loss {tr['loss']:.4f} "
                  f"acc {tr['accuracy']:.4f} | val loss {va['loss']:.4f} "
                  f"acc {va['accuracy']:.4f} macro-F1 {va['macro_f1']:.4f}")

        if va["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, bad_epochs = va["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"  early stopping (best epoch: {best_epoch})")
                break
    _sync(device)
    elapsed = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, elapsed


# --------------------------------------------------------------------------- #
# Result files
# --------------------------------------------------------------------------- #
def load_metrics(results_dir: str = "results") -> pd.DataFrame:
    """Read results/metrics.csv (empty DataFrame if it does not exist yet)."""
    path = os.path.join(results_dir, METRICS_FILE)
    return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()


def append_metrics(row: Dict, results_dir: str = "results") -> None:
    """Append one row to metrics.csv (atomic write, so a stopped session cannot corrupt it)."""
    path = os.path.join(results_dir, METRICS_FILE)
    new = pd.DataFrame([row])
    df = pd.concat([pd.read_csv(path), new], ignore_index=True) if os.path.isfile(path) else new
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_run_details(run_id: str, results_dir: str = "results") -> Dict:
    """Load results/runs/<run_id>.json (history, confusion matrices, ...)."""
    import json
    with open(os.path.join(results_dir, "runs", f"{run_id}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_training_curves(history: Dict[str, List[float]], path: Optional[str] = None,
                         title: str = "Training curves"):
    """Loss, accuracy and validation macro-F1 per epoch."""
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[2].plot(epochs, history["val_macro_f1"], marker="o", color="tab:green",
                 label="validation")
    axes[2].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm: Sequence[Sequence[int]], class_names: Sequence[str],
                          path: Optional[str] = None, title: str = "Confusion matrix",
                          normalize: bool = True):
    """Confusion matrix heat-map (rows = true class, row-normalised by default)."""
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0) if normalize else cm
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{shown[i, j]:.2f}\n({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if shown[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(df: pd.DataFrame, path: Optional[str] = None,
                          title: str = "Model comparison (test set)"):
    """Mean +/- std over seeds of test accuracy / macro-F1, and parameter counts, per cell."""
    import matplotlib.pyplot as plt
    order = [c for c in CELL_REGISTRY if c in set(df["cell"])]
    labels = [CELL_LABELS[c] for c in order]
    agg = df.groupby("cell").agg(
        acc=("test_accuracy", "mean"), acc_std=("test_accuracy", "std"),
        f1=("test_macro_f1", "mean"), f1_std=("test_macro_f1", "std"),
        params=("trainable_params", "first"), rec=("recurrent_params", "first"),
    ).reindex(order).fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(order))
    width = 0.38
    axes[0].bar(x - width / 2, agg["acc"], width, yerr=agg["acc_std"], capsize=4, label="Accuracy")
    axes[0].bar(x + width / 2, agg["f1"], width, yerr=agg["f1_std"], capsize=4, label="Macro-F1")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Score", title="Test performance")
    lo = max(0.0, float(min(agg["acc"].min(), agg["f1"].min())) - 0.1)
    axes[0].set_ylim(lo, 1.0)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x - width / 2, agg["params"], width, label="Total trainable")
    axes[1].bar(x + width / 2, agg["rec"], width, label="Recurrent part only")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Parameters", title="Trainable parameters")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_hparam_effect(df: pd.DataFrame, axis: str, path: Optional[str] = None,
                       metric: str = "val_macro_f1", title: Optional[str] = None):
    """Grouped bars: effect of one hyper-parameter (``axis`` column) on ``metric``, per cell."""
    import matplotlib.pyplot as plt
    table = df.groupby(["cell", axis])[metric].mean().unstack("cell")
    table = table[[c for c in CELL_REGISTRY if c in table.columns]]
    values = list(table.index)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(table.columns), 1)
    x = np.arange(len(values))
    for k, cell in enumerate(table.columns):
        ax.bar(x + k * width, table[cell].values, width, label=CELL_LABELS[cell])
    ax.set(xticks=x + width * (len(table.columns) - 1) / 2,
           xticklabels=[str(v) for v in values], xlabel=axis, ylabel=metric,
           title=title or f"Effect of {axis} on {metric}")
    low = float(np.nanmin(table.values))
    ax.set_ylim(max(0.0, low - 0.05), min(1.0, float(np.nanmax(table.values)) + 0.03))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# One full experiment
# --------------------------------------------------------------------------- #
def run_experiment(cfg: ExperimentConfig, bundle: DataBundle, device: Optional[torch.device] = None,
                   results_dir: str = "results", save_figures: bool = True,
                   skip_if_done: bool = True, verbose: bool = True) -> Dict:
    """Train one model, evaluate it and save metrics, run details and figures.

    The test set is evaluated exactly once, with the weights of the best validation
    epoch (selection is done on the validation set only).
    """
    import json
    import matplotlib.pyplot as plt

    device = device or get_device()
    run_id = cfg.run_id
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "runs"), exist_ok=True)

    existing = load_metrics(results_dir)
    if skip_if_done and not existing.empty and run_id in set(existing["run_id"]):
        if verbose:
            print(f"[skip] {run_id} (already in {METRICS_FILE})")
        return existing[existing["run_id"] == run_id].iloc[0].to_dict()

    if verbose:
        print(f"[run ] {run_id} on {device}")
    set_seed(cfg.seed)
    num_classes = bundle.num_classes
    model = build_model(cfg, len(bundle.vocab), num_classes).to(device)
    train_loader, val_loader, test_loader = make_loaders(
        bundle, cfg.batch_size, cfg.seed, cfg.train_fraction)

    weights = None
    if cfg.use_class_weights:
        weights = torch.tensor(compute_class_weights(
            bundle.y["train"][train_loader.indices], num_classes), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(model.parameters(), cfg.optimizer, cfg.lr,
                                cfg.momentum, cfg.weight_decay)

    history, best_epoch, train_time = fit(model, train_loader, val_loader, criterion, optimizer,
                                          cfg, device, num_classes, verbose)
    val = run_epoch(model, val_loader, criterion, device, num_classes)
    test = run_epoch(model, test_loader, criterion, device, num_classes)

    row = asdict(cfg)
    row.update(
        run_id=run_id, num_classes=num_classes, max_len=bundle.cfg.max_len,
        vocab_size=len(bundle.vocab), n_train=train_loader.num_samples,
        n_val=len(bundle.y["val"]), n_test=len(bundle.y["test"]),
        split_fingerprint=bundle.fingerprint, epochs_run=len(history["train_loss"]),
        best_epoch=best_epoch, trainable_params=count_parameters(model),
        recurrent_params=count_parameters(model, only_recurrent=True),
        train_time_s=round(train_time, 2),
        time_per_epoch_s=round(train_time / max(len(history["train_loss"]), 1), 2),
        val_accuracy=val["accuracy"], val_macro_f1=val["macro_f1"],
        test_accuracy=test["accuracy"], test_macro_f1=test["macro_f1"], test_loss=test["loss"],
        device=str(device), timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    for name, score in zip(bundle.class_names, test["per_class_f1"]):
        row[f"test_f1_{name}"] = score

    details = {
        "config": asdict(cfg), "class_names": bundle.class_names, "history": history,
        "val": {k: val[k] for k in ("accuracy", "macro_f1", "per_class_f1", "confusion_matrix")},
        "test": {k: test[k] for k in ("accuracy", "macro_f1", "per_class_f1", "confusion_matrix")},
    }
    with open(os.path.join(results_dir, "runs", f"{run_id}.json"), "w", encoding="utf-8") as fh:
        json.dump(details, fh, indent=2)
    append_metrics(row, results_dir)

    if save_figures:
        label = CELL_LABELS[cfg.cell]
        fig = plot_training_curves(history, os.path.join(results_dir, "figures", f"curves_{run_id}.png"),
                                   title=f"{label} - training curves")
        plt.close(fig)
        fig = plot_confusion_matrix(test["confusion_matrix"], bundle.class_names,
                                    os.path.join(results_dir, "figures", f"cm_{run_id}.png"),
                                    title=f"{label} - test confusion matrix")
        plt.close(fig)

    if verbose:
        print(f"  -> test acc {test['accuracy']:.4f} | test macro-F1 {test['macro_f1']:.4f} | "
              f"params {row['trainable_params']:,} | time {train_time:.1f}s")
    return row


# --------------------------------------------------------------------------- #
# Smoke test on synthetic data
# --------------------------------------------------------------------------- #
def _reference_step(name: str, cell, x, h, c):
    """Independent, explicit (per-gate) implementation of the appendix equations."""
    hs = cell.hidden_size
    wx, b, wh = cell.x_lin.weight, cell.x_lin.bias, cell.h_lin.weight

    def lin(k):
        sl = slice(k * hs, (k + 1) * hs)
        return x @ wx[sl].T + h @ wh[sl].T + b[sl]

    sig, tanh = torch.sigmoid, torch.tanh
    if name == "lstm":
        i, f, g, o = sig(lin(0)), sig(lin(1)), tanh(lin(2)), sig(lin(3))
        c_new = f * c + i * g
    elif name == "cifg":
        f, g, o = sig(lin(0)), tanh(lin(1)), sig(lin(2))
        c_new = f * c + (1.0 - f) * g
    else:  # wmc
        i = sig(lin(0) + tanh(c @ cell.c_i.weight.T))
        f = sig(lin(1) + tanh(c @ cell.c_f.weight.T))
        g = tanh(lin(2))
        c_new = f * c + i * g
        o = sig(lin(3) + tanh(c_new @ cell.c_o.weight.T))
    return o * tanh(c_new), c_new


def smoke_test(device: Optional[torch.device] = None) -> None:
    """Fast self-check on synthetic data. Raises ``AssertionError`` on a real problem."""
    from .cells import build_cell

    device = device or get_device()
    torch.manual_seed(0)
    print(f"Smoke test on {device}\n")

    # 1. Cells match the equations, and parameter counts match the formulas.
    batch, d, hs = 4, 7, 5
    expected = {"lstm": 4 * hs * (d + hs) + 4 * hs,
                "cifg": 3 * hs * (d + hs) + 3 * hs,
                "wmc": 4 * hs * (d + hs) + 4 * hs + 3 * hs * hs}
    for name in CELL_REGISTRY:
        cell = build_cell(name, d, hs)
        x, h, c = torch.randn(batch, d), torch.randn(batch, hs), torch.randn(batch, hs)
        h_out, c_out = cell(x, (h, c))
        h_ref, c_ref = _reference_step(name, cell, x, h, c)
        assert h_out.shape == (batch, hs) and c_out.shape == (batch, hs)
        assert torch.allclose(h_out, h_ref, atol=1e-6) and torch.allclose(c_out, c_ref, atol=1e-6), \
            f"{name}: cell output differs from the reference equations"
        n_params = sum(p.numel() for p in cell.parameters())
        assert n_params == expected[name], f"{name}: {n_params} params, expected {expected[name]}"
        print(f"[ok] cell '{name}': equations match, {n_params} parameters (D={d}, H={hs})")

    # 2. No built-in nn.LSTM / nn.LSTMCell inside the models.
    for name in CELL_REGISTRY:
        model = TextClassifier(50, 3, name, 8, 6, num_layers=2, bidirectional=True)
        assert not any(isinstance(m, (nn.LSTM, nn.LSTMCell)) for m in model.modules())
    print("[ok] no nn.LSTM / nn.LSTMCell used")

    # 3. Padding must not change the output (all poolings, uni/bi-directional, 2 layers).
    lengths = [6, 4, 2]
    base = torch.zeros(3, 6, dtype=torch.long)
    for row, n in enumerate(lengths):
        base[row, :n] = torch.randint(2, 50, (n,))
    longer = torch.nn.functional.pad(base, (0, 6), value=0)
    for name in CELL_REGISTRY:
        for pooling in ("last", "mean", "max"):
            for bi in (False, True):
                model = TextClassifier(50, 3, name, 8, 6, num_layers=2, bidirectional=bi,
                                       dropout=0.0, pooling=pooling).eval()
                with torch.no_grad():
                    out_a, out_b = model(base), model(longer)
                assert out_a.shape == (3, 3)
                assert torch.allclose(out_a, out_b, atol=1e-5), \
                    f"padding changes the output ({name}, {pooling}, bidirectional={bi})"
    print("[ok] padding invariance (3 cells x 3 poolings x 2 directions, 2 layers)")

    # 4. Initialisations run and the forget bias is applied.
    for scheme in ("xavier", "orthogonal", "default"):
        model = TextClassifier(50, 3, "wmc", 8, 6)
        initialize_weights(model, scheme)
    model = TextClassifier(50, 3, "cifg", 8, 6)
    initialize_weights(model, "xavier", forget_bias=1.0)
    assert torch.allclose(model.rnn.fwd_layers[0].cell.x_lin.bias[:6], torch.ones(6))
    print("[ok] weight initialisations (xavier / orthogonal / default)")

    # 5. End-to-end run on synthetic data for the three cells.
    train_raw, test_raw = make_synthetic_dataframes(900, 300, seed=0)
    bundle = build_bundle_from_dataframes(
        train_raw, test_raw, DataConfig(max_len=30, min_freq=1, max_vocab=500, val_size=0.15))
    print("[ok] data bundle:", bundle.summary())

    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for name in CELL_REGISTRY:
            cfg = ExperimentConfig(cell=name, group="smoke", embed_dim=16, hidden_size=16,
                                   epochs=4, patience=4, batch_size=32, lr=3e-3,
                                   use_class_weights=True, seed=0)
            rows.append(run_experiment(cfg, bundle, device, results_dir=tmp,
                                       save_figures=(name == "wmc"), verbose=False))
            details = load_run_details(cfg.run_id, tmp)
            hist = details["history"]
            assert all(np.isfinite(v) for v in hist["train_loss"] + hist["val_loss"])
            trend = "decreased" if hist["train_loss"][-1] < hist["train_loss"][0] else "DID NOT DECREASE"
            print(f"[ok] run '{name}': train loss {hist['train_loss'][0]:.3f} -> "
                  f"{hist['train_loss'][-1]:.3f} ({trend}), test acc "
                  f"{rows[-1]['test_accuracy']:.3f}, macro-F1 {rows[-1]['test_macro_f1']:.3f}, "
                  f"params {rows[-1]['trainable_params']}")
        assert len({r["split_fingerprint"] for r in rows}) == 1
        metrics = load_metrics(tmp)
        assert len(metrics) == 3 and os.path.isfile(os.path.join(tmp, "figures", f"cm_{rows[-1]['run_id']}.png"))
        run_experiment(ExperimentConfig(cell="wmc", group="smoke", embed_dim=16, hidden_size=16,
                                        epochs=4, patience=4, batch_size=32, lr=3e-3,
                                        use_class_weights=True, seed=0),
                       bundle, device, results_dir=tmp, verbose=False)
        assert len(load_metrics(tmp)) == 3, "finished runs must be skipped"
        print("[ok] identical split for all models, metrics.csv + figures written, resume works")

    print("\nSmoke test passed.")


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train / test one LSTM-family model.")
    parser.add_argument("--smoke-test", action="store_true", help="run the synthetic self check")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--cell", default="lstm", choices=sorted(CELL_REGISTRY))
    parser.add_argument("--optimizer", default="adam", choices=OPTIMIZER_CHOICES)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", default="xavier", choices=("xavier", "orthogonal", "default"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.smoke_test:
        smoke_test()
        return

    bundle = build_bundle(DataConfig(data_dir=args.data_dir))
    print(bundle.summary())
    cfg = ExperimentConfig(cell=args.cell, optimizer=args.optimizer, lr=args.lr, init=args.init,
                           batch_size=args.batch_size, hidden_size=args.hidden_size,
                           epochs=args.epochs, train_fraction=args.train_fraction,
                           use_class_weights=args.class_weights, seed=args.seed, group="cli")
    run_experiment(cfg, bundle, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
