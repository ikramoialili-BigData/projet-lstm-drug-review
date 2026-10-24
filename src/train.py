"""Training, evaluation, trainable-parameter counting, resumable experiments and plots.

Colab-friendly design:
* every epoch writes a checkpoint (atomically) so a disconnected run resumes where it stopped;
* every finished experiment is appended to results/metrics.csv and is skipped on the next call;
* all data tensors live on the device once (see DeviceData), batches are cheap slices.
"""
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .cells import CELL_LONG_NAMES, CELL_NAMES
from .data import (PreparedData, SplitArrays, compute_class_weights, stratified_subsample)
from .model import RNNClassifier

OPTIMIZERS = ("adam", "sgd", "rmsprop")
# Reasonable default learning rate of each optimizer (SGD needs a much larger one).
DEFAULT_LR = {"adam": 1e-3, "rmsprop": 1e-3, "sgd": 0.05}


# =============================================================================================
# 1. Reproducibility and devices
# =============================================================================================
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU when available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (embedding + cells + output layer)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================================
# 2. Experiment configuration
# =============================================================================================
@dataclass(frozen=True)
class ExperimentConfig:
    cell: str = "lstm"              # 'lstm' | 'cifg' | 'wmc'
    embed_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    pooling: str = "last"           # 'last' | 'mean' | 'max'
    dropout: float = 0.3
    optimizer: str = "adam"         # 'adam' | 'sgd' | 'rmsprop'
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9           # used by SGD only
    init: str = "xavier"            # 'default' | 'xavier' | 'orthogonal'
    forget_bias: float = 1.0
    batch_size: int = 128
    epochs: int = 8                 # maximum number of epochs
    patience: int = 2               # early stopping on validation macro-F1
    grad_clip: float = 1.0
    use_class_weights: bool = False
    train_fraction: float = 1.0     # stratified sub-sampling of the training split
    seed: int = 0
    stage: str = "base"             # free label (base / ofat / final), not part of the experiment id


def exp_id(cfg: ExperimentConfig) -> str:
    """Deterministic, filename-safe identifier of an experiment (the 'stage' label is ignored)."""
    parts = [
        cfg.cell, f"h{cfg.hidden_size}", f"L{cfg.num_layers}{'bi' if cfg.bidirectional else ''}",
        cfg.pooling, cfg.optimizer, f"lr{cfg.lr:g}", cfg.init, f"bs{cfg.batch_size}",
        "cw" if cfg.use_class_weights else "nocw", f"frac{cfg.train_fraction:g}", f"seed{cfg.seed}",
    ]
    rest = {k: v for k, v in asdict(cfg).items() if k != "stage"}
    digest = hashlib.md5(json.dumps(rest, sort_keys=True).encode()).hexdigest()[:6]
    return "_".join(parts) + "_" + digest


def config_from_row(row: pd.Series, **overrides) -> ExperimentConfig:
    """Rebuild an ExperimentConfig from a row of metrics.csv (used to re-run the best settings)."""
    values = {}
    for f in fields(ExperimentConfig):
        if f.name in row.index and not pd.isna(row[f.name]):
            raw = row[f.name]
            if f.type in (bool, "bool"):
                values[f.name] = (raw.strip().lower() == "true") if isinstance(raw, str) else bool(raw)
            elif f.type in (int, "int"):
                values[f.name] = int(raw)
            elif f.type in (float, "float"):
                values[f.name] = float(raw)
            else:
                values[f.name] = str(raw)
    values.update(overrides)
    return ExperimentConfig(**values)


def build_model(cfg: ExperimentConfig, vocab_size: int, n_classes: int) -> RNNClassifier:
    return RNNClassifier(
        vocab_size=vocab_size, n_classes=n_classes, cell=cfg.cell, embed_dim=cfg.embed_dim,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, bidirectional=cfg.bidirectional,
        pooling=cfg.pooling, dropout=cfg.dropout, init=cfg.init, forget_bias=cfg.forget_bias,
    )


def make_optimizer(parameters, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. Choose among {OPTIMIZERS}.")


def make_ofat_configs(base: ExperimentConfig, factors: Dict[str, Sequence],
                      cells: Sequence[str] = CELL_NAMES) -> List[ExperimentConfig]:
    """One-factor-at-a-time grid: the base setting plus every single change of one factor.

    Order: for each setting, all cells are listed together, so that if your Colab time runs
    out you still have a complete comparison of the three cells on the first settings.
    When the factor is 'optimizer', the learning rate is switched to DEFAULT_LR[optimizer].
    Duplicates (for example the base value repeated inside a factor list) are removed.
    """
    settings = [("base", None, base)]
    for name, values in factors.items():
        for value in values:
            if name == "optimizer":
                settings.append((name, value, replace(base, optimizer=value, lr=DEFAULT_LR[value])))
            else:
                settings.append((name, value, replace(base, **{name: value})))
    configs, seen = [], set()
    for factor, _, setting in settings:
        for cell in cells:
            cfg = replace(setting, cell=cell, stage="base" if factor == "base" else "ofat")
            if exp_id(cfg) not in seen:
                seen.add(exp_id(cfg))
                configs.append(cfg)
    return configs


# =============================================================================================
# 3. Data on the device and batching
# =============================================================================================
@dataclass
class TorchSplit:
    ids: torch.Tensor       # (N, max_len) int32
    lengths: torch.Tensor   # (N,) int64
    labels: torch.Tensor    # (N,) int64

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def _to_device(split: SplitArrays, device) -> TorchSplit:
    return TorchSplit(torch.from_numpy(split.ids).to(device),
                      torch.from_numpy(split.lengths).to(device),
                      torch.from_numpy(split.labels).to(device))


class DeviceData:
    """Holds the train/val/test tensors on the device. Sub-sampled training sets are cached."""

    def __init__(self, data: PreparedData, device):
        self.data = data
        self.device = device
        self.val = _to_device(data.splits["val"], device)
        self.test = _to_device(data.splits["test"], device)
        self._train_full = _to_device(data.splits["train"], device)
        self._subsets: Dict[float, TorchSplit] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.data.vocab)

    @property
    def n_classes(self) -> int:
        return self.data.n_classes

    @property
    def class_names(self) -> List[str]:
        return self.data.class_names

    def train(self, fraction: float = 1.0) -> TorchSplit:
        """Training split, or a STRATIFIED subset of it. The subset depends only on the fixed
        split seed and the fraction, so every model sees exactly the same training samples."""
        if fraction >= 1.0:
            return self._train_full
        key = round(float(fraction), 6)
        if key not in self._subsets:
            idx_np = stratified_subsample(self.data.splits["train"].labels, key, self.data.cfg.split_seed)
            idx = torch.from_numpy(idx_np).to(self.device)
            full = self._train_full
            self._subsets[key] = TorchSplit(full.ids[idx], full.lengths[idx], full.labels[idx])
        return self._subsets[key]


def iterate_batches(split: TorchSplit, batch_size: int, shuffle: bool,
                    seed: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (ids, lengths, labels). Columns that are padding for the whole batch are trimmed."""
    n = len(split)
    device = split.ids.device
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        order = torch.randperm(n, generator=generator).to(device)
    else:
        order = torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        lengths = split.lengths[idx]
        longest = int(lengths.max())
        yield split.ids[idx][:, :longest].long(), lengths, split.labels[idx]


# =============================================================================================
# 4. Metrics and evaluation
# =============================================================================================
def confusion_matrix_np(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    index = labels.astype(np.int64) * n_classes + preds.astype(np.int64)
    return np.bincount(index, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, object]:
    """Accuracy, macro-F1 and per-class precision/recall/F1 from a confusion matrix."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"accuracy": float(tp.sum() / max(cm.sum(), 1.0)), "macro_f1": float(f1.mean()),
            "precision": precision, "recall": recall, "f1_per_class": f1}


@torch.no_grad()
def evaluate(model: nn.Module, split: TorchSplit, n_classes: int,
             batch_size: int = 512) -> Dict[str, object]:
    """Unweighted cross-entropy, accuracy, macro-F1, per-class F1 and confusion matrix."""
    model.eval()
    loss_sum = torch.zeros((), device=split.ids.device)
    all_preds, all_labels = [], []
    for ids, lengths, labels in iterate_batches(split, batch_size, shuffle=False):
        logits = model(ids, lengths)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum")
        all_preds.append(logits.argmax(dim=1))
        all_labels.append(labels)
    preds = torch.cat(all_preds).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    cm = confusion_matrix_np(preds, labels, n_classes)
    stats = metrics_from_confusion(cm)
    stats.update({"loss": float(loss_sum.item() / len(labels)), "confusion": cm})
    return stats


# =============================================================================================
# 5. Training one experiment (with per-epoch checkpoints and resume)
# =============================================================================================
def _atomic_save(obj, path: Path) -> None:
    """Write to a temporary file, then rename: a disconnection cannot corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj)}")


def train_experiment(cfg: ExperimentConfig, device_data: DeviceData, device,
                     checkpoint_dir, verbose: bool = True,
                     keep_last_checkpoint: bool = False) -> Tuple[dict, List[dict], np.ndarray]:
    """Train one model with early stopping, resuming from a checkpoint if one exists.

    Returns (row, history, test_confusion_matrix). `row` is one line of metrics.csv.
    """
    checkpoint_dir = Path(checkpoint_dir)
    eid = exp_id(cfg)
    last_path = checkpoint_dir / f"{eid}_last.pt"
    best_path = checkpoint_dir / f"{eid}_best.pt"

    set_seed(cfg.seed)
    train_split = device_data.train(cfg.train_fraction)
    model = build_model(cfg, device_data.vocab_size, device_data.n_classes).to(device)
    optimizer = make_optimizer(model.parameters(), cfg)
    n_params = count_trainable_parameters(model)
    n_params_cell = sum(p.numel() for p in model.cell_parameters())

    weights = None
    if cfg.use_class_weights:
        weights = torch.tensor(compute_class_weights(train_split.labels.cpu().numpy(),
                                                     device_data.n_classes),
                               dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history: List[dict] = []
    start_epoch, best_f1, best_epoch, bad_epochs, elapsed = 0, -1.0, -1, 0, 0.0
    if last_path.exists():
        try:
            state = torch.load(last_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            history, start_epoch = state["history"], state["epoch"]
            best_f1, best_epoch = state["best_f1"], state["best_epoch"]
            bad_epochs, elapsed = state["bad_epochs"], state["elapsed"]
            if verbose:
                print(f"  [resume] {eid}: restarting after epoch {start_epoch}")
        except Exception as error:  # corrupted file: start again from scratch
            print(f"  [warning] could not read {last_path.name} ({error}); training from scratch.")
            history, start_epoch, best_f1, best_epoch, bad_epochs, elapsed = [], 0, -1.0, -1, 0, 0.0

    status = "ok"
    for epoch in range(start_epoch, cfg.epochs):
        if bad_epochs >= cfg.patience:
            break
        t0 = time.time()
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0
        for ids, lengths, labels in iterate_batches(train_split, cfg.batch_size, shuffle=True,
                                                    seed=cfg.seed * 1000 + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += loss.detach() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum()
            seen += labels.size(0)
        train_loss, train_acc = loss_sum.item() / seen, correct.item() / seen
        if not math.isfinite(train_loss):
            status = "diverged"
            if verbose:
                print(f"  [diverged] {eid}: non-finite training loss at epoch {epoch + 1}")
            break
        val = evaluate(model, device_data.val, device_data.n_classes)
        epoch_time = time.time() - t0
        elapsed += epoch_time
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val["loss"], "val_acc": val["accuracy"],
                        "val_macro_f1": val["macro_f1"], "time_s": epoch_time})
        if val["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch + 1, 0
            _atomic_save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "history": history, "epoch": epoch + 1, "best_f1": best_f1,
                      "best_epoch": best_epoch, "bad_epochs": bad_epochs, "elapsed": elapsed},
                     last_path)
        if verbose:
            print(f"  epoch {epoch + 1:02d}/{cfg.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f}"
                  f" | val loss {val['loss']:.4f} acc {val['accuracy']:.4f} macro-F1 {val['macro_f1']:.4f}"
                  f" | {epoch_time:.1f}s")

    row = {"exp_id": eid, "stage": cfg.stage}
    row.update({k: v for k, v in asdict(cfg).items() if k != "stage"})
    row.update({"n_params": n_params, "n_params_cell": n_params_cell, "epochs_run": len(history),
                "best_epoch": best_epoch, "train_time_s": round(elapsed, 2), "status": status})
    nan = float("nan")
    cm = np.zeros((device_data.n_classes, device_data.n_classes), dtype=np.int64)
    if best_path.exists() and status == "ok":
        model.load_state_dict(torch.load(best_path, map_location=device))
        val = evaluate(model, device_data.val, device_data.n_classes)
        test = evaluate(model, device_data.test, device_data.n_classes)
        cm = test["confusion"]
        row.update({"val_loss": val["loss"], "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
                    "test_loss": test["loss"], "test_acc": test["accuracy"],
                    "test_macro_f1": test["macro_f1"]})
        for name, score in zip(device_data.class_names, test["f1_per_class"]):
            row[f"test_f1_{name}"] = float(score)
    else:
        row.update({"val_loss": nan, "val_acc": nan, "val_macro_f1": nan, "test_loss": nan,
                    "test_acc": nan, "test_macro_f1": nan})
        for name in device_data.class_names:
            row[f"test_f1_{name}"] = nan
        if status == "ok":
            row["status"] = "no_epoch"
    row["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if not keep_last_checkpoint and last_path.exists():
        last_path.unlink()  # saves Drive space; the best weights stay in *_best.pt
    return row, history, cm


# =============================================================================================
# 6. Resumable experiment loop and results files
# =============================================================================================
def results_paths(results_dir=None) -> Dict[str, Path]:
    root = Path(results_dir) if results_dir is not None else config.RESULTS_DIR
    return {"root": root, "metrics": root / "metrics.csv", "runs": root / "runs",
            "figures": root / "figures", "run_figures": root / "figures" / "runs",
            "checkpoints": root / "checkpoints"}


def load_metrics(metrics_csv=None) -> pd.DataFrame:
    """Read metrics.csv (empty DataFrame if it does not exist yet)."""
    path = Path(metrics_csv) if metrics_csv is not None else config.METRICS_CSV
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def completed_ids(metrics_csv) -> set:
    df = load_metrics(metrics_csv)
    return set(df["exp_id"]) if "exp_id" in df.columns else set()


def append_metrics_row(row: dict, metrics_csv) -> None:
    """Append one row to metrics.csv with an atomic rewrite (safe against disconnections)."""
    metrics_csv = Path(metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    df = pd.concat([pd.read_csv(metrics_csv), new], ignore_index=True) if metrics_csv.exists() else new
    tmp = metrics_csv.with_name(metrics_csv.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, metrics_csv)


def run_experiments(configs: Sequence[ExperimentConfig], device_data: DeviceData, device,
                    results_dir=None, save_run_figures: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Run every configuration that is not yet in metrics.csv. Safe to call again after a crash.

    To re-run an experiment on purpose: delete its row in results/metrics.csv.
    """
    paths = results_paths(results_dir)
    done = completed_ids(paths["metrics"])
    todo = []
    for cfg in configs:
        if exp_id(cfg) not in {exp_id(c) for c in todo}:
            todo.append(cfg)
    n_skipped = sum(exp_id(c) in done for c in todo)
    print(f"[experiments] {len(todo)} configurations, {n_skipped} already done, "
          f"{len(todo) - n_skipped} to run.")
    for k, cfg in enumerate(todo, start=1):
        eid = exp_id(cfg)
        if eid in done:
            if verbose:
                print(f"[{k}/{len(todo)}] skip (already in metrics.csv): {eid}")
            continue
        print(f"[{k}/{len(todo)}] {CELL_LONG_NAMES[cfg.cell]} | {eid}")
        try:
            row, history, cm = train_experiment(cfg, device_data, device, paths["checkpoints"], verbose)
        except torch.cuda.OutOfMemoryError:
            print("  [out of memory] skipped. Reduce batch_size/hidden_size/max_len and re-run this cell.")
            torch.cuda.empty_cache()
            continue
        paths["runs"].mkdir(parents=True, exist_ok=True)
        with open(paths["runs"] / f"{eid}.json", "w") as handle:
            json.dump({"config": asdict(cfg), "history": history, "test_confusion": cm,
                       "class_names": device_data.class_names, "row": row}, handle,
                      default=_json_default, indent=1)
        if save_run_figures and history:
            plot_training_curves(history, title=eid, path=paths["run_figures"] / f"{eid}_curves.png")
            if row["status"] == "ok":
                plot_confusion_matrix(cm, device_data.class_names, title=f"Test confusion - {eid}",
                                      path=paths["run_figures"] / f"{eid}_confusion.png")
        append_metrics_row(row, paths["metrics"])
        done.add(eid)
        print(f"  -> test acc {row['test_acc']:.4f} | test macro-F1 {row['test_macro_f1']:.4f} | "
              f"params {row['n_params']:,} | time {row['train_time_s']:.0f}s | status {row['status']}")
    return load_metrics(paths["metrics"])


def select_best_per_cell(metrics: pd.DataFrame, seed: int, metric: str = "val_macro_f1",
                         stages: Sequence[str] = ("base", "ofat")) -> Dict[str, ExperimentConfig]:
    """Best configuration of each cell according to the VALIDATION metric (never the test set)."""
    df = metrics[(metrics["status"] == "ok") & (metrics["seed"] == seed) & metrics["stage"].isin(stages)]
    best = {}
    for cell in CELL_NAMES:
        subset = df[df["cell"] == cell]
        if len(subset):
            best[cell] = config_from_row(subset.sort_values(metric, ascending=False).iloc[0])
    return best


def benchmark_speed(cfg: ExperimentConfig, device_data: DeviceData, device,
                    n_batches: int = 15) -> Dict[str, float]:
    """Time a few training steps to estimate the duration of one epoch (no result is saved)."""
    set_seed(cfg.seed)
    model = build_model(cfg, device_data.vocab_size, device_data.n_classes).to(device)
    optimizer = make_optimizer(model.parameters(), cfg)
    criterion = nn.CrossEntropyLoss()
    split = device_data.train(cfg.train_fraction)
    model.train()
    times = []
    for k, (ids, lengths, labels) in enumerate(iterate_batches(split, cfg.batch_size, True, seed=123)):
        if k >= n_batches + 2:
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(ids, lengths), labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    per_batch = float(np.mean(times[2:])) if len(times) > 2 else float(np.mean(times))  # skip warm-up
    n_batches_epoch = math.ceil(len(split) / cfg.batch_size)
    return {"seconds_per_batch": per_batch, "batches_per_epoch": n_batches_epoch,
            "seconds_per_epoch": per_batch * n_batches_epoch}


# =============================================================================================
# 7. Plots (matplotlib is imported lazily; every function can save to a file)
# =============================================================================================
def _finish_figure(fig, path=None, show: bool = True) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_curves(history: List[dict], title: str = "", path=None, show: bool = False) -> None:
    """Loss (train/val) and accuracy / macro-F1 per epoch."""
    import matplotlib.pyplot as plt
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "s-", label="validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [h["train_acc"] for h in history], "o-", label="train accuracy")
    axes[1].plot(epochs, [h["val_acc"] for h in history], "s-", label="validation accuracy")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], "^--", label="validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy and macro-F1")
    axes[1].legend()
    fig.suptitle(title, fontsize=9)
    _finish_figure(fig, path, show)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str = "",
                          path=None, show: bool = False, normalize: bool = True) -> None:
    """Confusion matrix (rows = true class). Normalized by row by default (= recall per class)."""
    import matplotlib.pyplot as plt
    values = cm.astype(np.float64)
    if normalize:
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}\n({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    _finish_figure(fig, path, show)


def plot_grouped_bars(table: pd.DataFrame, ylabel: str, title: str, path=None,
                      errors: Optional[pd.DataFrame] = None, show: bool = True,
                      ylim: Optional[Tuple[float, float]] = None) -> None:
    """Grouped bar chart. `table`: index = x groups (e.g. a hyperparameter value), columns = cells."""
    import matplotlib.pyplot as plt
    n_groups, n_series = table.shape
    width = 0.8 / max(n_series, 1)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.6 * n_groups + 2), 3.8))
    for k, column in enumerate(table.columns):
        positions = np.arange(n_groups) + (k - (n_series - 1) / 2) * width
        err = errors[column].to_numpy() if errors is not None and column in errors else None
        label = CELL_LONG_NAMES.get(column, column)
        ax.bar(positions, table[column].to_numpy(), width=width, yerr=err, capsize=3, label=label)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([str(v) for v in table.index])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _finish_figure(fig, path, show)
