"""Data pipeline for the Drug Review classification task.

Responsibilities
----------------
* locate and read the raw CSV files (automatic path detection on Kaggle),
* clean the text (HTML entities, URLs, punctuation, lower-casing),
* map ratings to classes (configurable thresholds),
* create ONE fixed train/validation/test split (seeded, stratified validation),
* build the vocabulary on the TRAINING split only (``<PAD>`` = 0, ``<UNK>`` = 1),
* encode / pad / truncate reviews, and serve mini-batches,
* stratified sub-sampling of the training set (keeps class balance),
* class weights for the loss.

All the heavy lifting is done with numpy / pandas so the functions are easy to test.
"""

import hashlib
import html
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

TRAIN_FILE = "drugsComTrain_raw.csv"
TEST_FILE = "drugsComTest_raw.csv"
KAGGLE_DEFAULT_DIR = "/kaggle/input/kuc-hackathon-winter-2018"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Everything that defines the data side of an experiment."""

    data_dir: Optional[str] = None            # None -> automatic detection
    text_col: str = "review"
    rating_col: str = "rating"
    # Rating -> class mapping. A rating r gets class k = number of edges < r.
    # (4, 6) -> negative: 1-4, neutral: 5-6, positive: 7-10.
    label_edges: Tuple[float, ...] = (4, 6)
    class_names: Optional[Tuple[str, ...]] = None
    max_len: int = 200                        # pad / truncate length (tokens)
    min_freq: int = 2                         # min training frequency to enter the vocab
    max_vocab: int = 30000                    # includes <PAD> and <UNK>
    val_size: float = 0.1                     # fraction of the official train set
    split_seed: int = 42                      # fixed seed shared by all models
    lowercase: bool = True


def default_class_names(num_classes: int, class_names: Optional[Sequence[str]] = None) -> List[str]:
    """Names of the classes (defaults to negative/neutral/positive for 3 classes)."""
    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError("class_names length must equal the number of classes.")
        return list(class_names)
    if num_classes == 3:
        return ["negative", "neutral", "positive"]
    return [f"class_{k}" for k in range(num_classes)]


# --------------------------------------------------------------------------- #
# Reading the raw files
# --------------------------------------------------------------------------- #
def find_data_dir(data_dir: Optional[str] = None) -> str:
    """Return the folder containing the two raw CSV files.

    Search order: explicit argument, ``DRUG_DATA_DIR`` env variable, the usual
    Kaggle path, ``./data``, then a recursive walk of ``/kaggle/input`` and
    the current directory.
    """
    candidates = [
        data_dir,
        os.environ.get("DRUG_DATA_DIR"),
        KAGGLE_DEFAULT_DIR,
        "data",
        os.path.join("..", "data"),
    ]

    def has_files(folder: Optional[str]) -> bool:
        return bool(folder) and os.path.isfile(os.path.join(folder, TRAIN_FILE)) \
            and os.path.isfile(os.path.join(folder, TEST_FILE))

    for folder in candidates:
        if has_files(folder):
            return folder

    for root_dir in ("/kaggle/input", os.getcwd()):
        if not os.path.isdir(root_dir):
            continue
        for folder, _, files in os.walk(root_dir):
            if TRAIN_FILE in files and TEST_FILE in files:
                return folder

    raise FileNotFoundError(
        f"Could not find {TRAIN_FILE} and {TEST_FILE}. On Kaggle, add the dataset "
        "'UCI ML Drug Review dataset' (jessicali9530) to the notebook; locally, put "
        "the files in ./data or set the DRUG_DATA_DIR environment variable."
    )


def _read_table(path: str) -> pd.DataFrame:
    """Read a CSV (comma separated); fall back to tab separated if needed."""
    df = pd.read_csv(path)
    if "review" not in df.columns:
        df = pd.read_csv(path, sep="\t")
    return df


def load_raw(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official train and test tables without any processing."""
    folder = find_data_dir(data_dir)
    train_df = _read_table(os.path.join(folder, TRAIN_FILE))
    test_df = _read_table(os.path.join(folder, TEST_FILE))
    return train_df, test_df


# --------------------------------------------------------------------------- #
# Text cleaning, tokenization, labels
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_KEEP_RE = re.compile(r"[^a-z0-9' ]+")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: object, lowercase: bool = True) -> str:
    """Clean one review.

    Steps: decode HTML entities (``&#039;`` -> ``'``), drop URLs, lower-case,
    replace everything except letters, digits and apostrophes by a space and
    collapse whitespace. Non-string input (NaN) becomes an empty string.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    text = _KEEP_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(clean: str) -> List[str]:
    """Whitespace tokenization of an already cleaned review (edge apostrophes removed)."""
    tokens = (tok.strip("'") for tok in clean.split())
    return [tok for tok in tokens if tok]


def rating_to_label(ratings: Sequence[float], edges: Sequence[float] = (4, 6)) -> np.ndarray:
    """Map ratings to class ids. With edges (4, 6): <=4 -> 0, 5-6 -> 1, >=7 -> 2."""
    return np.searchsorted(np.asarray(edges, dtype=float),
                           np.asarray(ratings, dtype=float), side="left").astype(np.int64)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocabulary:
    """Token <-> id mapping with ``<PAD>`` = 0 and ``<UNK>`` = 1."""

    def __init__(self, itos: List[str]) -> None:
        self.itos = list(itos)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists: Sequence[Sequence[str]], min_freq: int = 2,
              max_size: int = 30000) -> "Vocabulary":
        """Build the vocabulary from (training) token lists only."""
        counter: Counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        # sort by (-count, token) so the result is fully deterministic
        ranked = sorted(((tok, c) for tok, c in counter.items() if c >= min_freq),
                        key=lambda item: (-item[1], item[0]))
        words = [tok for tok, _ in ranked[: max(max_size - 2, 0)]]
        return cls([PAD_TOKEN, UNK_TOKEN] + words)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str], max_len: int) -> List[int]:
        """Token ids, truncated to ``max_len``. Empty reviews become ``[<UNK>]``."""
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokens[:max_len]]
        return ids if ids else [UNK_IDX]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.itos, handle)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))


def encode_corpus(token_lists: Sequence[Sequence[str]], vocab: Vocabulary,
                  max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Encode, pad (right, with ``<PAD>``) and truncate a list of token lists.

    Returns ``X`` of shape (N, max_len), dtype int32, and ``lengths`` (N,), int64,
    the number of real (non-pad) tokens of each review (always >= 1).
    """
    X = np.full((len(token_lists), max_len), PAD_IDX, dtype=np.int32)
    lengths = np.zeros(len(token_lists), dtype=np.int64)
    for row, tokens in enumerate(token_lists):
        ids = vocab.encode(tokens, max_len)
        X[row, : len(ids)] = ids
        lengths[row] = len(ids)
    return X, lengths


# --------------------------------------------------------------------------- #
# Splits and bundle
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    """All the data needed by the experiments (identical for every model)."""

    cfg: DataConfig
    vocab: Vocabulary
    class_names: List[str]
    splits: Dict[str, pd.DataFrame]        # 'train' / 'val' / 'test' tables
    X: Dict[str, np.ndarray]               # encoded + padded reviews
    lengths: Dict[str, np.ndarray]         # true lengths
    y: Dict[str, np.ndarray]               # class ids
    fingerprint: str                       # hash of the split membership

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> str:
        parts = [f"{name}: {len(self.y[name]):,}" for name in ("train", "val", "test")]
        dist = {name: np.bincount(self.y[name], minlength=self.num_classes).tolist()
                for name in ("train", "val", "test")}
        return (f"splits -> {', '.join(parts)} | vocab: {len(self.vocab):,} | "
                f"max_len: {self.cfg.max_len} | classes: {self.class_names} | "
                f"class counts: {dist} | split fingerprint: {self.fingerprint}")


def split_fingerprint(splits: Dict[str, pd.DataFrame]) -> str:
    """Short hash of which review ids belong to which split (proof of identical splits)."""
    digest = hashlib.sha1()
    for name in sorted(splits):
        ids = splits[name]["uniqueID"].astype(str).tolist()
        digest.update(f"{name}:{','.join(sorted(ids))}|".encode("utf-8"))
    return digest.hexdigest()[:12]


def _prepare_table(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Drop unusable rows, add the cleaned text and the class label."""
    df = df.dropna(subset=[cfg.text_col, cfg.rating_col]).copy()
    if "uniqueID" not in df.columns:
        df["uniqueID"] = np.arange(len(df))
    df["review_clean"] = [clean_text(t, cfg.lowercase) for t in df[cfg.text_col]]
    df["label"] = rating_to_label(df[cfg.rating_col].values, cfg.label_edges)
    return df.reset_index(drop=True)


def build_bundle_from_dataframes(train_raw: pd.DataFrame, test_raw: pd.DataFrame,
                                 cfg: DataConfig) -> DataBundle:
    """Create the single fixed train/val/test split and encode everything.

    * test  = the official test table,
    * val   = ``cfg.val_size`` of the official train table, stratified, seeded,
    * train = the rest of the official train table,
    * vocab = built from the train split only.
    """
    train_full = _prepare_table(train_raw, cfg)
    test_df = _prepare_table(test_raw, cfg)

    train_df, val_df = train_test_split(
        train_full, test_size=cfg.val_size, stratify=train_full["label"],
        random_state=cfg.split_seed, shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    tokens = {name: [tokenize(t) for t in df["review_clean"]] for name, df in splits.items()}
    vocab = Vocabulary.build(tokens["train"], cfg.min_freq, cfg.max_vocab)

    X, lengths, y = {}, {}, {}
    for name, df in splits.items():
        X[name], lengths[name] = encode_corpus(tokens[name], vocab, cfg.max_len)
        y[name] = df["label"].values.astype(np.int64)

    num_classes = len(cfg.label_edges) + 1
    return DataBundle(
        cfg=cfg, vocab=vocab, class_names=default_class_names(num_classes, cfg.class_names),
        splits=splits, X=X, lengths=lengths, y=y, fingerprint=split_fingerprint(splits),
    )


def build_bundle(cfg: Optional[DataConfig] = None) -> DataBundle:
    """Read the raw files from disk and build the :class:`DataBundle`."""
    cfg = cfg or DataConfig()
    train_raw, test_raw = load_raw(cfg.data_dir)
    return build_bundle_from_dataframes(train_raw, test_raw, cfg)


# --------------------------------------------------------------------------- #
# Sub-sampling and class weights
# --------------------------------------------------------------------------- #
def stratified_subsample_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Indices of a class-balanced random subset (same class proportions as ``labels``)."""
    labels = np.asarray(labels)
    n = len(labels)
    if fraction >= 1.0:
        return np.arange(n)
    if fraction <= 0.0:
        raise ValueError("fraction must be in (0, 1].")
    chosen, _ = train_test_split(np.arange(n), train_size=fraction, stratify=labels,
                                 random_state=seed, shuffle=True)
    return np.sort(chosen)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency weights ``N / (K * n_k)`` (a perfectly balanced set gives all 1)."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return (counts.sum() / (num_classes * counts)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Mini-batches
# --------------------------------------------------------------------------- #
class BatchLoader:
    """Light-weight mini-batch iterator over pre-encoded arrays.

    Each batch is cropped to the longest review it contains, so short batches
    are cheap. Yields ``(x, lengths, y)`` tensors on the CPU.
    """

    def __init__(self, X: np.ndarray, lengths: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool = False, seed: int = 0, indices: Optional[np.ndarray] = None) -> None:
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.lengths = torch.from_numpy(np.ascontiguousarray(lengths))
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.indices = np.arange(len(y)) if indices is None else np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    @property
    def num_samples(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        order = self.rng.permutation(self.indices) if self.shuffle else self.indices
        for start in range(0, len(order), self.batch_size):
            batch = torch.from_numpy(order[start:start + self.batch_size].astype(np.int64))
            lengths = self.lengths[batch]
            max_len = int(lengths.max())
            yield self.X[batch, :max_len], lengths, self.y[batch]


def make_loaders(bundle: DataBundle, batch_size: int, seed: int = 0,
                 train_fraction: float = 1.0) -> Tuple[BatchLoader, BatchLoader, BatchLoader]:
    """Build train / validation / test loaders.

    ``train_fraction < 1`` keeps a stratified random subset of the training split.
    The subset depends only on ``bundle.cfg.split_seed`` (not on the training seed),
    so every model sees exactly the same training examples. Validation and test
    sets are never sub-sampled.
    """
    subset = stratified_subsample_indices(bundle.y["train"], train_fraction, bundle.cfg.split_seed)
    train_loader = BatchLoader(bundle.X["train"], bundle.lengths["train"], bundle.y["train"],
                               batch_size, shuffle=True, seed=seed, indices=subset)
    eval_bs = max(batch_size, 256)
    val_loader = BatchLoader(bundle.X["val"], bundle.lengths["val"], bundle.y["val"], eval_bs)
    test_loader = BatchLoader(bundle.X["test"], bundle.lengths["test"], bundle.y["test"], eval_bs)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# Synthetic data (smoke tests only)
# --------------------------------------------------------------------------- #
def make_synthetic_dataframes(n_train: int = 900, n_test: int = 300,
                              seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tiny fake dataset with the same columns as the real one.

    Reviews contain sentiment words correlated with the rating, HTML entities and
    imbalanced ratings, so the full pipeline can be exercised without the real data.
    """
    rng = np.random.RandomState(seed)
    positive = ["great", "excellent", "works", "relief", "love", "happy", "amazing"]
    negative = ["terrible", "awful", "nausea", "pain", "worse", "hate", "dizzy"]
    neutral = ["okay", "average", "unchanged", "mixed", "fine", "moderate"]
    filler = ["the", "pill", "doctor", "week", "taking", "day", "i", "it", "my", "and", "for"]
    pools = [negative, neutral, positive]
    rating_probs = np.array([4, 2, 2, 2, 3, 3, 6, 8, 10, 12], dtype=float)
    rating_probs /= rating_probs.sum()

    def make(n: int, offset: int) -> pd.DataFrame:
        ratings = rng.choice(np.arange(1, 11), size=n, p=rating_probs)
        labels = rating_to_label(ratings)
        reviews = []
        for label in labels:
            length = rng.randint(6, 40)
            words = [rng.choice(pools[label]) if rng.rand() < 0.35 else rng.choice(filler)
                     for _ in range(length)]
            text = " ".join(words)
            if rng.rand() < 0.3:
                text = "&quot;" + text.capitalize() + " &#039;s done&quot;"
            reviews.append(text)
        return pd.DataFrame({
            "uniqueID": np.arange(offset, offset + n),
            "drugName": rng.choice(["DrugA", "DrugB", "DrugC"], size=n),
            "condition": rng.choice(["Pain", "Anxiety", "Acne"], size=n),
            "review": reviews, "rating": ratings.astype(float),
            "date": "January 1, 2018", "usefulCount": rng.randint(0, 50, size=n),
        })

    return make(n_train, 0), make(n_test, 100000)
