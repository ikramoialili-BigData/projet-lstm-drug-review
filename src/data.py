"""Data pipeline for the Drug Reviews (Drugs.com) dataset.

This module deliberately depends only on numpy and pandas (no torch), so that it can be
tested anywhere. Conversion to GPU tensors and batching live in train.py.

Pipeline: locate files -> detect separator -> read -> clean/tokenize -> map rating to class
-> ONE fixed stratified train/validation split -> vocabulary built on the training split only
-> pad/truncate to MAX_LEN -> cache as .npz.
"""
import hashlib
import html
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_ID, UNK_ID = 0, 1

TRAIN_STEM = "drugsComTrain_raw"
TEST_STEM = "drugsComTest_raw"
ACCEPTED_EXTENSIONS = (".tsv", ".csv", ".txt", "")
REQUIRED_COLUMNS = ("review", "rating")
EXPECTED_COLUMNS = ("uniqueID", "drugName", "condition", "review", "rating", "date", "usefulCount")


# =============================================================================================
# 1. Locating and reading the files
# =============================================================================================
class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset files cannot be found; the message explains how to fix it."""


def list_directory(path) -> List[str]:
    """Names of the entries of a directory (empty list if it does not exist)."""
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())


def find_dataset_file(data_dir, stem: str) -> Path:
    """Find '<stem>.tsv', '<stem>.csv', '<stem>.txt' or '<stem>' (case-insensitive) in data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        parent_listing = list_directory(data_dir.parent)
        raise DatasetNotFoundError(
            f"\nThe dataset folder does not exist: {data_dir}\n"
            f"Content of its parent folder ({data_dir.parent}): {parent_listing}\n"
            "How to fix it:\n"
            "  1. Check that Google Drive is mounted (drive.mount('/content/drive')).\n"
            "  2. Check that the folder exists in your Drive and that DATA_DIR (top of the notebook) "
            "points to it."
        )
    wanted = {(stem + ext).lower() for ext in ACCEPTED_EXTENSIONS}
    for entry in sorted(data_dir.iterdir()):
        if entry.is_file() and entry.name.lower() in wanted:
            return entry
    raise DatasetNotFoundError(
        f"\nCould not find '{stem}' (.tsv / .csv / .txt or no extension) in: {data_dir}\n"
        f"Files found in this folder: {list_directory(data_dir)}\n"
        "How to fix it:\n"
        f"  - Put {TRAIN_STEM}.* and {TEST_STEM}.* directly inside this folder (not in a sub-folder).\n"
        "  - Or edit DATA_DIR at the top of the notebook so that it points to the right folder."
    )


def detect_separator(path) -> str:
    """Return '\\t' or ',' by looking at the header line of the file."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    return "\t" if header.count("\t") >= header.count(",") else ","


def load_raw_dataframe(path) -> pd.DataFrame:
    """Read one raw file with the detected separator and validate its columns."""
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    if "uniqueID" not in df.columns and df.columns[0].lower().startswith("unnamed"):
        df = df.rename(columns={df.columns[0]: "uniqueID"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{Path(path).name}: missing column(s) {missing}. Columns found: {list(df.columns)}. "
            f"Expected columns: {list(EXPECTED_COLUMNS)} (detected separator: {sep!r})."
        )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    return df


# =============================================================================================
# 2. Text cleaning, tokenization and labels
# =============================================================================================
_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[!?]")


def clean_text(text: str) -> str:
    """Decode HTML entities (e.g. &#039; -> '), drop tags/URLs, lowercase, normalize spaces."""
    text = str(text)
    for _ in range(2):  # a second pass handles double-escaped entities such as &amp;#039;
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Clean a review and split it into word tokens (words with apostrophes, numbers, ! and ?)."""
    return _TOKEN_RE.findall(clean_text(text))


def validate_label_bins(bins: Sequence[Tuple[float, float, str]]) -> None:
    """Check that the bins are well formed and do not overlap."""
    if len(bins) < 2:
        raise ValueError("At least two classes are needed.")
    ordered = sorted(bins, key=lambda b: b[0])
    for (lo, hi, name) in ordered:
        if lo > hi:
            raise ValueError(f"Bin '{name}' has lower bound {lo} > upper bound {hi}.")
    for previous, current in zip(ordered[:-1], ordered[1:]):
        if current[0] <= previous[1]:
            raise ValueError(f"Bins '{previous[2]}' and '{current[2]}' overlap.")


def rating_to_label(ratings, bins=config.LABEL_BINS) -> np.ndarray:
    """Map ratings to class indices following `bins` (index = position in `bins`)."""
    validate_label_bins(bins)
    ratings = np.asarray(ratings, dtype=np.float64)
    labels = np.full(ratings.shape, -1, dtype=np.int64)
    for index, (lo, hi, _) in enumerate(bins):
        labels[(ratings >= lo) & (ratings <= hi)] = index
    if (labels < 0).any():
        bad = np.unique(ratings[labels < 0])
        raise ValueError(f"Some ratings are not covered by LABEL_BINS: {bad[:10]}")
    return labels


def class_names_from_bins(bins=config.LABEL_BINS) -> List[str]:
    return [b[2] for b in bins]


def class_counts(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)


# =============================================================================================
# 3. Vocabulary and encoding
# =============================================================================================
class Vocabulary:
    """Token <-> id mapping. Index 0 is <PAD>, index 1 is <UNK>."""

    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        if self.itos[:2] != [PAD_TOKEN, UNK_TOKEN]:
            raise ValueError("The first two vocabulary entries must be <PAD> and <UNK>.")
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, token_lists: Sequence[Sequence[str]], min_freq: int = 3,
              max_size: Optional[int] = None) -> "Vocabulary":
        """Build the vocabulary from TRAINING token lists only."""
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        kept = [(tok, n) for tok, n in counter.items() if n >= min_freq]
        kept.sort(key=lambda item: (-item[1], item[0]))  # frequency desc, then alphabetical (deterministic)
        if max_size is not None:
            kept = kept[: max(max_size - 2, 0)]
        return cls([PAD_TOKEN, UNK_TOKEN] + [tok for tok, _ in kept])

    def encode(self, tokens: Sequence[str], max_len: int) -> List[int]:
        """Token ids truncated to max_len (head truncation). Never empty (falls back to <UNK>)."""
        ids = [self.stoi.get(tok, UNK_ID) for tok in tokens[:max_len]]
        return ids if ids else [UNK_ID]


def encode_corpus(token_lists: Sequence[Sequence[str]], vocab: Vocabulary,
                  max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Encode, pad (right, with <PAD>) and truncate. Returns ids (N, max_len) and lengths (N,)."""
    ids = np.full((len(token_lists), max_len), PAD_ID, dtype=np.int32)
    lengths = np.zeros(len(token_lists), dtype=np.int64)
    for row, tokens in enumerate(token_lists):
        encoded = vocab.encode(tokens, max_len)
        ids[row, : len(encoded)] = encoded
        lengths[row] = len(encoded)
    return ids, lengths


# =============================================================================================
# 4. Splits and sub-sampling (stratified, deterministic)
# =============================================================================================
def stratified_split_indices(labels: np.ndarray, val_size: float,
                             seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) with the class proportions preserved in both parts."""
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be strictly between 0 and 1.")
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_parts, val_parts = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_size))
        n_val = min(max(n_val, 1 if len(idx) > 1 else 0), len(idx))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def stratified_subsample(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Indices of a random subset of about `fraction` of the data, with class proportions kept."""
    labels = np.asarray(labels)
    if fraction >= 1.0:
        return np.arange(len(labels))
    if fraction <= 0.0:
        raise ValueError("fraction must be > 0.")
    rng = np.random.default_rng(seed)
    parts = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        parts.append(idx[: max(int(round(len(idx) * fraction)), 1)])
    return np.sort(np.concatenate(parts))


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """'Balanced' weights: N / (n_classes * count_c). A perfectly balanced set gives all ones."""
    counts = class_counts(labels, n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return counts.sum() / (n_classes * counts)


# =============================================================================================
# 5. Prepared data (what the training code consumes) and caching
# =============================================================================================
@dataclass(frozen=True)
class DataConfig:
    """All settings that influence the prepared data (used as the cache key)."""
    max_len: int = config.MAX_LEN
    min_freq: int = config.MIN_FREQ
    max_vocab: Optional[int] = config.MAX_VOCAB
    val_size: float = config.VAL_SIZE
    split_seed: int = config.SPLIT_SEED
    label_bins: Tuple[Tuple[float, float, str], ...] = tuple(config.LABEL_BINS)


@dataclass
class SplitArrays:
    ids: np.ndarray      # (N, max_len) int32, 0 = <PAD>
    lengths: np.ndarray  # (N,) int64, number of real tokens (>= 1)
    labels: np.ndarray   # (N,) int64

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class PreparedData:
    splits: Dict[str, SplitArrays]
    vocab: Vocabulary
    class_names: List[str]
    cfg: DataConfig
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _cache_key(cfg: DataConfig, train_path: Path, test_path: Path) -> str:
    payload = {
        "cfg": asdict(cfg),
        "train": [train_path.name, train_path.stat().st_size],
        "test": [test_path.name, test_path.stat().st_size],
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def save_prepared(data: PreparedData, path) -> None:
    arrays = {}
    for name, split in data.splits.items():
        arrays[f"{name}_ids"] = split.ids
        arrays[f"{name}_lengths"] = split.lengths
        arrays[f"{name}_labels"] = split.labels
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        vocab=np.array(json.dumps(data.vocab.itos)),
        class_names=np.array(json.dumps(data.class_names)),
        cfg=np.array(json.dumps(asdict(data.cfg))),
        info=np.array(json.dumps(data.info)),
        **arrays,
    )
    tmp.replace(path)  # atomic: a disconnection cannot leave a half-written cache


def load_prepared(path) -> PreparedData:
    with np.load(path, allow_pickle=False) as npz:
        cfg_dict = json.loads(str(npz["cfg"]))
        cfg_dict["label_bins"] = tuple(tuple(b) for b in cfg_dict["label_bins"])
        splits = {
            name: SplitArrays(npz[f"{name}_ids"], npz[f"{name}_lengths"], npz[f"{name}_labels"])
            for name in ("train", "val", "test")
        }
        return PreparedData(
            splits=splits,
            vocab=Vocabulary(json.loads(str(npz["vocab"]))),
            class_names=json.loads(str(npz["class_names"])),
            cfg=DataConfig(**cfg_dict),
            info=json.loads(str(npz["info"])),
        )


def prepare_data(data_dir, cfg: Optional[DataConfig] = None, cache_dir=None,
                 force: bool = False, verbose: bool = True) -> PreparedData:
    """Build (or load from cache) the single train/val/test arrays shared by all models.

    * test    = the official test file;
    * train/val = the official train file split ONCE, stratified, with cfg.split_seed;
    * vocabulary = built on the training split only (validation and test never leak in).
    """
    cfg = cfg or DataConfig()
    train_path = find_dataset_file(data_dir, TRAIN_STEM)
    test_path = find_dataset_file(data_dir, TEST_STEM)
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"prepared_{_cache_key(cfg, train_path, test_path)}.npz"
        if cache_file.exists() and not force:
            if verbose:
                print(f"[data] Loaded prepared data from cache: {cache_file}")
            return load_prepared(cache_file)

    if verbose:
        print(f"[data] Reading {train_path.name} and {test_path.name} (separator detected: "
              f"{detect_separator(train_path)!r})")
    train_df = load_raw_dataframe(train_path)
    test_df = load_raw_dataframe(test_path)

    if verbose:
        print("[data] Cleaning and tokenizing reviews ...")
    train_tokens = [tokenize(t) for t in train_df["review"].tolist()]
    test_tokens = [tokenize(t) for t in test_df["review"].tolist()]
    train_labels_all = rating_to_label(train_df["rating"].to_numpy(), cfg.label_bins)
    test_labels = rating_to_label(test_df["rating"].to_numpy(), cfg.label_bins)

    train_idx, val_idx = stratified_split_indices(train_labels_all, cfg.val_size, cfg.split_seed)
    vocab = Vocabulary.build([train_tokens[i] for i in train_idx], cfg.min_freq, cfg.max_vocab)

    def make_split(tokens, labels, indices=None) -> SplitArrays:
        if indices is not None:
            tokens = [tokens[i] for i in indices]
            labels = labels[indices]
        ids, lengths = encode_corpus(tokens, vocab, cfg.max_len)
        return SplitArrays(ids, lengths, labels.astype(np.int64))

    splits = {
        "train": make_split(train_tokens, train_labels_all, train_idx),
        "val": make_split(train_tokens, train_labels_all, val_idx),
        "test": make_split(test_tokens, test_labels),
    }
    n_classes = len(cfg.label_bins)
    info = {
        "train_file": train_path.name,
        "test_file": test_path.name,
        "vocab_size": len(vocab),
        "sizes": {k: int(len(v)) for k, v in splits.items()},
        "class_counts": {k: class_counts(v.labels, n_classes).tolist() for k, v in splits.items()},
    }
    data = PreparedData(splits, vocab, class_names_from_bins(cfg.label_bins), cfg, info)
    if cache_file is not None:
        save_prepared(data, cache_file)
        if verbose:
            print(f"[data] Cache written: {cache_file}")
    if verbose:
        print(f"[data] Sizes: {info['sizes']} | vocabulary: {len(vocab)} tokens")
        print(f"[data] Class counts (train/val/test): {info['class_counts']}")
    return data


# =============================================================================================
# 6. Synthetic dataset (used by the sanity checks and the tests; NOT real data)
# =============================================================================================
def write_synthetic_dataset(data_dir, n_train: int = 900, n_test: int = 300, seed: int = 0,
                            sep: str = "\t", extension: str = ".tsv") -> Tuple[Path, Path]:
    """Write fake drugsComTrain_raw / drugsComTest_raw files with the real column layout.

    Sentiment words are correlated with the rating, so a model can learn something.
    Reviews contain HTML entities to exercise the cleaning code.
    """
    rng = np.random.default_rng(seed)
    positive = ["great", "effective", "love", "helped", "amazing", "relief", "works", "recommend"]
    negative = ["terrible", "awful", "nausea", "worse", "stopped", "pain", "horrible", "useless"]
    neutral = ["okay", "average", "mixed", "fine", "slowly", "some", "moderate", "unsure"]
    filler = ["the", "this", "medication", "i", "took", "for", "months", "and", "my", "doctor",
              "said", "it", "was", "side", "effects", "daily", "dose", "morning", "&#039;s", "&amp;"]
    rating_probs = np.array([14, 3, 3, 4, 6, 4, 8, 14, 18, 26], dtype=float)
    rating_probs /= rating_probs.sum()

    def make_frame(n: int, offset: int) -> pd.DataFrame:
        ratings = rng.choice(np.arange(1, 11), size=n, p=rating_probs)
        reviews = []
        for rating in ratings:
            pool = negative if rating <= 4 else neutral if rating <= 6 else positive
            length = int(rng.integers(5, 60))
            words = [rng.choice(pool) if rng.random() < 0.35 else rng.choice(filler)
                     for _ in range(length)]
            reviews.append('"' + " ".join(words) + " I&#039;ve tried it. &quot;ok&quot;\"")
        return pd.DataFrame({
            "uniqueID": np.arange(offset, offset + n),
            "drugName": rng.choice(["DrugA", "DrugB", "DrugC"], size=n),
            "condition": rng.choice(["Pain", "Anxiety", "Birth Control"], size=n),
            "review": reviews,
            "rating": ratings.astype(float),
            "date": "May 20, 2012",
            "usefulCount": rng.integers(0, 100, size=n),
        })

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / f"{TRAIN_STEM}{extension}"
    test_path = data_dir / f"{TEST_STEM}{extension}"
    make_frame(n_train, 0).to_csv(train_path, sep=sep, index=False)
    make_frame(n_test, 100000).to_csv(test_path, sep=sep, index=False)
    return train_path, test_path
