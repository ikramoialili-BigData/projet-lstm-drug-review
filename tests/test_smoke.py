"""Smoke tests on SYNTHETIC data (no real dataset needed). Run from the project root:

    python -m pytest tests -v

They check shapes, the gate equations, padding invariance, parameter counts, the resumable
experiment loop and the data pipeline. They do not measure model quality.
"""
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402  (no torch needed)


# ------------------------------------------------------------------------------------------
# Data pipeline (no torch)
# ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_dir():
    folder = Path(tempfile.mkdtemp())
    D.write_synthetic_dataset(folder, n_train=600, n_test=200)
    return folder


def test_clean_text_decodes_html_entities():
    assert "'" in D.clean_text("I&#039;ve tried &amp; it")
    assert "&#039;" not in D.clean_text("it&#039;s")
    assert D.tokenize("I&#039;ve been fine!") == ["i've", "been", "fine", "!"]


def test_label_mapping_default_and_custom():
    labels = D.rating_to_label([1, 4, 5, 6, 7, 10])
    assert labels.tolist() == [0, 0, 1, 1, 2, 2]
    binary = D.rating_to_label([1, 5, 6, 10], bins=((1, 5, "neg"), (6, 10, "pos")))
    assert binary.tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        D.rating_to_label([11])


def test_separator_detection_and_csv(tmp_path):
    D.write_synthetic_dataset(tmp_path, 50, 20, sep=",", extension=".csv")
    data = D.prepare_data(tmp_path, verbose=False)
    assert data.info["sizes"]["test"] == 20


def test_missing_files_give_helpful_error(tmp_path):
    (tmp_path / "something_else.txt").write_text("x")
    with pytest.raises(D.DatasetNotFoundError) as error:
        D.prepare_data(tmp_path, verbose=False)
    assert "something_else.txt" in str(error.value)


def test_prepared_data_is_consistent_and_cached(synthetic_dir):
    cache = synthetic_dir / "cache"
    first = D.prepare_data(synthetic_dir, cache_dir=cache, verbose=False)
    second = D.prepare_data(synthetic_dir, cache_dir=cache, verbose=False)
    assert np.array_equal(first.splits["train"].ids, second.splits["train"].ids)
    for split in first.splits.values():
        assert split.ids.shape[1] == first.cfg.max_len
        assert (split.lengths >= 1).all()
        for row in range(5):  # everything after `length` is <PAD>
            assert (split.ids[row, split.lengths[row]:] == D.PAD_ID).all()
    assert first.vocab.itos[:2] == [D.PAD_TOKEN, D.UNK_TOKEN]


def test_stratified_split_keeps_proportions(synthetic_dir):
    data = D.prepare_data(synthetic_dir, verbose=False)
    train = np.bincount(data.splits["train"].labels) / len(data.splits["train"])
    val = np.bincount(data.splits["val"].labels) / len(data.splits["val"])
    assert np.allclose(train, val, atol=0.03)
    sub = D.stratified_subsample(data.splits["train"].labels, 0.3, seed=1)
    ratio = np.bincount(data.splits["train"].labels[sub]) / len(sub)
    assert np.allclose(ratio, train, atol=0.03)


def test_vocabulary_uses_training_split_only(synthetic_dir):
    data = D.prepare_data(synthetic_dir, verbose=False)
    assert len(data.vocab) <= data.cfg.max_vocab


# ------------------------------------------------------------------------------------------
# Everything below needs torch
# ------------------------------------------------------------------------------------------
torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.cells import CELL_NAMES, build_cell  # noqa: E402
from src.model import RNNClassifier, reverse_padded  # noqa: E402
from src import train as T  # noqa: E402

H, E, B = 6, 5, 4


def test_no_builtin_lstm_in_sources():
    pattern = re.compile(r"nn\.(LSTM|GRU|RNN)\s*\(")
    for path in (ROOT / "src").glob("*.py"):
        assert not pattern.search(path.read_text()), f"built-in recurrent layer used in {path.name}"


@pytest.mark.parametrize("name", CELL_NAMES)
def test_cell_step_shapes(name):
    cell = build_cell(name, E, H)
    h, c = cell(torch.randn(B, E))
    assert h.shape == (B, H) and c.shape == (B, H)
    h2, c2 = cell(torch.randn(B, E), (h, c))
    assert torch.isfinite(h2).all() and torch.isfinite(c2).all()


def _block(weight, k):
    return weight[k * H:(k + 1) * H]


def test_vanilla_matches_equations():
    cell = build_cell("lstm", E, H)
    x, h, c = torch.randn(B, E), torch.randn(B, H), torch.randn(B, H)
    Wx, b, Wh = cell.x_linear.weight, cell.x_linear.bias, cell.h_linear.weight

    def gate(k, act):
        return act(x @ _block(Wx, k).T + h @ _block(Wh, k).T + b[k * H:(k + 1) * H])

    i, f, o, g = gate(0, torch.sigmoid), gate(1, torch.sigmoid), gate(2, torch.sigmoid), gate(3, torch.tanh)
    c_ref = f * c + i * g
    h_ref = o * torch.tanh(c_ref)
    h_out, c_out = cell(x, (h, c))
    assert torch.allclose(h_out, h_ref, atol=1e-6) and torch.allclose(c_out, c_ref, atol=1e-6)


def test_cifg_matches_equations_and_couples_gates():
    cell = build_cell("cifg", E, H)
    x, h, c = torch.randn(B, E), torch.randn(B, H), torch.randn(B, H)
    Wx, b, Wh = cell.x_linear.weight, cell.x_linear.bias, cell.h_linear.weight

    def pre(k):
        return x @ _block(Wx, k).T + h @ _block(Wh, k).T + b[k * H:(k + 1) * H]

    f, o, g = torch.sigmoid(pre(0)), torch.sigmoid(pre(1)), torch.tanh(pre(2))
    i = 1 - f
    c_ref = f * c + i * g
    h_ref = o * torch.tanh(c_ref)
    h_out, c_out = cell(x, (h, c))
    assert torch.allclose(h_out, h_ref, atol=1e-6) and torch.allclose(c_out, c_ref, atol=1e-6)


def test_wmc_matches_equations_and_uses_new_cell_state_for_output_gate():
    cell = build_cell("wmc", E, H)
    x, h, c = torch.randn(B, E), torch.randn(B, H), torch.randn(B, H)
    Wx, b, Wh = cell.x_linear.weight, cell.x_linear.bias, cell.h_linear.weight
    Wc, Wco = cell.c_prev_linear.weight, cell.c_curr_linear.weight

    def pre(k):
        return x @ _block(Wx, k).T + h @ _block(Wh, k).T + b[k * H:(k + 1) * H]

    i = torch.sigmoid(pre(0) + torch.tanh(c @ _block(Wc, 0).T))
    f = torch.sigmoid(pre(1) + torch.tanh(c @ _block(Wc, 1).T))
    g = torch.tanh(pre(3))
    c_ref = f * c + i * g
    o = torch.sigmoid(pre(2) + torch.tanh(c_ref @ Wco.T))   # c_t, not c_{t-1}
    h_ref = o * torch.tanh(c_ref)
    h_out, c_out = cell(x, (h, c))
    assert torch.allclose(h_out, h_ref, atol=1e-6) and torch.allclose(c_out, c_ref, atol=1e-6)
    # using c_{t-1} in the output gate would give a different result
    o_wrong = torch.sigmoid(pre(2) + torch.tanh(c @ Wco.T))
    assert not torch.allclose(o_wrong * torch.tanh(c_ref), h_out, atol=1e-6)


def test_parameter_count_formulas():
    n_lstm = sum(p.numel() for p in build_cell("lstm", E, H).parameters())
    n_cifg = sum(p.numel() for p in build_cell("cifg", E, H).parameters())
    n_wmc = sum(p.numel() for p in build_cell("wmc", E, H).parameters())
    assert n_lstm == 4 * H * (E + H) + 4 * H
    assert n_cifg == 3 * H * (E + H) + 3 * H
    assert n_wmc == n_lstm + 3 * H * H


@pytest.mark.parametrize("scheme", ["default", "xavier", "orthogonal"])
@pytest.mark.parametrize("name", CELL_NAMES)
def test_initialization_schemes(name, scheme):
    cell = build_cell(name, E, H)
    cell.reset_parameters(scheme, forget_bias=1.0)
    start = cell.forget_index * H
    assert torch.allclose(cell.x_linear.bias[start:start + H], torch.ones(H))
    if scheme == "orthogonal":
        block = _block(cell.h_linear.weight, 0)
        assert torch.allclose(block @ block.T, torch.eye(H), atol=1e-4)


def test_reverse_padded_is_an_involution():
    x = torch.randn(3, 7, 2)
    lengths = torch.tensor([7, 3, 1])
    once = reverse_padded(x, lengths)
    assert torch.equal(once[1, :3], x[1, :3].flip(0))
    assert torch.equal(once[1, 3:], x[1, 3:])          # padding untouched
    assert torch.equal(reverse_padded(once, lengths), x)


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("pooling", ["last", "mean", "max"])
@pytest.mark.parametrize("name", CELL_NAMES)
def test_padding_does_not_change_predictions(name, pooling, bidirectional):
    torch.manual_seed(0)
    model = RNNClassifier(50, 3, cell=name, embed_dim=E, hidden_size=H, num_layers=2,
                          bidirectional=bidirectional, pooling=pooling, dropout=0.0).eval()
    ids = torch.randint(2, 50, (B, 9))
    lengths = torch.tensor([9, 5, 2, 7])
    for row, length in enumerate(lengths.tolist()):
        ids[row, length:] = 0
    short = model(ids[:, :9], lengths)
    padded = model(torch.cat([ids, torch.zeros(B, 6, dtype=ids.dtype)], dim=1), lengths)
    assert short.shape == (B, 3)
    assert torch.allclose(short, padded, atol=1e-5)
    # each sequence alone (batch of one, no padding at all) gives the same logits as in the batch
    alone = model(ids[1:2, :5], lengths[1:2])
    assert torch.allclose(alone, short[1:2], atol=1e-5)


def test_gradients_flow_to_every_cell_parameter():
    for name in CELL_NAMES:
        model = RNNClassifier(30, 3, cell=name, embed_dim=E, hidden_size=H, dropout=0.0)
        ids = torch.randint(2, 30, (B, 6))
        loss = nn.functional.cross_entropy(model(ids, torch.full((B,), 6)), torch.zeros(B, dtype=torch.long))
        loss.backward()
        for pname, param in model.layers.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), (name, pname)


def test_metrics_from_confusion():
    cm = np.array([[8, 1, 1], [2, 5, 3], [0, 1, 9]])
    stats = T.metrics_from_confusion(cm)
    assert stats["accuracy"] == pytest.approx(22 / 30)
    assert 0 < stats["macro_f1"] < 1
    preds, labels = np.array([0, 1, 2, 2]), np.array([0, 1, 1, 2])
    assert T.confusion_matrix_np(preds, labels, 3).tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


@pytest.fixture(scope="module")
def device_data(synthetic_dir):
    prepared = D.prepare_data(synthetic_dir, D.DataConfig(max_len=40, min_freq=1), verbose=False)
    return T.DeviceData(prepared, torch.device("cpu"))


def test_train_subset_is_stratified_and_shared(device_data):
    full = device_data.train(1.0)
    sub = device_data.train(0.3)
    assert 0.25 * len(full) < len(sub) < 0.35 * len(full)
    assert device_data.train(0.3) is sub  # cached: identical samples for every model


def test_experiment_loop_resumes_and_skips(device_data, tmp_path):
    cpu = torch.device("cpu")
    base = T.ExperimentConfig(embed_dim=8, hidden_size=8, batch_size=64, epochs=2, patience=2)
    configs = T.make_ofat_configs(base, {"optimizer": ["adam", "sgd", "rmsprop"],
                                         "init": ["default", "xavier", "orthogonal"],
                                         "use_class_weights": [False, True]})
    assert len({T.exp_id(c) for c in configs}) == len(configs)       # no duplicates
    configs = [c for c in configs if c.optimizer == "adam" and c.init == "xavier"
               and not c.use_class_weights] + [replace(base, cell="wmc", use_class_weights=True),
                                              replace(base, cell="cifg", optimizer="sgd", lr=0.05)]
    metrics = T.run_experiments(configs, device_data, cpu, results_dir=tmp_path)
    assert len(metrics) == len({T.exp_id(c) for c in configs})
    assert set(metrics["cell"]) == set(CELL_NAMES)
    assert (metrics["status"] == "ok").all()
    assert metrics["n_params"].gt(0).all() and (metrics["n_params"] > metrics["n_params_cell"]).all()
    assert (tmp_path / "figures" / "runs").exists() and (tmp_path / "runs").exists()
    again = T.run_experiments(configs, device_data, cpu, results_dir=tmp_path)   # everything skipped
    assert len(again) == len(metrics)
    best = T.select_best_per_cell(again, seed=0)
    assert set(best) <= set(CELL_NAMES) and all(isinstance(c, T.ExperimentConfig) for c in best.values())


def test_resume_from_checkpoint(device_data, tmp_path):
    cpu = torch.device("cpu")
    cfg = T.ExperimentConfig(embed_dim=8, hidden_size=8, batch_size=64, epochs=3, patience=10)
    _, history, _ = T.train_experiment(cfg, device_data, cpu, tmp_path, verbose=False,
                                       keep_last_checkpoint=True)
    assert len(history) == 3
    last = tmp_path / f"{T.exp_id(cfg)}_last.pt"
    state = torch.load(last)
    state["epoch"], state["history"] = 1, state["history"][:1]      # pretend we crashed after epoch 1
    torch.save(state, last)
    row, resumed, _ = T.train_experiment(cfg, device_data, cpu, tmp_path, verbose=False)
    assert len(resumed) == 3 and resumed[0] == history[0]
    assert row["epochs_run"] == 3 and row["status"] == "ok"


def test_speed_benchmark_runs(device_data):
    info = T.benchmark_speed(T.ExperimentConfig(embed_dim=8, hidden_size=8, batch_size=64),
                             device_data, torch.device("cpu"), n_batches=3)
    assert info["seconds_per_epoch"] > 0
