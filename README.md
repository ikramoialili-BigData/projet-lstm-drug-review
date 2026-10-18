# Lab Project on Recurrent Neural Networks - Drug Review Classification

Advanced Natural Language Processing, International University of Rabat, academic year 2026/2027.

We classify patient drug reviews (UCI "Drug Reviews (Drugs.com)") into **3 sentiment classes** derived from the 1-10 rating, and compare three recurrent cells implemented **by hand with `nn.Linear`** (the built-in `nn.LSTM` is never used):

| Model | Description |
|-------|-------------|
| **Vanilla LSTM** | Baseline (appendix equations 1-6) |
| **CIFG** | Coupled Input-Forget Gate LSTM: `i_t = 1 - f_t` (equations 13-18) |
| **WMC** | Working Memory Connections LSTM: the cell state feeds the gates through `tanh(W_c c)`; the output gate uses the *new* cell state `c_t` (equations 7-12) |

**Team:** _add the names of the 5-6 team members here_

## Repository structure

```
projet-lstm-drug-review/
├── README.md
├── requirements.txt
├── src/
│   ├── cells.py    # vanilla LSTM, CIFG, WMC (nn.Linear), equations as comments
│   ├── data.py     # cleaning, vocabulary, fixed split, batching, sub-sampling
│   ├── model.py    # embedding + (stacked) cell + output layer, weight init
│   └── train.py    # training, evaluation, parameter counts, plots, smoke test
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis
│   └── 02_experiments.ipynb  # hyper-parameter study + final comparison
└── results/
    ├── metrics.csv   # one row per run (created by the experiments)
    └── figures/      # curves, confusion matrices, comparison plots
```

## Dataset (not included in the repo)

The dataset files are **not** committed because of license restrictions. Download them yourself:

* Kaggle: dataset **"UCI ML Drug Review dataset"** (author `jessicali9530`), files `drugsComTrain_raw.csv` and `drugsComTest_raw.csv`.
* Original source: <https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com>.

**On Kaggle:** in your notebook, click **Add Input** (right panel) -> search "UCI ML Drug Review dataset" -> **Add**. The files appear under `/kaggle/input/...` (usually `/kaggle/input/kuc-hackathon-winter-2018/`). The code finds them automatically, wherever they are mounted.

**Locally:** put the two CSV files in a `data/` folder at the repository root, or set the `DRUG_DATA_DIR` environment variable to the folder that contains them.

## Quick start

```bash
pip install -r requirements.txt

# 1) self check on synthetic data (about a minute, no dataset needed)
python -m src.train --smoke-test

# 2) train one model on the real data
python -m src.train --cell wmc --epochs 5 --train-fraction 0.2
```

For the full study, run the notebooks in order: `notebooks/01_eda.ipynb`, then `notebooks/02_experiments.ipynb` (Kaggle GPU recommended). The first cell of each notebook finds the code automatically (local folder, a Kaggle dataset containing the repo, or `git clone` of the GitHub URL set at the top of the cell).

## Method

**Data** (`src/data.py`)
* HTML entities decoded with `html.unescape` (e.g. `&#039;`), URLs removed, lower-casing, punctuation removed (apostrophes kept).
* **Classes** (configurable via `DataConfig.label_edges`): negative = rating 1-4, neutral = 5-6, positive = 7-10.
* **One fixed split** shared by all models: test = official `drugsComTest_raw.csv`; validation = stratified 10% of the official train file (seed 42); train = the rest. A fingerprint of the split is stored in every row of `metrics.csv` to prove that all runs use the same split.
* Vocabulary built **on the training split only** (`<PAD>` = 0, `<UNK>` = 1, min frequency 2, max 30,000 words); reviews are padded/truncated to 200 tokens.
* Optional **stratified sub-sampling** of the training split (`train_fraction`), which keeps the class proportions. The subset is the same for every model; validation and test sets are never sub-sampled.

**Models** (`src/cells.py`, `src/model.py`)
* Each gate projection is an `nn.Linear`. For speed, the gate matrices are stored side by side in two fused `nn.Linear` layers (inputs with biases, hidden state without bias) and split with `chunk`; this is mathematically identical to one `nn.Linear` per gate. WMC adds three extra `nn.Linear(H, H, bias=False)` for `W_ci`, `W_cf`, `W_co` (full matrices).
* Padding is masked: on padded steps the hidden and cell states are frozen, so the result does not depend on the amount of padding (checked in the smoke test).
* Architecture: `Embedding -> dropout -> [stacked / optionally bidirectional] cell -> pooling (last state | mean | max) -> dropout -> Linear`. Default: 1 layer, unidirectional, last hidden state.
* Trainable parameters of the recurrent part (input size `D`, hidden size `H`): vanilla `4H(D+H) + 4H`; CIFG `3H(D+H) + 3H`; WMC `4H(D+H) + 4H + 3H^2`. The embedding and the output layer are identical for the three models.

**Evaluation** (`src/train.py`)
* Accuracy, **macro-F1** (the classes are imbalanced), per-class F1, confusion matrix, number of **trainable parameters** (total and recurrent part), wall-clock training time.
* Model selection (early stopping, choice of hyper-parameters) uses the **validation** set only; the test set is evaluated once per run with the weights of the best validation epoch.
* Optional **class weights** (inverse frequency) in the cross-entropy loss, compared as one of the hyper-parameters.

**Hyper-parameter study** (`notebooks/02_experiments.ipynb`)
* Stage 1: one factor at a time around a default configuration, on a stratified 20% of the training split - learning rate, optimizer (Adam / RMSprop / SGD with momentum), initialisation (Xavier / orthogonal / PyTorch default), batch size, hidden size, class weights - for the three cells with identical grids. The default configuration is run with two seeds to estimate seed noise.
* Stage 2: the best configuration of each cell is retrained on the full training split with several seeds; these runs give the final numbers.
* Everything is appended to `results/metrics.csv` after each run, so a stopped Kaggle session loses nothing, and re-running the notebook skips finished runs.

## Results

_To be filled in after the experiments have been run on Kaggle._ The numbers come from `results/metrics.csv` and `results/summary_final.csv`; figures are in `results/figures/`.

| Model | Test accuracy | Test macro-F1 | Trainable params (total / recurrent) | Train time (s) |
|-------|---------------|---------------|--------------------------------------|----------------|
| Vanilla LSTM | | | | |
| CIFG | | | | |
| WMC | | | | |

## Reproducibility

* Seeds fixed for `random`, `numpy` and `torch`; the data split uses its own fixed seed (42) so it is identical across models and training seeds.
* GPU/CPU is detected automatically. Some CUDA operations are non-deterministic, so GPU runs with the same seed can differ slightly; this is why the final stage uses several seeds.
* `requirements.txt` lists the dependencies.

## License / data note

Code: for course use. The Drug Review dataset belongs to its authors (Gräßer et al., 2018, UCI ML Repository) and must be downloaded from its original source.
