# Lab Project on Recurrent Neural Networks - Drug Review Classification

Advanced Natural Language Processing - International University of Rabat - 2026/2027

Text classification of patient drug reviews (UCI "Drug Reviews (Drugs.com)") into 3 sentiment classes derived from the
rating, with **three recurrent cells implemented by hand with `nn.Linear`**:

| Model | Idea |
|---|---|
| **Vanilla LSTM** (baseline) | input, forget, output gates and a candidate `g_t` |
| **CIFG** | coupled input-forget gate: `i_t = 1 - f_t` |
| **WMC** | working memory connections: gates also read `tanh(W_c c)`; the output gate reads the *new* `c_t` |

The built-in recurrent layers of PyTorch are **not** used. The equations are written as comments above the matching lines
of `src/cells.py`.

> **No result is stored in this README on purpose.** All numbers come from `results/metrics.csv`, produced by
> `notebooks/02_experiments.ipynb`.

## Repository structure

```
projet-lstm-drug-review/
├── README.md
├── requirements.txt
├── .gitignore                # the dataset (data/), cache/ and checkpoints are NOT pushed
├── src/
│   ├── __init__.py
│   ├── config.py             # DATA_DIR, label mapping, text/split settings
│   ├── cells.py              # vanilla LSTM, CIFG, WMC with nn.Linear
│   ├── data.py               # cleaning, vocabulary, fixed splits (seed), cache
│   ├── model.py              # embedding + stacked/bidirectional cells + pooling + output layer
│   └── train.py              # training, evaluation, parameter count, resumable experiments, plots
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis + data preparation
│   └── 02_experiments.ipynb  # experiments, comparison, figures
├── tests/
│   └── test_smoke.py         # shape / equation / resume tests on synthetic data
└── results/
    ├── metrics.csv           # one row per experiment (created by the notebook)
    ├── final_comparison.csv  # best setting of each cell (created by the notebook)
    ├── runs/                 # per-run JSON: config, learning curve, test confusion matrix
    └── figures/              # EDA figures, comparison, hyperparameter and confusion-matrix figures
```

(`tests/`, `src/config.py` and `src/__init__.py` are small additions to the imposed structure.)

## Google Drive layout

Put the whole project folder in your Drive and the dataset inside `data/`:

```
MyDrive/projet-lstm-drug-review/
├── data/
│   ├── drugsComTrain_raw.tsv      (or .csv)
│   └── drugsComTest_raw.tsv       (or .csv)
├── src/  notebooks/  results/  requirements.txt  README.md  ...
```

* The extension (`.tsv`, `.csv`, `.txt` or none) and the separator (tab or comma) are **detected automatically**.
* If your dataset is elsewhere, edit the single variable `DATA_DIR` at the top of each notebook (and, optionally, in
  `src/config.py`). If the files are not found, the code stops with a message that lists what it found in the folder.
* Expected columns: `uniqueID, drugName, condition, review, rating, date, usefulCount`.

## Running on Google Colab

1. Open `notebooks/01_eda.ipynb` from Drive (right click > Open with > Google Colaboratory).
2. `Runtime > Change runtime type > T4 GPU`.
3. Run the cells from top to bottom. The first cells mount Drive, add the project to `sys.path`, install any package of
   `requirements.txt` that is missing, print the GPU, set the seeds and enable `%autoreload`.
4. Open and run `notebooks/02_experiments.ipynb` the same way. It reuses the data cache written by notebook 01.

Editing code: change files in `src/` on Drive; thanks to `%autoreload 2` the next cell run uses the new code
(if not, restart the runtime or call `importlib.reload(src.train)`).

### Experiments and time budget

Edit the "EXPERIMENT BUDGET" cell of notebook 02 (`TRAIN_FRACTION`, `EPOCHS`, `PATIENCE`, `SEEDS`, `FACTORS`).
A speed benchmark cell prints the time per epoch of each model before you start.

* **Stage A**: base comparison of the three cells, several seeds.
* **Stage B**: one factor at a time around the base setting - learning rate, optimizer (Adam / SGD / RMSprop),
  initialization (default / Xavier / orthogonal), batch size, hidden size, class weights.
* **Stage C**: the best setting of each cell (selected on the **validation** macro-F1) re-run with more seeds.

`TRAIN_FRACTION < 1` uses a **stratified** sub-sample of the training split (class proportions are preserved, and every
model sees exactly the same samples).

### Resuming after a disconnection

Colab sessions can be interrupted. Everything important is on Drive:

* after **each epoch** a checkpoint is saved (`results/checkpoints/`);
* after **each finished experiment** a row is appended to `results/metrics.csv`, plus a JSON and two figures.

To resume: reconnect, re-run the setup cells, then re-run the stage cell you were in. Finished experiments are skipped;
the interrupted one restarts from its last epoch. To re-run an experiment on purpose, delete its row in `metrics.csv`.

## Experimental protocol (identical for the three models)

* Target: rating 1-4 -> *negative*, 5-6 -> *neutral*, 7-10 -> *positive* (configurable, `LABEL_BINS` in `src/config.py`).
* Test set = the official test file. Validation = 10 % of the official train file, **stratified**, drawn once with a fixed
  seed. The vocabulary is built on the remaining training reviews only (`<PAD>` = 0, `<UNK>` = 1).
* Reviews are cleaned (HTML entities decoded with `html.unescape`, lowercase, tokenization), truncated / padded to
  `MAX_LEN` tokens; padding is masked inside the recurrent layers (states are frozen on padding).
* Metrics: accuracy, **macro-F1** (important because of class imbalance), per-class F1, confusion matrix,
  number of **trainable parameters** (total and recurrent cell only), training time.
* Model selection uses validation macro-F1 only; the test set is used to report the final numbers.
* Seeds: Python, NumPy and PyTorch. Gradient clipping is applied. Mixed precision is not used (full fp32).

## Tests

```bash
python -m pytest tests -v
```

The tests use synthetic data (no real dataset needed): gate equations compared with a manual computation, parameter
counts, padding invariance, initialization, data pipeline, resumable experiment loop.

## Publishing on GitHub (without the dataset)

The `.gitignore` excludes `data/`, `*.tsv`, `cache/` and `results/checkpoints/`.

**Option 1 - from your computer:** download the project folder from Drive (without `data/`), then

```bash
cd projet-lstm-drug-review
git init && git add . && git status        # check that no dataset file appears
git commit -m "Lab project: LSTM, CIFG and WMC on Drug Review"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

**Option 2 - from Colab:** copy the project to a local Colab folder (git is faster there than on Drive), then push with a
GitHub token. Never write the token in a notebook that you share.

```python
from getpass import getpass
token = getpass("GitHub token: ")
!rsync -a --exclude data --exclude cache --exclude checkpoints --exclude __pycache__ /content/drive/MyDrive/projet-lstm-drug-review/ /content/repo/
%cd /content/repo
!git init -q && git add . && git status --short | head -30
!git -c user.name="<name>" -c user.email="<email>" commit -qm "Lab project: LSTM, CIFG and WMC on Drug Review"
!git branch -M main
!git push https://{token}@github.com/<user>/<repo>.git main
```

Create the repository on GitHub first (**public**, empty), then submit its link in the submission form before 25/10/2026.

## Data source

Drug Reviews (Drugs.com), UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com
