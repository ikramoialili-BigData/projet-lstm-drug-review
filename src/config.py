"""Central configuration shared by the notebooks and the modules in src/.

Edit DATA_DIR if your dataset lives somewhere else on Google Drive. The notebooks
define their own DATA_DIR variable at the top and pass it explicitly to the code,
so this value is only a default.
"""
from pathlib import Path

# The project root is the parent of the src/ folder, wherever the project is stored.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# >>> EDIT HERE if needed: folder that contains drugsComTrain_raw.* and drugsComTest_raw.* <<<
DATA_DIR = "/content/drive/MyDrive/projet-lstm-drug-review/data"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CACHE_DIR = PROJECT_ROOT / "cache"

# Rating -> class mapping: (lowest rating, highest rating, class name), bounds included.
# Change this tuple to try another mapping (for example binary: ((1, 5, "negative"), (6, 10, "positive"))).
LABEL_BINS = ((1, 4, "negative"), (5, 6, "neutral"), (7, 10, "positive"))

# Text / split settings
MAX_LEN = 200        # maximum number of tokens kept per review (longer reviews are truncated)
MIN_FREQ = 3         # a token must appear at least this many times in the training split
MAX_VOCAB = 30000    # maximum vocabulary size (including <PAD> and <UNK>)
VAL_SIZE = 0.1       # fraction of the official train file used for validation (stratified)
SPLIT_SEED = 42      # seed of the single, fixed train/validation split
