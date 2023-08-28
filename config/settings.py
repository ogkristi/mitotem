from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
TRAIN_ROOT = PROJECT_ROOT / "data" / "processed"
DATASET_SIZE = 256

MEAN = 0.6796
STD = 0.1451
