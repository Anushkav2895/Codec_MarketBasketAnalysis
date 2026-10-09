import shutil
import sys
from pathlib import Path

DATASET = "heeraldedhia/groceries-dataset"
FILE_NAME = "Groceries_dataset.csv"
DATA_DIR = Path(__file__).resolve().parent
TARGET = DATA_DIR / FILE_NAME

if TARGET.exists():
    print(f"{FILE_NAME} already exists in {DATA_DIR}. Nothing to do.")
    sys.exit(0)

try:
    import kagglehub
except ImportError:
    sys.exit("kagglehub is not installed. Run: pip install -r requirements.txt")

try:
    source = Path(kagglehub.dataset_download(DATASET)) / FILE_NAME
except Exception as exc:
    sys.exit(
        f"Download failed: {exc}\n"
        f"Manual option: download the dataset from https://www.kaggle.com/datasets/{DATASET} "
        f"and place {FILE_NAME} in {DATA_DIR}"
    )

if not source.exists():
    sys.exit(f"{FILE_NAME} was not found in the downloaded files.")

shutil.copy(source, TARGET)
print(f"Copied {FILE_NAME} to {TARGET}")