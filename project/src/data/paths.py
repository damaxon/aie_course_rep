from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_DIR = SRC_DIR.parent

DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
CONFIGS_DIR = PROJECT_DIR / "configs"
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"