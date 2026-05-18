import shutil
from pathlib import Path

from ..paths import DATA_RAW_DIR

def delete_all() -> int:
    
    if not DATA_RAW_DIR.exists():
        return 0
    
    count = 0
    for item in DATA_RAW_DIR.iterdir():
        if item.is_file():
            item.unlink()
            count += 1
        elif item.is_dir():
            shutil.rmtree(item)
            count += 1
    return count