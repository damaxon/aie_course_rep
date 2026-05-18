from pathlib import Path
import shutil

from ..paths import DATA_RAW_DIR,CONFIGS_DIR
    
def check_kaggle() -> bool:
    try:
        import kagglehub
        return True
    except ImportError:
        return False

def check_kaggle_config() -> bool:
    return (CONFIGS_DIR / "kaggle.json").exists()

def download_with_kagglehub(dataset_handle: str, target_dir: Path) -> bool:
    try:
        import kagglehub
        target_dir.mkdir(parents=True, exist_ok=True)
        
        path = kagglehub.dataset_download(dataset_handle)
        
        for item in Path(path).iterdir():
            if item.is_file():
                shutil.copy2(item, target_dir)
            else:
                shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
        
        return True
    except Exception:
        return False

def download_all() -> dict:

    datasets = [
        {
            "name": "Udacity Self Driving Car Dataset",
            "handle": "sshikamaru/udacity-self-driving-car-dataset",
        }
    ]

    results = {}

    for ds in datasets:
        target_dir = DATA_RAW_DIR / ds["name"]
        success = download_with_kagglehub(ds["handle"], target_dir)
        results[ds["name"]] = success
    
    return results