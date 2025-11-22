import os
import subprocess
import sys
from pathlib import Path
import shutil

SCRIPT_DIR = Path(__file__).parent
SCR_DIR = SCRIPT_DIR.parent
PROJECT_DIR = SCR_DIR.parent

configs_dir = PROJECT_DIR / "configs"
data_raw_dir = PROJECT_DIR / "data" / "raw"

def run_command(cmd,description) -> bool:
    try:
        result = subprocess.run(cmd,shell=True,check=True,capture_output=True,text=True)
        print(f"{description} - успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} - ошибка")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        return False
    
def check_kaggle() -> bool:
    try:
        import kagglehub
        return True
    except ImportError:
        print("Kagglehub не установлен")
        print("Установите: pip install kagglehub")
        return False

def check_kaggle_config() -> bool:
    kaggle_config = configs_dir / "kaggle.json"
    if not kaggle_config.exists():
        print("Конфиг Kaggle не найден")
        print("Получите API ключ на сайте https://www.kaggle.com")
        print("Сохраните в ~/project/configs/kaggle.json")
        return False
    return True

def download_with_kagglehub(dataset_handle, target_dir) -> bool:
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
    except Exception as e:
        print(f"Ошибка при загрузке: {str(e)}")
        return False

def main():

    if not check_kaggle() or not check_kaggle_config():
        sys.exit(1)
    
    data_dirs = [
        data_raw_dir / "vehicle_detection",
        data_raw_dir / "stanford_cars"
    ]

    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True,exist_ok=True)

    datasets = [
        {
            "name": "Udacity Self Driving Car Dataset",
            "handle": "sshikamaru/udacity-self-driving-car-dataset",
            "extract_dir": data_raw_dir / "vehicle_detection"
        },
        {
            "name": "Stanford Cars Classification", 
            "handle": "eduardo4jesus/stanford-cars-dataset",
            "extract_dir": data_raw_dir / "stanford_cars"
        }
    ]

    success_count = 0

    for dataset in datasets:
        if download_with_kagglehub(dataset["handle"], dataset["extract_dir"]):
            print(f"Загрузка {dataset['name']} - успешно")
            success_count += 1
        else:
            print(f"Загрузка {dataset['name']} - ошибка")
    
    print(f"Успешно загружено {success_count}/{len(datasets)} датасетов")

    if success_count == len(datasets):
        print("Все датасеты успешно загружены")
    else:
        print("Некоторые датасеты не загружены")
        sys.exit(1)
    
main()