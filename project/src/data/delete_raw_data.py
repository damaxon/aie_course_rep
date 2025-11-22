import shutil
from pathlib import Path

def delete_raw_data():
    SCRIPT_DIR = Path(__file__).parent
    SCR_DIR = SCRIPT_DIR.parent
    PROJECT_DIR = SCR_DIR.parent
    data_raw_dir = PROJECT_DIR / "data" / "raw"
    
    print(f"Очистка папки: {data_raw_dir}")
    
    if not data_raw_dir.exists():
        print("Папка data/raw не существует")
        return False
    
    for item in data_raw_dir.iterdir():
        if item.is_file():
            item.unlink()
            print(f"Удален файл: {item.name}")
        elif item.is_dir():
            shutil.rmtree(item)
            print(f"Удалена папка: {item.name}")
    
    print("Папка data/raw полностью очищена")
    return True

delete_raw_data()