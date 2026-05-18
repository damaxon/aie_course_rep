import shutil
from pathlib import Path

from ..paths import DATA_RAW_DIR

def _organize_vehicle(vehicle_detection_dir: Path):
    
    data_dir = vehicle_detection_dir / "data"
    
    if data_dir.exists():
        images_dir = vehicle_detection_dir / "images"
        labels_dir = vehicle_detection_dir / "labels"
        
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        for file in data_dir.rglob("*"):
            if file.is_file():
                if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(file, images_dir / file.name)
                elif file.suffix.lower() in ['.json', '.xml', '.txt', '.csv']:
                    shutil.copy2(file, labels_dir / file.name)
        
        shutil.rmtree(data_dir)
    
    pass

def organize_all() -> dict:
    results = {}
    
    vehicle_dir = DATA_RAW_DIR / "Udacity Self Driving Car Dataset"
     
    if vehicle_dir.exists():
        _organize_vehicle(vehicle_dir)
        results["vehicle_detection"] = True
    else:
        results["vehicle_detection"] = False
    
    return results