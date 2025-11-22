import os
import shutil
from pathlib import Path

def organize_stanford_cars(stanford_cars_dir: Path):
    print("Организация Stanford Cars dataset...")
    
    car_devkit = stanford_cars_dir / "car_devkit"
    cars_test = stanford_cars_dir / "cars_test"
    cars_train = stanford_cars_dir / "cars_train"
    
    images_dir = stanford_cars_dir / "images"
    annotations_dir = stanford_cars_dir / "annotations"
    
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    
    if cars_train.exists():
        train_images = cars_train / "cars_train"
        if train_images.exists():
            for img_file in train_images.glob("*.jpg"):
                shutil.copy2(img_file, images_dir / f"train_{img_file.name}")
    
    if cars_test.exists():
        test_images = cars_test / "cars_test"
        if test_images.exists():
            for img_file in test_images.glob("*.jpg"):
                shutil.copy2(img_file, images_dir / f"test_{img_file.name}")
    
    if car_devkit.exists():
        devkit_files = [
            "cars_meta.mat",
            "cars_test_annos.mat", 
            "cars_train_annos.mat",
            "README.txt",
            "train_perfect_preds.txt",
            "eval_train.m"
        ]
        for file in devkit_files:
            devkit_file = car_devkit / "devkit" / file
            if devkit_file.exists():
                shutil.copy2(devkit_file, annotations_dir / file)
    
    for old_dir in [car_devkit, cars_test, cars_train]:
        if old_dir.exists():
            shutil.rmtree(old_dir)
    
    print(f"Stanford Cars организован: {images_dir}, {annotations_dir}")

def organize_vehicle_detection(vehicle_detection_dir: Path):
    print("Организация Vehicle Detection dataset...")
    
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
    
    print(f"Vehicle Detection организован: {images_dir}, {labels_dir}")

def main():
    SCRIPT_DIR = Path(__file__).parent
    SCR_DIR = SCRIPT_DIR.parent
    PROJECT_DIR = SCR_DIR.parent
    data_raw_dir = PROJECT_DIR / "data" / "raw"
    
    stanford_cars_dir = data_raw_dir / "stanford_cars"
    if stanford_cars_dir.exists():
        organize_stanford_cars(stanford_cars_dir)
    else:
        print("Stanford Cars директория не найдена")
     
    vehicle_detection_dir = data_raw_dir / "vehicle_detection"
    if vehicle_detection_dir.exists():
        organize_vehicle_detection(vehicle_detection_dir)
    else:
        print("Vehicle Detection директория не найдена")

main()