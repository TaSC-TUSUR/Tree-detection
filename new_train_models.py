import os
import shutil
import random
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path("/home/st3pegor/Projects/gpo_zalupa")
RUN_DIR = BASE_DIR / "templates/runable_sucking"
TEST_DIR = BASE_DIR / "templates/Tree Detection.v11i.yolov8/test"

REAL_DIR = BASE_DIR / "templates/Tree Detection.v11i.yolov8/train"
ART_DIR = BASE_DIR / "templates/Super Mega Gay trees.v2i.yolov8/train"
ADJ_DIR = BASE_DIR / "templates/adjusted/train"

# Count real data
REAL_IMAGES = sorted((REAL_DIR / "images").glob("*"))
REAL_COUNT = len(REAL_IMAGES)

# Model configurations
MODELS = {
    "model_a_real_only": {
        "train": [(REAL_DIR, REAL_COUNT)],
        "val": [REAL_DIR],
    },
    "model_b_real_generated": {
        "train": [(REAL_DIR, REAL_COUNT), (ART_DIR, None)],
        "val": [REAL_DIR],
    },
    "model_c_real_adjusted": {
        "train": [(REAL_DIR, REAL_COUNT), (ADJ_DIR, None)],
        "val": [REAL_DIR],
    },
    "model_d_real_generated_val_real_generated": {
        "train": [(REAL_DIR, REAL_COUNT), (ART_DIR, None)],
        "val": [REAL_DIR, ART_DIR],
    },
    "model_e_real_adjusted_val_real_adjusted": {
        "train": [(REAL_DIR, REAL_COUNT), (ADJ_DIR, None)],
        "val": [REAL_DIR, ADJ_DIR],
    },
}

# Training params
EPOCHS = 12
IMGSZ = 640
BATCH = 16
MODEL_NAME = "yolov8s.pt"

# Utilities
def clear_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def copy_data(src_dir, dst_dir, count=None):
    images = sorted((src_dir / "images").glob("*"))
    labels = sorted((src_dir / "labels").glob("*"))
    paired = list(zip(images, labels))
    random.shuffle(paired)
    if count is not None:
        paired = paired[:count]
    for img_path, lbl_path in paired:
        shutil.copy(img_path, dst_dir / "images" / img_path.name)
        shutil.copy(lbl_path, dst_dir / "labels" / lbl_path.name)

def prepare_data(train_config, val_dirs):
    # Clear train
    train_images = RUN_DIR / "train/images"
    train_labels = RUN_DIR / "train/labels"
    clear_directory(train_images)
    clear_directory(train_labels)

    # Add each dataset to train
    for src_dir, count in train_config:
        copy_data(src_dir, RUN_DIR / "train", count)

    # Clear val
    val_images = RUN_DIR / "valid/images"
    val_labels = RUN_DIR / "valid/labels"
    clear_directory(val_images)
    clear_directory(val_labels)

    # Add each dataset to validation
    for src_dir in val_dirs:
        copy_data(src_dir, RUN_DIR / "valid")  # copy all

    # Clear and copy fixed test set
    shutil.rmtree(RUN_DIR / "test", ignore_errors=True)
    shutil.copytree(TEST_DIR, RUN_DIR / "test")

def run_training(model_name):
    print(f"\n🚀 Training model: {model_name}")
    result = subprocess.run([
        "python3", "train.py",
        "--model", MODEL_NAME,
        "--data", "data.yaml",
        "--epochs", str(EPOCHS),
        "--imgsz", str(IMGSZ),
        "--batch", str(BATCH)
    ], check=True)

    # Save weights
    runs_path = BASE_DIR / "runs/detect"
    latest_run = max(runs_path.glob("train*"), key=os.path.getmtime)
    best_pt = latest_run / "weights/best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, BASE_DIR / f"{model_name}.pt")
    else:
        raise FileNotFoundError(f"best.pt not found in {latest_run}")

def run_metrics(model_name):
    print(f"📊 Running metrics for: {model_name}")
    subprocess.run([
        "python3", "metrics.py",
        "--model_name", model_name,
        "--save_dir", "detections_test"
    ], check=True)

def main():
    print(f"📦 Using {REAL_COUNT} real images as constant training base.\n")
    for model_name, config in MODELS.items():
        print(f"\n==============================")
        print(f"🧪 Preparing dataset for: {model_name}")
        prepare_data(config["train"], config["val"])
        run_training(model_name)
        run_metrics(model_name)
    print("\n✅ All models trained and evaluated.")

if __name__ == "__main__":
    main()
