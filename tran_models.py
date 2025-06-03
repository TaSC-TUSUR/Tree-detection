import os
import shutil
import random
import subprocess
from pathlib import Path

BASE_DIR = Path("/home/st3pegor/Projects/gpo_zalupa")
RUN_DIR = BASE_DIR / "templates/runable_sucking"
VAL_DIR = BASE_DIR / "templates/Tree Detection.v11i.yolov8/valid"
TEST_DIR = BASE_DIR / "templates/Tree Detection.v11i.yolov8/test"

REAL_DIR = BASE_DIR / "templates/Tree Detection.v11i.yolov8/train"
ART_DIR = BASE_DIR / "templates/Super Mega Gay trees.v2i.yolov8/train"
ADJ_DIR = BASE_DIR / "templates/adjusted/train"

MIXES = [
    ("real_art_80_20", REAL_DIR, ART_DIR, 0.8),
    ("real_art_50_50", REAL_DIR, ART_DIR, 0.5),
    ("real_art_30_70", REAL_DIR, ART_DIR, 0.3),
    ("real_art_0_100", REAL_DIR, ART_DIR, 0.0),
    ("real_adj_80_20", REAL_DIR, ADJ_DIR, 0.8),
    ("real_adj_50_50", REAL_DIR, ADJ_DIR, 0.5),
    ("real_adj_30_70", REAL_DIR, ADJ_DIR, 0.3),
    ("real_adj_0_100", REAL_DIR, ADJ_DIR, 0.0),
]

EPOCHS = 12
IMGSZ = 640
BATCH = 16
MODEL_NAME = "yolov8s.pt"

def clear_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

def copy_data(src_dir, dst_dir, count):
    images = list((src_dir / "images").glob("*"))
    labels = list((src_dir / "labels").glob("*"))

    paired = list(zip(images, labels))
    random.shuffle(paired)
    selected = paired[:count]

    for img_path, lbl_path in selected:
        shutil.copy(img_path, dst_dir / "images" / img_path.name)
        shutil.copy(lbl_path, dst_dir / "labels" / lbl_path.name)

def prepare_data(real_dir, mix_dir, ratio):
    train_dir = RUN_DIR / "train"
    clear_directory(train_dir / "images")
    clear_directory(train_dir / "labels")

    real_images = list((real_dir / "images").glob("*"))
    total = len(real_images)
    real_count = int(ratio * total)
    mix_count = total - real_count

    copy_data(real_dir, train_dir, real_count)
    copy_data(mix_dir, train_dir, mix_count)

    # Always copy real validation and test data
    shutil.rmtree(RUN_DIR / "valid", ignore_errors=True)
    shutil.rmtree(RUN_DIR / "test", ignore_errors=True)
    shutil.copytree(VAL_DIR, RUN_DIR / "valid")
    shutil.copytree(TEST_DIR, RUN_DIR / "test")

def run_training(mix_name):
    print(f"\n🚀 Training model: {mix_name}")
    result = subprocess.run([
        "python3", "train.py",
        "--model", MODEL_NAME,
        "--data", "data.yaml",
        "--epochs", str(EPOCHS),
        "--imgsz", str(IMGSZ),
        "--batch", str(BATCH)
    ], check=True)

    # Find the latest training run folder
    runs_path = BASE_DIR / "runs/detect"
    latest_run = max(runs_path.glob("train*"), key=os.path.getmtime)

    # Copy the best weights to the root dir with the mix_name
    best_pt = latest_run / "weights/best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, BASE_DIR / f"{mix_name}.pt")
    else:
        raise FileNotFoundError(f"best.pt not found in {latest_run}")

def run_metrics(mix_name):
    print(f"📊 Running metrics for: {mix_name}")
    subprocess.run([
        "python3", "metrics.py",
        "--model_name", mix_name,
        "--save_dir", "detections_test"
    ], check=True)

def main():
    for mix_name, real_dir, mix_dir, ratio in MIXES:
        print(f"\n==============================")
        print(f"🧪 Preparing dataset: {mix_name} (ratio {ratio*100:.0f}% real)")
        prepare_data(real_dir, mix_dir, ratio)
        run_training(mix_name)
        run_metrics(mix_name)
    print("\n✅ All models trained and evaluated.")

if __name__ == "__main__":
    main()
