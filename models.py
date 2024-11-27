import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Define the YOLOv8 models to compare
models = [
    "yolov8n.pt",  # Nano model
    "yolov8s.pt",  # Small model
    "yolov8m.pt",  # Medium model
    "yolov8l.pt",  # Large model
    "yolov8x.pt"   # Extra-large model
]

# Initialize results storage
results = []

# Train and validate each model
for model_name in models:
    print(f"Training and validating model: {model_name}")
    model = YOLO(model_name)

    # Train the model (adjust parameters as needed)
    model.train(data="data.yaml", epochs=20, imgsz=640, batch=16, device=0)

    # Validate the model
    metrics = model.val()

    # Extract relevant metrics
    precision = metrics.box.p[0]  # Precision at IoU=0.5
    recall = metrics.box.r[0]     # Recall at IoU=0.5
    map_50 = metrics.box.map50    # mAP at IoU=0.5
    map_5095 = metrics.box.map    # mAP at IoU=0.5:0.95

    # Store the metrics for comparison
    results.append({
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "mAP_50": map_50,
        "mAP_50_95": map_5095,
    })

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Plot and save comparison graphs
# Precision Comparison
plt.figure(figsize=(10, 6))
models = [result["model"] for result in results]
precisions = [result["precision"] for result in results]
plt.bar(models, precisions, color='skyblue')
plt.title("Precision (IoU=0.5) Comparison Across YOLOv8 Models")
plt.ylabel("Precision")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/precision_comparison_models.png")
plt.show()

# Recall Comparison
plt.figure(figsize=(10, 6))
recalls = [result["recall"] for result in results]
plt.bar(models, recalls, color='orange')
plt.title("Recall (IoU=0.5) Comparison Across YOLOv8 Models")
plt.ylabel("Recall")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/recall_comparison_models.png")
plt.show()

# mAP (IoU=0.5) Comparison
plt.figure(figsize=(10, 6))
map_50s = [result["mAP_50"] for result in results]
plt.bar(models, map_50s, color='green')
plt.title("mAP (IoU=0.5) Comparison Across YOLOv8 Models")
plt.ylabel("mAP (IoU=0.5)")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/map_50_comparison_models.png")
plt.show()

# mAP (IoU=0.5:0.95) Comparison
plt.figure(figsize=(10, 6))
map_5095s = [result["mAP_50_95"] for result in results]
plt.bar(models, map_5095s, color='purple')
plt.title("mAP (IoU=0.5:0.95) Comparison Across YOLOv8 Models")
plt.ylabel("mAP (IoU=0.5:0.95)")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/map_5095_comparison_models.png")
plt.show()
