import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Define hyperparameter sets to test
hyperparams = [
    {"lr0": 0.001, "batch": 16},  
    {"lr0": 0.001, "batch": 32},  
    {"lr0": 0.0005, "batch": 16}, 
    {"lr0": 0.0005, "batch": 32}, 
    {"lr0": 0.002, "batch": 16},  
    {"lr0": 0.002, "batch": 32},  
    {"lr0": 0.001, "batch": 16, "momentum": 0.9},  
    {"lr0": 0.001, "batch": 32, "momentum": 0.8},  
    {"lr0": 0.001, "batch": 16, "weight_decay": 0.0001},  
    {"lr0": 0.001, "batch": 32, "weight_decay": 0.0005},  
]


# Initialize model and results storage
model_name = "yolov8n.pt"
data_yaml = "data.yaml"
results = []

# Loop through each hyperparameter configuration
for params in hyperparams:
    # Initialize and train model
    model = YOLO(model_name)
    print(f"Training with params: {params}")
    
    metrics = model.train(
        data=data_yaml,
        epochs=10,  # Use a smaller number for faster testing
        imgsz=640,
        batch=params["batch"],
        lr0=params["lr0"],
        device=0
    )
    
    # Validate and get metrics
    val_metrics = model.val()
    precision = val_metrics.box.p[0]
    recall = val_metrics.box.r[0]
    map_50 = val_metrics.box.map50
    map_5095 = val_metrics.box.map

    # Store results for comparison
    results.append({
        "params": params,
        "precision": precision,
        "recall": recall,
        "mAP_50": map_50,
        "mAP_50_95": map_5095,
    })

# Plot and save comparison graphs
os.makedirs("results", exist_ok=True)

# Precision vs Hyperparameters
plt.figure(figsize=(10, 6))
for result in results:
    plt.bar(
        str(result["params"]), 
        result["precision"], 
        label=f"LR: {result['params']['lr0']}, Batch: {result['params']['batch']}"
    )
plt.title("Precision (IoU=0.5) vs Hyperparameters")
plt.ylabel("Precision")
plt.xlabel("Hyperparameter Configurations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/precision_comparison.png")
plt.show()

# Recall vs Hyperparameters
plt.figure(figsize=(10, 6))
for result in results:
    plt.bar(
        str(result["params"]),
        result["recall"],
        label=f"LR: {result['params']['lr0']}, Batch: {result['params']['batch']}"
    )
plt.title("Recall (IoU=0.5) vs Hyperparameters")
plt.ylabel("Recall")
plt.xlabel("Hyperparameter Configurations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/recall_comparison.png")
plt.show()

# mAP (IoU=0.5) vs Hyperparameters
plt.figure(figsize=(10, 6))
for result in results:
    plt.bar(
        str(result["params"]),
        result["mAP_50"],
        label=f"LR: {result['params']['lr0']}, Batch: {result['params']['batch']}"
    )
plt.title("mAP (IoU=0.5) vs Hyperparameters")
plt.ylabel("mAP (IoU=0.5)")
plt.xlabel("Hyperparameter Configurations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/map_50_comparison.png")
plt.show()

# mAP (IoU=0.5:0.95) vs Hyperparameters
plt.figure(figsize=(10, 6))
for result in results:
    plt.bar(
        str(result["params"]),
        result["mAP_50_95"],
        label=f"LR: {result['params']['lr0']}, Batch: {result['params']['batch']}"
    )
plt.title("mAP (IoU=0.5:0.95) vs Hyperparameters")
plt.ylabel("mAP (IoU=0.5:0.95)")
plt.xlabel("Hyperparameter Configurations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/map_5095_comparison.png")
plt.show()
