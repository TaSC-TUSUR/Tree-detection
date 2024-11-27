from ultralytics import YOLO
import os

# Define model name and evaluate metrics
model_name = 'best  '
model = YOLO(f"{model_name}.pt")

metrics = model.val()  #

# Extract specific metrics like precision, recall, and mAP
precision = metrics.box.p  # Precision at IoU=0.5
recall = metrics.box.r if hasattr(metrics, "box") else None  # Recall at IoU=0.5
map_50 = metrics.box.map50  # mAP at IoU=0.5
map_5095 = metrics.box.map  # mAP at IoU=0.5:0.95

# Print the metrics
print(f"Precision (IoU=0.5): {precision}")
print(f"Recall (IoU=0.5): {recall}")
print(f"mAP (IoU=0.5): {map_50}")
print(f"mAP (IoU=0.5:0.95): {map_5095}")

# Save metrics to a file
os.makedirs("metrics", exist_ok=True)  # Create 'metrics' directory if it doesn't exist
file_path = f"metrics/{model_name}.txt"

with open(file_path, "w") as file:
    file.write(f"Precision (IoU=0.5): {precision}\n")
    file.write(f"Recall (IoU=0.5): {recall}\n")
    file.write(f"mAP (IoU=0.5): {map_50}\n")
    file.write(f"mAP (IoU=0.5:0.95): {map_5095}\n")

print(f"Metrics saved to {file_path}")
