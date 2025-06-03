import os
import re
import matplotlib.pyplot as plt

def parse_metric(line):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if matches:
        return float(matches[-1])
    else:
        raise ValueError(f"No float found in line: {line}")

# Load metrics from files
metrics = {}
for file in os.listdir("metrics"):
    if file.endswith(".txt"):
        with open(os.path.join("metrics", file)) as f:
            lines = f.readlines()
            model = file.replace(".txt", "")
            metrics[model] = {
                "Precision": parse_metric(lines[0]),
                "Recall": parse_metric(lines[1]),
                "mAP@0.5": parse_metric(lines[2]),
            }

# Sorted list of model names
labels = sorted(metrics.keys())

# Metrics to plot
metric_names = ["Precision", "Recall", "mAP@0.5"]

# Create one bar chart per metric
for metric_name in metric_names:
    values = [metrics[label][metric_name] for label in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="skyblue")
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{metric_name} Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f"{metric_name.lower().replace('@', '').replace(':', '')}.png"
    plt.savefig(filename)
    plt.close()

print("✅ All metric plots saved (one per image).")
