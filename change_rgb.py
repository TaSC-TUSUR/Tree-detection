import cv2
import numpy as np
import json
import os

# === Load average stats from JSON ===
with open('stats/average_stats.json', 'r') as f:
    stats = json.load(f)
r_scale = stats['average_r']
g_scale = stats['average_g']
b_scale = stats['average_b']

# === Functions ===
def zero_cost(data):
    return data[data != 0]

def ch_size(original_data, filtered_data, target_mean):
    scale_factor = target_mean / np.mean(filtered_data)
    scaled = original_data * scale_factor
    return np.clip(scaled, 0, 255).astype(np.uint8)

# === Directories ===
input_dir = 'soser'  # Source images directory
output_dir = 'gagaed'  # Directory to save adjusted images
os.makedirs(output_dir, exist_ok=True)

# === Process each image in the directory ===
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not read image: {filename}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        b_channel = image[:, :, 2]

        r_data = r_channel.ravel().copy()
        g_data = g_channel.ravel().copy()
        b_data = b_channel.ravel().copy()

        new_r = zero_cost(r_data)
        new_g = zero_cost(g_data)
        new_b = zero_cost(b_data)

        r_scaled = ch_size(r_data, new_r, r_scale).reshape(height, width)
        g_scaled = ch_size(g_data, new_g, g_scale).reshape(height, width)
        b_scaled = ch_size(b_data, new_b, b_scale).reshape(height, width)

        merged = cv2.merge([b_scaled, g_scaled, r_scaled])
        cv2.imwrite(output_path, merged)

        print(f"Processed and saved: {output_path}")
