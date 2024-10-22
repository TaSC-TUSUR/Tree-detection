# -*- coding: utf-8 -*-
"""gpo-ai.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Or_MU49TSL-Y3M-aZ0BMF9aw-tZw5BKk
"""

import os
from google.colab import drive

drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/sukh/v3')

# Install YOLOv8
!pip install ultralytics

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# Load a pretrained YOLOv8 model
#  (YOLOv8n is the nano version, you can choose 's', 'm', 'l', etc.)
model = YOLO('yolov8n.pt')

model.train(data='data.yaml', epochs=50, imgsz=1024, batch=16)

import torch

# Get the current device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the memory statistics
total_memory = torch.cuda.get_device_properties(device).total_memory
allocated_memory = torch.cuda.memory_allocated(device)
reserved_memory = torch.cuda.memory_reserved(device)
free_memory = total_memory - allocated_memory

# Print memory details
print(f'Total Memory: {total_memory / (1024 ** 2):.2f} MiB')
print(f'Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MiB')
print(f'Reserved Memory: {reserved_memory / (1024 ** 2):.2f} MiB')
print(f'Free Memory: {free_memory / (1024 ** 2):.2f} MiB')