# ===================== grasp_visualization.py =====================
import torch
import random
from dataset_loader import CornellGraspDataset
from model import GraspCNN
from visualize import show_rgb_depth_grasps
import os

# Load model
model = GraspCNN()
model.load_state_dict(torch.load("outputs/saved_models/grasp_cnn.pth"))
model.eval()

# Load dataset
val_dataset = CornellGraspDataset(root="./data/cornell-grasp")

# Directory to save outputs
os.makedirs("outputs/grasp_outputs", exist_ok=True)

# Number of random samples to visualize
n_samples = 5

for i in range(n_samples):
    idx = random.randint(0, len(val_dataset) - 1)
    sample = val_dataset[idx]

    with torch.no_grad():
        pred = model(sample['rgb'].unsqueeze(0), sample['depth'].unsqueeze(0))
        save_path = f"outputs/grasp_outputs/grasp_output_{i+1:03d}.png"
        show_rgb_depth_grasps(sample['rgb'], sample['depth'], sample['grasp'], pred[0], save_path=save_path, original_size=(480, 640), resized_size=(224, 224))
        print(f"✅ Saved: {save_path}")
