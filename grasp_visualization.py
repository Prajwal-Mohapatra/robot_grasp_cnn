# ===================== grasp_visualization.py =====================
import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset_loader import CornellGraspDataset
from model import GraspCNN
from visualize import draw_grasp
import torchvision.transforms.functional as TF
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = GraspCNN(output_dim=6).to(device)
model.load_state_dict(torch.load('saved_models/grasp_cnn.pth', map_location=device))
model.eval()

# Load validation dataset
dataset = CornellGraspDataset(split='val')
num_samples = len(dataset)
output_dir = "grasp_outputs"
os.makedirs(output_dir, exist_ok=True)

# Pick 5 random indices
random_indices = random.sample(range(num_samples), 5)

for i, idx in enumerate(random_indices):
    sample = dataset[idx]
    rgb, depth = sample['rgb'].unsqueeze(0).to(device), sample['depth'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(rgb, depth)[0]  # shape: (6,)

    # Convert RGB image to numpy for visualization
    rgb_img = TF.to_pil_image(sample['rgb'])
    rgb_np = np.array(rgb_img)

    # Draw predicted grasp
    grasp_img = draw_grasp(rgb_np, output)

    # Save image
    out_path = os.path.join(output_dir, f"grasp_sample_{idx}.png")
    Image.fromarray(grasp_img).save(out_path)
    print(f"✅ Saved grasp visualization to {out_path}")

