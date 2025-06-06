# ===================== grasp_visualization.py =====================
import torch
from dataset_loader import CornellGraspDataset
from model import GraspCNN
from visualize import show_rgb_depth_grasps

model = GraspCNN()
model.load_state_dict(torch.load("saved_models/grasp_cnn.pth"))
model.eval()

val_dataset = CornellGraspDataset(root="./data/cornell")
sample = val_dataset[0]

with torch.no_grad():
    pred = model(sample['rgb'].unsqueeze(0), sample['depth'].unsqueeze(0))
    show_rgb_depth_grasps(sample['rgb'], sample['depth'], sample['grasps'], pred[0])
