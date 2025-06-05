import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob

class CornellGraspDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(glob.glob(os.path.join(root_dir, "*", "rgb.png")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        rgb_path = self.image_files[idx]
        depth_path = rgb_path.replace("rgb.png", "depth.png")

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (480, 480))
        depth = cv2.resize(depth, (480, 480))

        depth = depth.astype(np.float32)
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))  # normalize

        rgb = rgb.astype(np.float32) / 255.0

        # Stack RGB + Depth as 4 channels
        rgbd = np.dstack((rgb, depth))
        rgbd = np.transpose(rgbd, (2, 0, 1))  # CxHxW

        # Dummy labels: replace with actual loading of grasp rectangles here
        # Format: [x_center, y_center, angle_radians, height, width]
        label = np.array([240, 240, 0.0, 50, 50], dtype=np.float32)

        rgbd = torch.tensor(rgbd, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return rgbd, label
