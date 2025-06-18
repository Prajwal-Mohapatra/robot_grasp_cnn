# ===================== dataset_loader.py ===================== #
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class GraspDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.rgb_paths = sorted(glob.glob(os.path.join(data_dir, split, '*_RGB.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(data_dir, split, '*_depth.tiff')))
        self.label_paths = sorted(glob.glob(os.path.join(data_dir, split, '*_cpos.txt')))

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        depth = Image.open(self.depth_paths[idx])
        grasps = self.load_grasps(self.label_paths[idx])

        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)

        depth = depth.unsqueeze(0)  # [1, H, W]
        rgb = rgb / 255.0

        return rgb, depth, grasps

    def load_grasps(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        grasps = []
        for i in range(0, len(lines), 4):
            pts = [list(map(float, lines[i + j].strip().split())) for j in range(4)]
            pts = np.array(pts)
            g = self.rectangle_to_grasp(pts)
            grasps.append(g)
        return grasps

    def rectangle_to_grasp(self, rect):
        center = np.mean(rect, axis=0)
        dx = rect[1] - rect[0]
        width = np.linalg.norm(dx)
        height = np.linalg.norm(rect[2] - rect[1])
        theta = np.arctan2(dx[1], dx[0])
        return torch.tensor([center[0], center[1], theta, width, height, 1.0], dtype=torch.float32)
