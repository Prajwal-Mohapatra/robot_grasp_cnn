# ===================== dataset_loader.py =====================
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CornellGraspDataset(Dataset):
    def __init__(self, root='./data/cornell', transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._load_dataset()

    def _load_dataset(self):
        samples = []
        for obj_folder in sorted(os.listdir(self.root)):
            obj_path = os.path.join(self.root, obj_folder)
            if not os.path.isdir(obj_path):
                continue
            try:
                rgb_path = os.path.join(obj_path, f"{obj_folder}_rgb.png")
                depth_path = os.path.join(obj_path, f"{obj_folder}_dep.png")
                grasp_path = os.path.join(obj_path, f"{obj_folder}_grasps.txt")
                if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(grasp_path):
                    samples.append((rgb_path, depth_path, grasp_path))
            except Exception as e:
                continue
        print(f"Total samples loaded: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, grasp_path = self.samples[idx]

        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path)

        rgb = np.asarray(rgb).astype(np.float32) / 255.0
        depth = np.asarray(depth).astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)

        grasps = self._load_grasp_rectangles(grasp_path)
        grasp_tensor = torch.tensor(grasps, dtype=torch.float32)

        return {'rgb': rgb, 'depth': depth, 'grasps': grasp_tensor}

    def _load_grasp_rectangles(self, file_path):
        grasps = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 4):
                    rect = []
                    for j in range(4):
                        x, y = map(float, lines[i + j].strip().split())
                        rect.append([x, y])
                    grasps.append(rect)
        except Exception as e:
            pass
        return grasps
