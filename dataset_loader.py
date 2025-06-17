# ===================== dataset_loader.py =====================
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CornellGraspDataset(Dataset):
    def __init__(self, root='./data/cornell-grasp', transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._load_dataset()

    def _load_dataset(self):
        samples = []
        for folder in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder)
            if not os.path.isdir(folder_path):
                continue

            for file in sorted(os.listdir(folder_path)):
                if file.endswith('r.png'):  # RGB image
                    base_name = file.replace('r.png', '')  # e.g., pcd0100
                    rgb_path = os.path.join(folder_path, base_name + 'r.png')
                    depth_path = os.path.join(folder_path, base_name + 'd.tiff')
                    grasp_path = os.path.join(folder_path, base_name + 'cpos.txt')

                    if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(grasp_path):
                        grasps = self._load_grasp_rectangles(grasp_path)
                        if len(grasps) > 0:
                            samples.append((rgb_path, depth_path, grasp_path))
                        else:
                            print(f"⚠️ No valid grasps in {grasp_path}")
                    else:
                        print(f"⚠️ Missing files for base {base_name} in {folder}")

        print(f"✅ Total valid samples loaded: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, grasp_path = self.samples[idx]

        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path)

        # Convert to numpy arrays
        rgb = np.asarray(rgb).astype(np.float32) / 255.0
        depth = np.asarray(depth).astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Convert to torch tensors
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

        # Normalize RGB using ImageNet stats
        rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = (rgb - rgb_mean) / rgb_std

        # Normalize depth to mean 0 and std 1
        depth_mean = depth.mean()
        depth_std = depth.std() + 1e-8
        depth = (depth - depth_mean) / depth_std

        # Load grasp rectangles
        grasps = self._load_grasp_rectangles(grasp_path)
        if len(grasps) == 0:
            return None  # skip if no valid grasps
        grasp_tensor = torch.tensor(grasps, dtype=torch.float32)  # [N, 4, 2]

        return {'rgb': rgb, 'depth': depth, 'grasp': grasp_tensor}

    def _load_grasp_rectangles(self, file_path):
        grasps = []
        rect = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    x, y = map(float, line.split())
                    rect.append([x, y])
                    if len(rect) == 4:
                        grasps.append(rect)
                        rect = []
                except:
                    continue
        return grasps
