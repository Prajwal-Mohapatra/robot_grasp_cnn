# ===================== dataset_loader.py =====================
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

class CornellGraspDataset(Dataset):
    def __init__(self, root='./data/cornell-grasp', split='train', transform=None, val_split=0.2, seed=42):
        self.root = root
        self.transform = transform
        self.val_split = val_split
        self.seed = seed

        # Load all valid samples
        all_samples = self._load_dataset()

        # Split into train/val
        if split == 'train':
            self.samples, _ = train_test_split(all_samples, test_size=val_split, random_state=seed)
        elif split == 'val':
            _, self.samples = train_test_split(all_samples, test_size=val_split, random_state=seed)
        else:
            self.samples = all_samples

        print(f"✅ Loaded {len(self.samples)} samples for split = '{split}'")

    def _load_dataset(self):
        samples = []
        for folder in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder)
            if not os.path.isdir(folder_path):
                continue

            for file in sorted(os.listdir(folder_path)):
                if file.endswith('r.png'):
                    base = file.replace('r.png', '')
                    rgb_path = os.path.join(folder_path, base + 'r.png')
                    depth_path = os.path.join(folder_path, base + 'd.tiff')
                    grasp_path = os.path.join(folder_path, base + 'cpos.txt')

                    if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(grasp_path):
                        grasps = self._load_grasp_rectangles(grasp_path)
                        if len(grasps) > 0:
                            samples.append((rgb_path, depth_path, grasp_path))
                        else:
                            print(f"⚠️ No valid grasps in {grasp_path}")
                    else:
                        print(f"⚠️ Missing files for base {base} in {folder}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            rgb_path, depth_path, grasp_path = self.samples[idx]
            rgb = Image.open(rgb_path).convert('RGB')
            depth = Image.open(depth_path)

            rgb = np.asarray(rgb).astype(np.float32) / 255.0
            depth = np.asarray(depth).astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            rgb = torch.from_numpy(rgb).permute(2, 0, 1)
            depth = torch.from_numpy(depth).unsqueeze(0)

            # Normalize RGB (ImageNet)
            rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb = (rgb - rgb_mean) / rgb_std

            # Normalize depth
            depth = (depth - depth.mean()) / (depth.std() + 1e-8)

            grasps = self._load_grasp_rectangles(grasp_path)
            if len(grasps) == 0:
                return None

            grasp_tensor = torch.tensor(grasps, dtype=torch.float32)  # [N, 4, 2]
            return {'rgb': rgb, 'depth': depth, 'grasp': grasp_tensor}

        except Exception as e:
            print(f"❌ Error loading index {idx}: {e}")
            return None

    def _load_grasp_rectangles(self, file_path):
        grasps = []
        rect = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    x, y = map(float, line.split())
                    rect.append([x, y])
                    if len(rect) == 4:
                        grasps.append(rect)
                        rect = []
        except Exception as e:
            print(f"⚠️ Error reading grasp file {file_path}: {e}")
        return grasps
