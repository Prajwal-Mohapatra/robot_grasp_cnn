# ===================== dataset_loader.py =====================
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

class CornellGraspDataset(Dataset):
    def __init__(self, root='./data/cornell-grasp', split='train', transform=None, val_split=0.2, seed=42):
        self.root = root
        self.transform = transform
        self.val_split = val_split
        self.seed = seed
        self.original_size = (480, 640)  # (height, width)
        self.target_size = (224, 224)

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
                    pos_grasp_path = os.path.join(folder_path, base + 'cpos.txt')
                    neg_grasp_path = os.path.join(folder_path, base + 'cneg.txt')

                    if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(pos_grasp_path):
                        pos_grasps = self._load_grasp_rectangles(pos_grasp_path)
                        neg_grasps = self._load_grasp_rectangles(neg_grasp_path) if os.path.exists(neg_grasp_path) else []
                        
                        if len(pos_grasps) > 0:
                            samples.append((rgb_path, depth_path, pos_grasps, neg_grasps))
                        else:
                            print(f"⚠️ No valid positive grasps in {pos_grasp_path}")
                    else:
                        continue

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            rgb_path, depth_path, pos_grasps, neg_grasps = self.samples[idx]
            
            # Load images
            rgb = Image.open(rgb_path).convert('RGB')
            depth = Image.open(depth_path)

            # Resize both RGB and depth to 224×224
            resize_transform = transforms.Resize(self.target_size)
            rgb = resize_transform(rgb)
            depth = resize_transform(depth)

            # Convert to arrays
            rgb = np.asarray(rgb).astype(np.float32) / 255.0
            depth = np.asarray(depth).astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            # Convert to tensors
            rgb = torch.from_numpy(rgb).permute(2, 0, 1)
            depth = torch.from_numpy(depth).unsqueeze(0)

            # Normalize RGB (ImageNet)
            rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb = (rgb - rgb_mean) / rgb_std

            # Normalize depth
            depth = (depth - depth.mean()) / (depth.std() + 1e-8)

            # Scale grasp coordinates from original to target size
            scaled_pos_grasps = self._scale_grasps(pos_grasps)
            scaled_neg_grasps = self._scale_grasps(neg_grasps)

            # Convert to tensors
            pos_grasp_tensor = torch.tensor(scaled_pos_grasps, dtype=torch.float32) if scaled_pos_grasps else torch.empty(0, 4, 2)
            neg_grasp_tensor = torch.tensor(scaled_neg_grasps, dtype=torch.float32) if scaled_neg_grasps else torch.empty(0, 4, 2)

            return {
                'rgb': rgb, 
                'depth': depth, 
                'pos_grasps': pos_grasp_tensor,
                'neg_grasps': neg_grasp_tensor
            }

        except Exception as e:
            print(f"❌ Error loading index {idx}: {e}")
            return None

    def _scale_grasps(self, grasps):
        """Scale grasp coordinates from original image size to target size"""
        if not grasps:
            return []
        
        scaled_grasps = []
        h_scale = self.target_size[0] / self.original_size[0]  # 224/480
        w_scale = self.target_size[1] / self.original_size[1]  # 224/640
        
        for grasp in grasps:
            scaled_grasp = []
            for point in grasp:
                x, y = point
                scaled_x = x * w_scale
                scaled_y = y * h_scale
                scaled_grasp.append([scaled_x, scaled_y])
            scaled_grasps.append(scaled_grasp)
        
        return scaled_grasps

    def _load_grasp_rectangles(self, file_path):
        """Load grasp rectangles from file"""
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
                        if self._is_valid_rectangle(rect):
                            grasps.append(rect)
                        rect = []
        except Exception as e:
            print(f"⚠️ Error reading grasp file {file_path}: {e}")
        return grasps

    def _is_valid_rectangle(self, rect):
        """Basic validation for grasp rectangle"""
        if len(rect) != 4:
            return False
        
        # Check if all points are within image bounds
        for point in rect:
            x, y = point
            if x < 0 or x >= self.original_size[1] or y < 0 or y >= self.original_size[0]:
                return False
        
        return True
