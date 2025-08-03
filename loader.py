# ===================== loader.py =====================
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import math

class Cutout(object):
    """Implements the Cutout data augmentation technique."""
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip(y - self.length // 2, 0, h), np.clip(y + self.length // 2, 0, h)
            x1, x2 = np.clip(x - self.length // 2, 0, w), np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

class CornellGraspDataset(Dataset):
    """
    Dataset class for the Cornell Grasp Dataset.
    Handles loading images, grasp rectangles, and applying transformations.
    """
    def __init__(self, root='./data/cornell-grasp', split='train', val_split=0.2, seed=42):
        self.root = root
        self.split = split
        self.target_size = (300, 300) # Using a consistent, higher resolution

        # Define transformations for training and validation splits
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Cutout(n_holes=1, length=50)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        all_samples = self._load_dataset()
        if not all_samples:
            raise ValueError(f"No valid samples found in {self.root}. Check dataset path and structure.")

        train_samples, val_samples = train_test_split(all_samples, test_size=val_split, random_state=seed)
        self.samples = train_samples if split == 'train' else val_samples
        print(f"✅ Loaded {len(self.samples)} samples for split = '{split}'")

    def _load_dataset(self):
        """Scans the dataset directory to find and validate samples."""
        samples = []
        for folder in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder)
            if not os.path.isdir(folder_path): continue
            
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('r.png'):
                    base = file.replace('r.png', '')
                    paths = {
                        'rgb': os.path.join(folder_path, base + 'r.png'),
                        'depth': os.path.join(folder_path, base + 'd.tiff'),
                        'pos': os.path.join(folder_path, base + 'cpos.txt')
                    }
                    if all(os.path.exists(p) for p in paths.values()):
                        pos_grasps = self._load_grasp_rectangles(paths['pos'])
                        if len(pos_grasps) > 0:
                            samples.append((paths['rgb'], paths['depth'], pos_grasps))
        return samples

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single sample from the dataset.
        """
        rgb_path, depth_path, pos_grasps = self.samples[idx]
        
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path)
        original_size = (rgb.size[1], rgb.size[0]) # (height, width)

        # Apply augmentations for training split
        if self.split == 'train':
            angle_deg = (np.random.random() - 0.5) * 2 * 30 # Random rotation between -30 and 30 degrees
            rgb = rgb.rotate(angle_deg, resample=Image.BILINEAR)
            depth = depth.rotate(angle_deg, resample=Image.NEAREST)
            
            # Update grasp coordinates to match the rotation
            angle_rad = -np.deg2rad(angle_deg)
            rot_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                                   [math.sin(angle_rad), math.cos(angle_rad)]])
            center = np.array(original_size[::-1]) / 2
            pos_grasps = [(np.dot(g - center, rot_matrix) + center) for g in pos_grasps]

        # Resize images and scale grasp coordinates
        resize_transform = transforms.Resize(self.target_size)
        rgb = resize_transform(rgb)
        depth = resize_transform(depth)
        
        h_scale = self.target_size[0] / original_size[0]
        w_scale = self.target_size[1] / original_size[1]
        scaled_pos_grasps = [g * np.array([w_scale, h_scale]) for g in pos_grasps]

        # Apply RGB transformations (ToTensor, Normalize, Cutout)
        rgb = self.transform(rgb)
        
        # Process depth image
        depth = np.asarray(depth).astype(np.float32)
        # 1. Scale to [0, 1] range
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = torch.from_numpy(depth).unsqueeze(0)
        # 2. Standardize using its own mean and std
        depth = (depth - depth.mean()) / (depth.std() + 1e-8)

        return {
            'rgb': rgb, 
            'depth': depth, 
            'pos_grasps': torch.tensor(np.array(scaled_pos_grasps), dtype=torch.float32)
        }

    def _load_grasp_rectangles(self, file_path):
        """Loads grasp rectangles from a cpos.txt file."""
        grasps = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Process lines in chunks of 4 to form rectangles
        for i in range(0, len(lines) - 3, 4):
            try:
                rect_pts = [list(map(float, line.strip().split())) for line in lines[i:i+4]]
                if len(rect_pts) == 4:
                    rect = np.array(rect_pts)
                    # Ensure the rectangle is valid before adding
                    if not (np.any(np.isnan(rect)) or np.any(np.isinf(rect))):
                        grasps.append(rect)
            except (ValueError, IndexError):
                # Skip malformed lines
                continue
        return grasps
