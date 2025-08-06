import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from skimage.draw import polygon

class CornellGraspDataset(Dataset):
    def __init__(self, root='./data/cornell-grasp', split='train', val_split=0.2, seed=42):
        self.root = root
        self.split = split
        self.target_size = (300, 300)

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
        # Unchanged
        samples = []
        for folder in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder)
            if not os.path.isdir(folder_path): continue
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('r.png'):
                    base = file.replace('r.png', '')
                    paths = {'rgb': os.path.join(folder_path, base + 'r.png'),
                             'depth': os.path.join(folder_path, base + 'd.tiff'),
                             'pos': os.path.join(folder_path, base + 'cpos.txt')}
                    if all(os.path.exists(p) for p in paths.values()):
                        samples.append(paths)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        
        rgb = Image.open(paths['rgb']).convert('RGB').resize(self.target_size)
        depth = Image.open(paths['depth']).resize(self.target_size)
        
        pos_grasps, ang_grasps, width_grasps = self._load_and_process_grasps(paths['pos'], (480, 640), self.target_size)

        quality_map, angle_map, width_map = self.generate_grasp_maps(pos_grasps, ang_grasps, width_grasps)

        rgb = self.transform(rgb)
        
        depth = np.asarray(depth).astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = (depth - depth.mean()) / (depth.std() + 1e-8)
        
        quality_map = torch.from_numpy(quality_map).float()
        angle_map = torch.from_numpy(angle_map).float()
        width_map = torch.from_numpy(width_map).float()

        # --- NEW: Flip Augmentation for training data ---
        if self.split == 'train':
            # Horizontal Flip
            if random.random() < 0.5:
                rgb = torch.fliplr(rgb)
                depth = torch.fliplr(depth)
                quality_map = torch.fliplr(quality_map)
                angle_map = torch.fliplr(angle_map)
                width_map = torch.fliplr(width_map)
                # When flipping horizontally, sin(2*(-a)) = -sin(2a)
                angle_map[0, :, :] = -angle_map[0, :, :]
            
            # Vertical Flip
            if random.random() < 0.5:
                rgb = torch.flipud(rgb)
                depth = torch.flipud(depth)
                quality_map = torch.flipud(quality_map)
                angle_map = torch.flipud(angle_map)
                width_map = torch.flipud(width_map)
                # When flipping vertically, sin(2*(-a)) = -sin(2a)
                angle_map[0, :, :] = -angle_map[0, :, :]

        return {
            'rgb': rgb, 'depth': depth, 
            'gt_bbs': pos_grasps,
            'quality_map': quality_map,
            'angle_map': angle_map,
            'width_map': width_map
        }

    def generate_grasp_maps(self, rects, angles, widths):
        # Unchanged
        quality_map = np.zeros(self.target_size, dtype=np.float32)
        angle_map = np.zeros((2, *self.target_size), dtype=np.float32)
        width_map = np.zeros(self.target_size, dtype=np.float32)
        for rect, angle, width in zip(rects, angles, widths):
            rr, cc = polygon(rect[:, 1], rect[:, 0], self.target_size)
            quality_map[rr, cc] = 1.0
            angle_map[0, rr, cc] = np.sin(2 * angle)
            angle_map[1, rr, cc] = np.cos(2 * angle)
            width_map[rr, cc] = width / self.target_size[1]
        return quality_map, angle_map, width_map

    def _load_and_process_grasps(self, file_path, original_size, target_size):
        # Unchanged
        pos_grasps, ang_grasps, width_grasps = [], [], []
        h_scale = target_size[0] / original_size[0]
        w_scale = target_size[1] / original_size[1]
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for i in range(0, len(lines) - 3, 4):
            try:
                rect_pts = [list(map(float, line.strip().split())) for line in lines[i:i+4]]
                if len(rect_pts) == 4:
                    rect = np.array(rect_pts)
                    rect[:, 0] *= w_scale
                    rect[:, 1] *= h_scale
                    dx = rect[1, 0] - rect[0, 0]
                    dy = rect[1, 1] - rect[0, 1]
                    width = np.linalg.norm([dx, dy])
                    angle = np.arctan2(dy, dx)
                    pos_grasps.append(rect)
                    ang_grasps.append(angle)
                    width_grasps.append(width)
            except (ValueError, IndexError):
                continue
        return pos_grasps, ang_grasps, width_grasps
