# ===================== loader.py =====================
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
        
        if len(all_samples) == 0:
            raise ValueError(f"No valid samples found in {root}. Check dataset structure.")

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

            # Get actual image dimensions for scaling
            actual_size = (rgb.size[1], rgb.size[0])  # (height, width)

            # Resize both RGB and depth to 224×224
            resize_transform = transforms.Resize(self.target_size)
            rgb = resize_transform(rgb)
            depth = resize_transform(depth)

            # Convert to arrays
            rgb = np.asarray(rgb).astype(np.float32) / 255.0
            depth = np.asarray(depth).astype(np.float32)
            
            # Safe depth normalization
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max - depth_min > 1e-8:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)

            # Convert to tensors
            rgb = torch.from_numpy(rgb).permute(2, 0, 1)
            depth = torch.from_numpy(depth).unsqueeze(0)

            # Normalize RGB (ImageNet)
            rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb = (rgb - rgb_mean) / rgb_std

            # Safe depth standardization
            depth_mean, depth_std = depth.mean(), depth.std()
            if depth_std > 1e-8:
                depth = (depth - depth_mean) / depth_std
            else:
                depth = depth - depth_mean

            # Scale grasp coordinates from actual to target size
            scaled_pos_grasps = self._scale_grasps(pos_grasps, actual_size)
            scaled_neg_grasps = self._scale_grasps(neg_grasps, actual_size)

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
            # Return a valid fallback sample instead of None
            return {
                'rgb': torch.zeros(3, *self.target_size),
                'depth': torch.zeros(1, *self.target_size),
                'pos_grasps': torch.empty(0, 4, 2),
                'neg_grasps': torch.empty(0, 4, 2)
            }

    def _scale_grasps(self, grasps, actual_size=None):
        """Scale grasp coordinates from actual image size to target size"""
        if not grasps:
            return []
        
        # Use actual size if provided, otherwise fall back to original_size
        source_size = actual_size if actual_size else self.original_size
        
        scaled_grasps = []
        h_scale = self.target_size[0] / source_size[0]  
        w_scale = self.target_size[1] / source_size[1]  
        
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
        """Load grasp rectangles from file with robust error handling"""
        grasps = []
        rect = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        coords = line.split()
                        if len(coords) != 2:
                            print(f"⚠️ Invalid coordinate format in {file_path}: {line} (expected 2 values)")
                            continue
                        x, y = map(float, coords)
                        rect.append([x, y])
                        if len(rect) == 4:
                            if self._is_valid_rectangle(rect):
                                grasps.append(rect[:])  # Make a copy
                            rect = []
                    except ValueError as ve:
                        print(f"⚠️ Invalid coordinate values in {file_path}: {line} ({ve})")
                        continue
                
                # Handle any remaining incomplete rectangle
                if len(rect) > 0:
                    print(f"⚠️ Incomplete rectangle in {file_path}: {len(rect)} points remaining")
                    
        except Exception as e:
            print(f"⚠️ Error reading grasp file {file_path}: {e}")
        return grasps

    def _is_valid_rectangle(self, rect):
        """Basic validation for grasp rectangle"""
        if len(rect) != 4:
            return False
        
        # Check if all points are within reasonable bounds (more flexible)
        for point in rect:
            x, y = point
            if x < -100 or x > 1000 or y < -100 or y > 1000:  # Allow some margin
                return False
        
        return True
