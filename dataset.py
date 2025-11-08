import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import cv2

# Updated import to use the new Gaussian map generator
from utils.data_processing import generate_grasp_maps_gaussian, normalize_depth, normalize_rgb

class GraspDataset(Dataset):
    """
    Dataset for loading the Cornell Grasping data.
    """
    def __init__(self, data_dir, output_size=(224, 224), augment=False):
        self.data_dir = data_dir
        self.output_size = output_size
        self.augment = augment

        search_path = os.path.join(data_dir, '**', '*cpos.txt')
        self.grasp_files = glob.glob(search_path, recursive=True)
        self.grasp_files.sort()
        
        print(f"Searching for grasp files in: {search_path}")
        print(f"Found {len(self.grasp_files)} grasp files.")
        
        # --- FIX 1 (from PDF): Add warning for incomplete dataset ---
        print("Expected ~885-1035 files for the full Cornell dataset.")
        if len(self.grasp_files) < 800:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"! WARNING: Dataset appears incomplete ({len(self.grasp_files)} files found) !")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # --- End of Fix ---

        if len(self.grasp_files) == 0:
            raise FileNotFoundError(
                f"No grasp files ('*cpos.txt') found in directory '{data_dir}'. "
            )

    def __len__(self):
        return len(self.grasp_files)
        
    def _load_grasp_rectangles(self, file_path):
        """
        Loads grasp rectangles from a cpos.txt file.
        """
        grasps = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        rect_points = []
        for line in lines:
            try:
                x, y = map(float, line.strip().split())
                rect_points.append([x, y])
                if len(rect_points) == 4:
                    grasps.append(np.array(rect_points))
                    rect_points = []
            except ValueError:
                continue
        return grasps

    def __getitem__(self, idx):
        grasp_file = self.grasp_files[idx]
        
        base_name = grasp_file.replace('cpos.txt', '')
        rgb_path = base_name + 'r.png'
        depth_path = base_name + 'd.tiff'

        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
            depth_img = Image.open(depth_path) # Load as PIL Image
            grasps = self._load_grasp_rectangles(grasp_file)
            original_size = rgb_img.size # (width, height)

            if self.augment:
                # 1. Random Horizontal Flip
                if random.random() > 0.5:
                    rgb_img = F.hflip(rgb_img)
                    depth_img = F.hflip(depth_img)
                    for i in range(len(grasps)):
                        grasps[i][:, 0] = original_size[0] - grasps[i][:, 0]
                
                # 2. Random Rotation
                angle = (random.random() - 0.5) * 20 # -10 to 10 degrees
                rgb_img = F.rotate(rgb_img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
                depth_img = F.rotate(depth_img, angle, interpolation=transforms.InterpolationMode.NEAREST)
                rot_matrix = cv2.getRotationMatrix2D((original_size[0]/2, original_size[1]/2), -angle, 1.0)
                for i, g in enumerate(grasps):
                    g_hom = np.hstack((g, np.ones((g.shape[0], 1))))
                    grasps[i] = (rot_matrix @ g_hom.T).T[:, :2]

                # 3. Random Translation
                translate_x = random.randint(-20, 20)
                translate_y = random.randint(-20, 20)
                rgb_img = F.affine(rgb_img, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0)
                depth_img = F.affine(depth_img, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0)
                for i in range(len(grasps)):
                    grasps[i][:, 0] += translate_x
                    grasps[i][:, 1] += translate_y

                # 4. Random Scaling (Zoom)
                scale = random.random() * 0.2 + 0.9 # 0.9 to 1.1 scale
                new_width = int(original_size[0] * scale)
                new_height = int(original_size[1] * scale)
                rgb_img = F.resize(rgb_img, (new_height, new_width))
                depth_img = F.resize(depth_img, (new_height, new_width))
                for i in range(len(grasps)):
                    grasps[i] *= scale
                left = (new_width - original_size[0]) // 2
                top = (new_height - original_size[1]) // 2
                rgb_img = F.crop(rgb_img, top, left, original_size[1], original_size[0])
                depth_img = F.crop(depth_img, top, left, original_size[1], original_size[0])
                for i in range(len(grasps)):
                    grasps[i][:, 0] -= left
                    grasps[i][:, 1] -= top

                # 5. Random Color Jittering
                jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                rgb_img = jitter(rgb_img)

            # Resize images and grasps to network output size
            scale_x = self.output_size[1] / original_size[0]
            scale_y = self.output_size[0] / original_size[1]
            
            rgb_img = F.resize(rgb_img, self.output_size)
            depth_img = F.resize(depth_img, self.output_size)
            
            for i in range(len(grasps)):
                grasps[i][:, 0] *= scale_x
                grasps[i][:, 1] *= scale_y

            # Convert to numpy and normalize
            rgb_np = normalize_rgb(rgb_img)
            # Pass PIL image to normalize_depth which converts to numpy
            depth_np = normalize_depth(depth_img) 

            # --- FIX 2 (from PDF): Reduce Gaussian Sigma from 5 to 2 ---
            q_map, cos_map, sin_map, width_map = generate_grasp_maps_gaussian(grasps, self.output_size, sigma=2)
            # --- End of Fix ---

            # Convert to Tensors
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float()
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()
            
            rgbd_tensor = torch.cat((rgb_tensor, depth_tensor), 0)

            gt_maps = {
                'q': torch.from_numpy(q_map).unsqueeze(0).float(),
                'cos': torch.from_numpy(cos_map).unsqueeze(0).float(),
                'sin': torch.from_numpy(sin_map).unsqueeze(0).float(),
                'width': torch.from_numpy(width_map).unsqueeze(0).float()
            }
            
            return rgbd_tensor, gt_maps
        except FileNotFoundError as e:
            print(f"Error loading files for index {idx}: {e}")
            return None
        except Exception as e:
            print(f"Error processing index {idx}, file: {grasp_file}, error: {e}")
            return None # Return None to be handled by collate_fn or dataloader

if __name__ == '__main__':
    # Test the dataset loader
    print("\nTesting Dataset Loader with Gaussian Maps...")
    data_dir = './data' 
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' not found. Skipping dataset test.")
    else:
        try:
            dataset = GraspDataset(data_dir, augment=True)
            if len(dataset) == 0:
                print("Dataset loaded but contains 0 samples. Exiting test.")
                exit()
                
            rgbd_tensor, gt_maps = dataset[0]
            
            print(f"Sample loaded successfully.")
            print(f"RGB-D Tensor shape: {rgbd_tensor.shape}")
            print("Ground Truth Maps:")
            for name, tensor in gt_maps.items():
                print(f"  - {name}: {tensor.shape}")
                
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            rgb_img_vis = rgbd_tensor[:3].permute(1, 2, 0).numpy()
            rgb_img_vis = (rgb_img_vis - rgb_img_vis.min()) / (rgb_img_vis.max() - rgb_img_vis.min())
            axs[0].imshow(rgb_img_vis)
            axs[0].set_title('RGB')
            axs[1].imshow(gt_maps['q'].squeeze(), cmap='viridis')
            axs[1].set_title('GT Quality (Gaussian)')
            axs[2].imshow(gt_maps['cos'].squeeze(), cmap='viridis')
            axs[2].set_title('GT Cos(2θ)')
            axs[3].imshow(gt_maps['sin'].squeeze(), cmap='viridis')
            axs[3].set_title('GT Sin(2θ)')
            axs[4].imshow(gt_maps['width'].squeeze(), cmap='viridis')
            axs[4].set_title('GT Width')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error during dataset test: {e}")
