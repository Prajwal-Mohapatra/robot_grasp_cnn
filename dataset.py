import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

from utils.data_processing import generate_grasp_maps, normalize_depth, normalize_rgb

class GraspDataset(Dataset):
    """
    Dataset for loading Cornell Grasping data and generating ground-truth maps.
    """
    def __init__(self, data_dir, output_size=(224, 224), augment=False):
        self.data_dir = data_dir
        self.output_size = output_size
        self.augment = augment

        # Find all grasp files (cpos.txt)
        self.grasp_files = glob.glob(os.path.join(data_dir, '*', '*cpos.txt'))
        self.grasp_files.sort()
        
        print(f"Found {len(self.grasp_files)} grasp files.")

    def __len__(self):
        return len(self.grasp_files)
        
    def _load_grasp_rectangles(self, file_path):
        """
        Loads grasp rectangles from a cpos.txt file.
        Each rectangle is a set of 4 (x, y) points.
        """
        grasps = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        rect_points = []
        for line in lines:
            x, y = map(float, line.strip().split())
            rect_points.append([x, y])
            if len(rect_points) == 4:
                grasps.append(np.array(rect_points))
                rect_points = []
        return grasps

    def __getitem__(self, idx):
        grasp_file = self.grasp_files[idx]
        
        # Construct paths for RGB and depth images
        base_name = grasp_file.replace('cpos.txt', '')
        rgb_path = base_name + 'r.png'
        depth_path = base_name + 'd.tiff'
        
        # Load images
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path)

        # Load grasp rectangles
        grasps = self._load_grasp_rectangles(grasp_file)

        # Get original image size for scaling grasps
        original_size = rgb_img.size # (width, height)

        # Data Augmentation
        if self.augment:
            angle = (random.random() - 0.5) * 20 # -10 to 10 degrees
            scale = random.random() * 0.2 + 0.9 # 0.9 to 1.1 scale
            
            # Apply rotation
            rgb_img = TF.rotate(rgb_img, angle)
            depth_img = TF.rotate(depth_img, angle)

            # Rotate grasp points
            rot_matrix = cv2.getRotationMatrix2D((original_size[0]/2, original_size[1]/2), -angle, 1.0)
            for i, g in enumerate(grasps):
                g_hom = np.hstack((g, np.ones((g.shape[0], 1))))
                grasps[i] = (rot_matrix @ g_hom.T).T[:, :2]

            # Apply scaling (zoom)
            new_width = int(original_size[0] * scale)
            new_height = int(original_size[1] * scale)
            rgb_img = TF.resize(rgb_img, (new_height, new_width))
            depth_img = TF.resize(depth_img, (new_height, new_width))
            
            # Scale grasp points
            for i in range(len(grasps)):
                grasps[i] *= scale

            # Center crop back to original size
            left = (new_width - original_size[0]) // 2
            top = (new_height - original_size[1]) // 2
            rgb_img = TF.crop(rgb_img, top, left, original_size[1], original_size[0])
            depth_img = TF.crop(depth_img, top, left, original_size[1], original_size[0])

        # Resize images and grasps to network output size
        scale_x = self.output_size[1] / original_size[0]
        scale_y = self.output_size[0] / original_size[1]
        
        rgb_img = TF.resize(rgb_img, self.output_size)
        depth_img = TF.resize(depth_img, self.output_size)
        
        for i in range(len(grasps)):
            grasps[i][:, 0] *= scale_x
            grasps[i][:, 1] *= scale_y

        # Convert to numpy and normalize
        rgb_np = normalize_rgb(np.array(rgb_img))
        depth_np = normalize_depth(np.array(depth_img))

        # Generate ground-truth maps
        q_map, cos_map, sin_map, width_map = generate_grasp_maps(grasps, self.output_size)

        # Convert to Tensors
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float()
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()
        
        # Combine RGB and Depth into a single 4-channel input
        rgbd_tensor = torch.cat((rgb_tensor, depth_tensor), 0)

        gt_maps = {
            'q': torch.from_numpy(q_map).unsqueeze(0).float(),
            'cos': torch.from_numpy(cos_map).unsqueeze(0).float(),
            'sin': torch.from_numpy(sin_map).unsqueeze(0).float(),
            'width': torch.from_numpy(width_map).unsqueeze(0).float()
        }
        
        return rgbd_tensor, gt_maps

if __name__ == '__main__':
    # Test the dataset loader
    print("\nTesting Dataset Loader...")
    # NOTE: You need to have the Cornell dataset in the specified path to run this test.
    data_dir = './data' # Assumes Cornell data is in ./data/cornell
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' not found. Skipping dataset test.")
    else:
        dataset = GraspDataset(data_dir, augment=True)
        
        # Get one sample
        rgbd_tensor, gt_maps = dataset[0]
        
        print(f"Sample loaded successfully.")
        print(f"RGB-D Tensor shape: {rgbd_tensor.shape}")
        print("Ground Truth Maps:")
        for name, tensor in gt_maps.items():
            print(f"  - {name}: {tensor.shape}")
            
        # Visualize the first sample's GT maps
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        axs[0].imshow(rgbd_tensor[:3].permute(1, 2, 0))
        axs[0].set_title('RGB')
        axs[1].imshow(gt_maps['q'].squeeze(), cmap='viridis')
        axs[1].set_title('GT Quality')
        axs[2].imshow(gt_maps['cos'].squeeze(), cmap='viridis')
        axs[2].set_title('GT Cos(2θ)')
        axs[3].imshow(gt_maps['sin'].squeeze(), cmap='viridis')
        axs[3].set_title('GT Sin(2θ)')
        axs[4].imshow(gt_maps['width'].squeeze(), cmap='viridis')
        axs[4].set_title('GT Width')
        plt.tight_layout()
        plt.show()

