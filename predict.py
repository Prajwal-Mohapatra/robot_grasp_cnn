import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import cv2

from model import GRConvNet
from dataset import GraspDataset
from utils.data_processing import normalize_rgb, normalize_depth

# --- Configuration ---
MODEL_PATH = './outputs/models/grconvnet_best.pth'
DATA_DIR = './data'
VIS_OUTPUT_DIR = './outputs/visualizations'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_VISUALIZATIONS = 5

os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

def post_process_output(q_map, cos_map, sin_map, width_map):
    """
    Post-process the raw output of the network to find the best grasp.
    
    Args:
        q_map (np.ndarray): The quality map.
        cos_map (np.ndarray): The cos(2θ) map.
        sin_map (np.ndarray): The sin(2θ) map.
        width_map (np.ndarray): The width map.

    Returns:
        tuple: (x, y, angle, width) of the best grasp.
    """
    # Find the pixel with the highest quality score
    max_q_val = np.max(q_map)
    max_q_idx = np.unravel_index(np.argmax(q_map), q_map.shape)
    y, x = max_q_idx

    # Get the angle and width at that pixel
    cos_val = cos_map[y, x]
    sin_val = sin_map[y, x]
    width_val = width_map[y, x]
    
    # Decode the angle
    angle = np.arctan2(sin_val, cos_val) / 2.0
    
    # Denormalize width (assuming max width of 150 pixels)
    MAX_GRASP_WIDTH = 150.0
    width_pixels = width_val * MAX_GRASP_WIDTH
    
    return x, y, angle, width_pixels

def draw_grasp(ax, x, y, angle, width, color='r'):
    """
    Draws a grasp rectangle on a matplotlib axis.
    """
    w = width / 2
    h = 10 # Gripper height for visualization
    
    # Create points for the rectangle centered at (x,y)
    points = np.array([
        [-h, -w], [h, -w], [h, w], [-h, w]
    ])
    
    # Rotate points
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_points = (rot_matrix @ points.T).T
    
    # Translate points to the grasp center
    translated_points = rotated_points + np.array([x, y])
    
    # Draw the rectangle
    rect = plt.Polygon(translated_points, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    # Draw center point
    ax.plot(x, y, f'{color}o', markersize=4)

def main():
    print(f"Using device: {DEVICE}")
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        return
        
    model = GRConvNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Dataset for visualization samples
    dataset = GraspDataset(DATA_DIR, augment=False)
    
    for i in range(NUM_VISUALIZATIONS):
        print(f"\nVisualizing sample {i+1}/{NUM_VISUALIZATIONS}...")
        
        # Get a random sample
        idx = random.randint(0, len(dataset) - 1)
        rgbd_tensor, _ = dataset[idx]
        
        # Perform inference
        with torch.no_grad():
            pred_maps = model(rgbd_tensor.unsqueeze(0).to(DEVICE))
        
        pred_maps_np = pred_maps.squeeze().cpu().numpy()
        q_map, cos_map, sin_map, width_map = np.split(pred_maps_np, 4)

        # Post-process to find the best grasp
        x, y, angle, width = post_process_output(q_map, cos_map, sin_map, width_map)
        
        # Visualization
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Grasp Prediction - Sample {idx}', fontsize=16)

        # 1. Original RGB Image
        rgb_img = rgbd_tensor[:3].permute(1, 2, 0).numpy()
        axs[0].imshow(rgb_img)
        axs[0].set_title('Input RGB')
        axs[0].axis('off')

        # 2. Predicted Quality Map
        im1 = axs[1].imshow(q_map, cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title('Predicted Quality (Q)')
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # 3. Predicted Angle Map
        # We visualize the angle itself, not the cos/sin components
        angle_map = np.arctan2(sin_map, cos_map) / 2.0
        im2 = axs[2].imshow(angle_map, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
        axs[2].set_title('Predicted Angle (θ)')
        axs[2].axis('off')
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        
        # 4. RGB with Best Grasp
        axs[3].imshow(rgb_img)
        axs[3].set_title('Best Predicted Grasp')
        axs[3].axis('off')
        draw_grasp(axs[3], x, y, angle, width, color='lime')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(VIS_OUTPUT_DIR, f'prediction_{i+1:02d}.png')
        plt.savefig(save_path)
        plt.show()
        print(f"✅ Visualization saved to {save_path}")

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
         print(f"Error: Data directory '{DATA_DIR}' is empty or does not exist.")
         print("Please download the Cornell Grasp Dataset and place it in the 'data' folder.")
    else:
        main()
