import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import cv2
from torch.utils.data import Subset # Added

from model import AC_GRConvNet
from dataset import GraspDataset
from utils.data_processing import normalize_rgb, normalize_depth

# --- Configuration ---
# Updated model path to use the fine-tuned model
MODEL_PATH = './outputs/models/ac_grconvnet_finetune_best.pth'
DATA_DIR = './data'
VIS_OUTPUT_DIR = './outputs/visualizations'
# --- FIX: Manually re-typed this line to remove hidden characters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- End of Fix ---
NUM_VISUALIZATIONS = 5

# Added paths for loading the correct data split
SPLIT_DIR = './outputs/splits'
TEST_INDICES_PATH = os.path.join(SPLIT_DIR, 'test_indices.npy')

os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

def post_process_output(q_map, cos_map, sin_map, width_map):
    """
    Post-process the raw output of the network to find the best grasp.
    
    Args:
        q_map (np.ndarray): The 2D quality map.
        cos_map (np.ndarray): The 2D cos(2θ) map.
        sin_map (np.ndarray): The 2D sin(2θ) map.
        width_map (np.ndarray): The 2D width map.

    Returns:
        tuple: (x, y, angle, width) of the best grasp.
    """
    # Find the pixel with the highest quality score
    max_q_val = np.max(q_map)
    # np.unravel_index converts a flat index into a tuple of coordinates
    max_q_idx = np.unravel_index(np.argmax(q_map), q_map.shape)
    y, x = max_q_idx # y is row, x is column

    # Get the angle and width at that pixel
    cos_val = cos_map[y, x]
    sin_val = sin_map[y, x]
    width_val = width_map[y, x]
    
    # Decode the angle
    # angle = atan2(sin, cos) / 2
    angle = np.arctan2(sin_val, cos_val) / 2.0
    
    # Denormalize width (assuming max width of 150 pixels for Cornell dataset)
    MAX_GRASP_WIDTH = 150.0
    width_pixels = width_val * MAX_GRASP_WIDTH
    
    return x, y, angle, width_pixels

def draw_grasp(ax, x, y, angle, width, color='r'):
    """
    Draws a grasp rectangle on a matplotlib axis.
    """
    w = width / 2
    h = 10 # Gripper height for visualization, can be adjusted
    
    # Create points for the rectangle centered at (0,0)
    points = np.array([
        [-h, -w], [h, -w], [h, w], [-h, w]
    ])
    
    # Create rotation matrix
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # Rotate points
    rotated_points = (rot_matrix @ points.T).T
    
    # Translate points to the grasp center
    translated_points = rotated_points + np.array([x, y])
    
    # Draw the rectangle
    rect = plt.Polygon(translated_points, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    
    # Draw center point
    ax.plot(x, y, marker='o', color=color, markersize=4)


def main():
    print(f"Using device: {DEVICE}")
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        return
        
    model = AC_GRConvNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Dataset for visualization samples
    # Updated to load only the test set indices
    try:
        full_dataset = GraspDataset(DATA_DIR, augment=False)
        if not os.path.exists(TEST_INDICES_PATH):
            print(f"Error: Test split file not found at '{TEST_INDICES_PATH}'.")
            print("Please run train.py first to generate the data splits.")
            return
            
        if len(full_dataset) == 0:
            print("Dataset is empty. Please check the data directory.")
            return
            
        test_indices = np.load(TEST_INDICES_PATH)
        test_dataset = Subset(full_dataset, test_indices)
        print(f"Loaded test set with {len(test_dataset)} samples for visualization.")
    except FileNotFoundError as e:
        print(e)
        return

    
    for i in range(NUM_VISUALIZATIONS):
        print(f"\nVisualizing sample {i+1}/{NUM_VISUALIZATIONS}...")
        
        # Get a random sample from the test set
        if len(test_dataset) == 0:
            print("Test dataset is empty, cannot visualize.")
            break
        idx_in_test_set = random.randint(0, len(test_dataset) - 1)
        
        # Handle potential None from dataset loader
        dataset_item = test_dataset[idx_in_test_set]
        if dataset_item is None:
            print(f"Skipping corrupt sample at index {idx_in_test_set}")
            continue
            
        rgbd_tensor, _ = dataset_item
        
        # Get the original index from the full dataset for the title
        original_idx = test_indices[idx_in_test_set]
        
        # Perform inference
        with torch.no_grad():
            pred_maps = model(rgbd_tensor.unsqueeze(0).to(DEVICE))
        
        # --- (No changes needed here, predict.py logic was correct) ---
        pred_maps_np = pred_maps.squeeze().cpu().numpy()
        q_map_raw, cos_map_raw, sin_map_raw, width_map_raw = np.split(pred_maps_np, 4)
        q_map = q_map_raw.squeeze()
        cos_map = cos_map_raw.squeeze()
        sin_map = sin_map_raw.squeeze()
        width_map = width_map_raw.squeeze()
        # --- End of section ---

        # Post-process to find the best grasp (for visualization)
        x, y, angle, width = post_process_output(q_map, cos_map, sin_map, width_map)
        
        # Visualization
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Grasp Prediction - Sample {original_idx} (from Test Set)', fontsize=16)

        # 1. Original RGB Image
        # Denormalize for display
        # --- FIX: Corrected typo perm_ute -> permute ---
        rgb_img = rgbd_tensor[:3].permute(1, 2, 0).numpy()
        # --- End of Fix ---
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        rgb_img = (rgb_img * 255.0).astype(np.uint8)
        axs[0].imshow(rgb_img)
        axs[0].set_title('Input RGB')
        axs[0].axis('off')

        # 2. Predicted Quality Map
        im1 = axs[1].imshow(q_map, cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title('Predicted Quality (Q)')
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # 3. Predicted Angle Map
        # --- (No changes needed here, logic was correct) ---
        angle_map = np.arctan2(sin_map, cos_map) / 2.0
        # --- End of section ---
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
        plt.close(fig) # Close figure to save memory
        print(f"✅ Visualization saved to {save_path}")

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
         print(f"Error: Data directory '{DATA_DIR}' is empty or does not exist.")
         print("Please download the Cornell Grasp Dataset and place it in the 'data' folder.")
    else:
        main()
