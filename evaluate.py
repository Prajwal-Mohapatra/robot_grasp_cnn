import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from shapely.geometry import Polygon

from model import AC_GRConvNet
from dataset import GraspDataset
from predict import post_process_output

# --- Configuration ---
MODEL_PATH = './outputs/models/ac_grconvnet_best.pth'
DATA_DIR = './data'
EVAL_OUTPUT_DIR = './outputs/evaluation'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.1 # Should be the same as in train.py

# Evaluation Criteria
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30.0
ANGLE_THRESHOLD_RAD = np.deg2rad(ANGLE_THRESHOLD_DEG)

os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

def grasp_to_polygon(x, y, angle, width, height=20):
    """
    Converts grasp parameters to a shapely Polygon object.
    """
    w = width / 2
    h = height / 2
    
    points = np.array([
        [-h, -w], [h, -w], [h, w], [-h, w]
    ])
    
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    rotated_points = (rot_matrix @ points.T).T
    translated_points = rotated_points + np.array([x, y])
    
    return Polygon(translated_points)

def calculate_iou(poly1, poly2):
    """
    Calculates the Intersection over Union (IoU) of two shapely Polygons.
    """
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def check_grasp_correctness(pred_grasp, gt_grasps):
    """
    Checks if a predicted grasp is correct against a list of ground-truth grasps.
    A grasp is correct if IoU > 0.25 and angle difference < 30 degrees
    with ANY ground-truth grasp.
    
    Args:
        pred_grasp (tuple): (x, y, angle, width) of the predicted grasp.
        gt_grasps (list of np.ndarray): List of ground-truth grasp rectangles.

    Returns:
        tuple: (is_correct, best_iou, angle_difference)
    """
    px, py, p_angle, p_width = pred_grasp
    pred_poly = grasp_to_polygon(px, py, p_angle, p_width)

    for gt_rect in gt_grasps:
        gt_poly = Polygon(gt_rect)
        
        # Calculate ground-truth angle
        gt_center = gt_poly.centroid.coords[0]
        edge1 = gt_rect[1] - gt_rect[0]
        gt_angle = np.arctan2(edge1[1], edge1[0])

        # Calculate IoU and Angle Difference
        iou = calculate_iou(pred_poly, gt_poly)
        angle_diff = abs((p_angle - gt_angle + np.pi/2) % np.pi - np.pi/2)

        if iou > IOU_THRESHOLD and angle_diff < ANGLE_THRESHOLD_RAD:
            return True, iou, np.rad2deg(angle_diff)
            
    return False, 0.0, 90.0 # Default values if no match is found

def plot_results(accuracy, iou_scores, angle_errors):
    """
    Plots and saves the evaluation results.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Grasp Prediction Evaluation Results', fontsize=16)

    # Accuracy Bar Chart
    axs[0].bar(['Accuracy'], [accuracy * 100], color='skyblue')
    axs[0].set_ylabel('Percentage (%)')
    axs[0].set_title('Overall Grasp Accuracy')
    axs[0].set_ylim(0, 100)
    axs[0].text(0, accuracy * 100 + 2, f'{accuracy*100:.2f}%', ha='center', va='bottom', fontsize=12)

    # IoU Distribution
    axs[1].hist(iou_scores, bins=20, color='lightgreen', range=(IOU_THRESHOLD, 1.0))
    axs[1].set_xlabel('Intersection over Union (IoU)')
    axs[1].set_ylabel('Count')
    axs[1].set_title('IoU Distribution for Correct Grasps')
    axs[1].axvline(np.mean(iou_scores), color='r', linestyle='--', label=f'Mean: {np.mean(iou_scores):.2f}')
    axs[1].legend()

    # Angle Error Distribution
    axs[2].hist(angle_errors, bins=20, color='salmon', range=(0, ANGLE_THRESHOLD_DEG))
    axs[2].set_xlabel('Angle Error (Degrees)')
    axs[2].set_ylabel('Count')
    axs[2].set_title('Angle Error for Correct Grasps')
    axs[2].axvline(np.mean(angle_errors), color='b', linestyle='--', label=f'Mean: {np.mean(angle_errors):.2f}°')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(EVAL_OUTPUT_DIR, 'evaluation_summary.png')
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Evaluation plots saved to {save_path}")

def main():
    print(f"Using device: {DEVICE}")
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        return
        
    model = AC_GRConvNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Dataset
    # We need the original ground truth rectangles, so we can't use the map-generating dataset directly.
    # We will adapt the dataset loader logic for evaluation.
    full_dataset = GraspDataset(DATA_DIR, augment=False)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Evaluating on {len(val_dataset)} samples...")

    # Evaluation Loop
    correct_grasps = 0
    total_grasps = 0
    iou_results = []
    angle_error_results = []

    with torch.no_grad():
        for rgbd_tensor, gt_maps in tqdm(val_loader, desc="Evaluating"):
            if rgbd_tensor is None:
                continue
            
            # Get model prediction
            pred_maps = model(rgbd_tensor.to(DEVICE))
            pred_maps_np = pred_maps.squeeze().cpu().numpy()
            q_map, cos_map, sin_map, width_map = np.split(pred_maps_np, 4)
            q_map, cos_map, sin_map, width_map = [m.squeeze() for m in [q_map, cos_map, sin_map, width_map]]
            
            # Post-process to get the best predicted grasp
            pred_grasp = post_process_output(q_map, cos_map, sin_map, width_map)

            # Get original ground truth rectangles (this part is tricky, need to re-load)
            # For simplicity, we'll assume the dataloader can give us the raw grasps.
            # Let's modify the dataset to also return raw grasps for evaluation.
            # NOTE: This requires a small modification in dataset.py. For now, we assume it's there.
            # A proper implementation would have a separate dataset class for evaluation.
            
            # We need to get the original grasp rectangles associated with this sample.
            # The current dataloader doesn't return them.
            # A simple workaround is to re-load them using the file path, but this is inefficient.
            # The BEST way is to adjust the dataset to return them. Let's assume this for now.
            # We'll get the index from the val_dataset to find the file.
            sample_idx = val_loader.dataset.indices[total_grasps]
            grasp_file = full_dataset.grasp_files[sample_idx]
            gt_rects = full_dataset._load_grasp_rectangles(grasp_file)
            
            # Scale ground truth to match the network's output size
            original_size = (640, 480) # Cornell dataset default
            output_size = (224, 224)
            scale_x = output_size[1] / original_size[0]
            scale_y = output_size[0] / original_size[1]
            for i in range(len(gt_rects)):
                gt_rects[i][:, 0] *= scale_x
                gt_rects[i][:, 1] *= scale_y

            is_correct, iou, angle_err = check_grasp_correctness(pred_grasp, gt_rects)
            
            if is_correct:
                correct_grasps += 1
                iou_results.append(iou)
                angle_error_results.append(angle_err)
            
            total_grasps += 1

    # Calculate final accuracy
    accuracy = correct_grasps / total_grasps if total_grasps > 0 else 0.0
    print("\n--- Evaluation Complete ---")
    print(f"Total Grasps Evaluated: {total_grasps}")
    print(f"Correct Grasps: {correct_grasps}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    if iou_results:
        print(f"Average IoU for Correct Grasps: {np.mean(iou_results):.3f}")
        print(f"Average Angle Error for Correct Grasps: {np.mean(angle_error_results):.2f}°")
    
    # Plot results
    plot_results(accuracy, iou_results, angle_error_results)

if __name__ == '__main__':
    main()
