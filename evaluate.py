import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from shapely.geometry import Polygon
from PIL import Image
from scipy.ndimage import maximum_filter # --- FIX 5 (from PDF): Import for Top-K ---

from model import AC_GRConvNet
from dataset import GraspDataset
# post_process_output is still used as a fallback
from predict import post_process_output

# --- Configuration ---
EVAL_OUTPUT_DIR = './outputs/evaluation'
DATA_DIR = './data'
# --- FIX: Re-typing this line to remove any hidden characters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- End of Fix ---

# --- Paths for the two models to evaluate ---
MODEL_PATH_MAIN = './outputs/models/ac_grconvnet_main_best.pth'
MODEL_PATH_FINETUNE = './outputs/models/ac_grconvnet_finetune_best.pth'

# --- Path to load data splits ---
SPLIT_DIR = './outputs/splits'
TEST_INDICES_PATH = os.path.join(SPLIT_DIR, 'test_indices.npy')

# Evaluation Criteria
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30.0
ANGLE_THRESHOLD_RAD = np.deg2rad(ANGLE_THRESHOLD_DEG)
EVAL_TOP_K = 5 # Evaluate the top 5 grasp predictions

os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

def grasp_to_polygon(x, y, angle, width, height=20):
    """
    Converts grasp parameters to a shapely Polygon object.
    """
    w = width / 2
    h = height / 2
    points = np.array([[-h, -w], [h, -w], [h, w], [-h, w]])
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
    try:
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        if union_area == 0:
            return 0.0
        return intersection_area / union_area
    except Exception:
        # Log invalid polygons if needed
        return 0.0

def get_gt_grasp_params(gt_rect):
    """
    Extracts center, angle, and width from a ground-truth rectangle.
    """
    gt_poly = Polygon(gt_rect)
    gt_center = gt_poly.centroid.coords[0]
    edge1 = gt_rect[1] - gt_rect[0]
    edge2 = gt_rect[3] - gt_rect[0]
    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        gt_angle = np.arctan2(edge1[1], edge1[0])
        gt_width = np.linalg.norm(edge2)
    else:
        gt_angle = np.arctan2(edge2[1], edge2[0])
        gt_width = np.linalg.norm(edge1)
    
    # --- FIX 6 (from PDF): Normalize angle to [0, pi) ---
    gt_angle = gt_angle % np.pi
    # --- End of Fix ---
    
    return gt_center[0], gt_center[1], gt_angle, gt_width

def check_grasp_correctness(pred_grasp, gt_grasps):
    """
    Checks if a predicted grasp is correct against a list of ground-truth grasps.
    """
    px, py, p_angle, p_width = pred_grasp
    
    # --- FIX 6 (from PDF): Normalize predicted angle for comparison ---
    # p_angle is from arctan2/2, range [-pi/2, pi/2]
    # Normalize to [0, pi) to match GT
    p_angle_norm = p_angle % np.pi
    # --- End of Fix ---
    
    # Use original p_angle for polygon creation
    pred_poly = grasp_to_polygon(px, py, p_angle, p_width, height=p_width/2)

    best_iou = 0.0
    best_angle_diff = np.pi

    for gt_rect in gt_grasps:
        gt_poly = Polygon(gt_rect)
        if not gt_poly.is_valid:
            continue
            
        gt_x, gt_y, gt_angle, gt_width = get_gt_grasp_params(gt_rect) # gt_angle is [0, pi)
        
        iou = calculate_iou(pred_poly, gt_poly)
        
        # --- FIX 6 (from PDF): Robust angle difference calculation ---
        # Compare two angles in [0, pi)
        d = abs(p_angle_norm - gt_angle)
        angle_diff = min(d, np.pi - d)
        # --- End of Fix ---
        
        if iou > best_iou:
            best_iou = iou
            best_angle_diff = angle_diff

        if iou > IOU_THRESHOLD and angle_diff < ANGLE_THRESHOLD_RAD:
            return True, iou, np.rad2deg(angle_diff)
            
    return False, best_iou, np.rad2deg(best_angle_diff)

def plot_results(accuracy, iou_scores, angle_errors, model_name):
    """
    Plots and saves the evaluation results for a specific model.
    """
    correct_iou_scores = [iou for iou, err in zip(iou_scores, angle_errors) 
                          if iou > IOU_THRESHOLD and err < ANGLE_THRESHOLD_DEG]
    correct_angle_errors = [err for iou, err in zip(iou_scores, angle_errors) 
                            if iou > IOU_THRESHOLD and err < ANGLE_THRESHOLD_DEG]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Grasp Evaluation Results - {model_name}', fontsize=16)

    # Accuracy Bar Chart
    axs[0].bar(['Accuracy'], [accuracy * 100], color='skyblue', width=0.5)
    axs[0].set_ylabel('Percentage (%)')
    axs[0].set_title(f'Overall Grasp Accuracy (Top-{EVAL_TOP_K})')
    axs[0].set_ylim(0, 100)
    axs[0].text(0, accuracy * 100, f'{accuracy*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # IoU Distribution
    if correct_iou_scores:
        axs[1].hist(correct_iou_scores, bins=20, color='lightgreen', range=(IOU_THRESHOLD, 1.0))
        mean_iou = np.mean(correct_iou_scores)
        axs[1].axvline(mean_iou, color='r', linestyle='--', label=f'Mean: {mean_iou:.2f}')
        axs[1].legend()
    axs[1].set_xlabel('Intersection over Union (IoU)')
    axs[1].set_ylabel('Count')
    axs[1].set_title(f'IoU Distribution (Correct Grasps > {IOU_THRESHOLD})')

    # Angle Error Distribution
    if correct_angle_errors:
        axs[2].hist(correct_angle_errors, bins=20, color='salmon', range=(0, ANGLE_THRESHOLD_DEG))
        mean_angle_err = np.mean(correct_angle_errors)
        axs[2].axvline(mean_angle_err, color='b', linestyle='--', label=f'Mean: {mean_angle_err:.2f}°')
        axs[2].legend()
    axs[2].set_xlabel('Angle Error (Degrees)')
    axs[2].set_ylabel('Count')
    axs[2].set_title(f'Angle Error (Correct Grasps < {ANGLE_THRESHOLD_DEG}°)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save with a model-specific name
    save_filename = f'evaluation_summary_{model_name.lower().replace(" ", "_")}.png'
    save_path = os.path.join(EVAL_OUTPUT_DIR, save_filename)
    plt.savefig(save_path)
    # plt.show()
    print(f"✅ Evaluation plots for {model_name} saved to {save_path}")

def run_evaluation(model_path, model_name, test_loader, full_dataset, test_indices):
    """
    Runs the full evaluation loop for a given model.
    """
    print(f"\n--- Evaluating Model: {model_name} ---")
    print(f"Loading model from: {model_path}")

    # Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'. Skipping evaluation.")
        return

    model = AC_GRConvNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Evaluation Loop
    correct_grasps = 0
    total_grasps_processed = 0
    all_iou_scores = []
    all_angle_errors = []
    MAX_GRASP_WIDTH = 150.0 # From data_processing

    with torch.no_grad():
        for i, (rgbd_tensor, gt_maps) in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
            if rgbd_tensor is None or not rgbd_tensor.numel():
                continue
            
            # Get model prediction
            pred_maps = model(rgbd_tensor.to(DEVICE))
            pred_maps_np = pred_maps.squeeze().cpu().numpy()
            q_map, cos_map, sin_map, width_map = [m.squeeze() for m in np.split(pred_maps_np, 4)]
            
            # Get original ground truth rectangles
            sample_idx = test_indices[i]
            grasp_file = full_dataset.grasp_files[sample_idx]
            gt_rects_orig = full_dataset._load_grasp_rectangles(grasp_file)
            
            if not gt_rects_orig:
                continue
                
            # Scale ground truth
            # --- FIX 4 (from PDF): Fix GT scaling fallback ---
            rgb_path = grasp_file.replace('cpos.txt', 'r.png')
            if not os.path.exists(rgb_path):
                print(f"! WARNING: RGB file missing, skipping sample: {rgb_path}")
                continue # Skip this sample instead of using wrong default size
            
            with Image.open(rgb_path) as img:
                original_size = img.size
            # --- End of Fix ---
            
            output_size = (224, 224)
            scale_x = output_size[1] / original_size[0]
            scale_y = output_size[0] / original_size[1]
            
            gt_rects_scaled = []
            for rect in gt_rects_orig:
                scaled_rect = np.copy(rect)
                scaled_rect[:, 0] *= scale_x
                scaled_rect[:, 1] *= scale_y
                gt_rects_scaled.append(scaled_rect)

            # --- FIX 5 (from PDF): Implement Top-K Evaluation ---
            
            # 1. Find local maxima
            # Use a filter size appropriate for the output resolution
            local_maxima_mask = (q_map == maximum_filter(q_map, size=11))
            
            # 2. Get coordinates and values of these maxima
            maxima_coords = np.argwhere(local_maxima_mask)
            maxima_values = q_map[local_maxima_mask]
            
            # 3. Sort by quality and take top K
            # [::-1] reverses to get highest first
            top_k_indices = np.argsort(maxima_values)[-EVAL_TOP_K:][::-1]
            top_k_coords = maxima_coords[top_k_indices]
            
            best_iou = 0.0
            best_angle_err = 180.0
            is_correct_for_sample = False

            if len(top_k_coords) == 0:
                # Fallback if no local maxima are found (unlikely)
                pred_grasp = post_process_output(q_map, cos_map, sin_map, width_map)
                is_correct_for_sample, best_iou, best_angle_err = check_grasp_correctness(pred_grasp, gt_rects_scaled)
            else:
                # 4. Test each of the Top-K grasps
                for y, x in top_k_coords:
                    # Get grasp parameters at this coord
                    cos_val = cos_map[y, x]
                    sin_val = sin_map[y, x]
                    width_val = width_map[y, x]
                    angle = np.arctan2(sin_val, cos_val) / 2.0
                    width_pixels = width_val * MAX_GRASP_WIDTH
                    
                    pred_grasp = (x, y, angle, width_pixels)
                    
                    is_correct, iou, angle_err = check_grasp_correctness(pred_grasp, gt_rects_scaled)
                    
                    # Keep track of the best IOU/angle error from the Top-K
                    if iou > best_iou:
                        best_iou = iou
                        best_angle_err = angle_err
                    
                    # If any of the top-k is correct, mark the sample as correct and stop
                    if is_correct:
                        is_correct_for_sample = True
                        best_iou = iou
                        best_angle_err = angle_err
                        break # Found a correct grasp, move to next sample

            if is_correct_for_sample:
                correct_grasps += 1
            
            all_iou_scores.append(best_iou)
            all_angle_errors.append(best_angle_err)
            total_grasps_processed += 1
            # --- End of Fix ---
            
    # Calculate and print final accuracy
    accuracy = correct_grasps / total_grasps_processed if total_grasps_processed > 0 else 0.0
    print(f"\n--- Evaluation Complete for {model_name} ---")
    print(f"Total Grasps Evaluated: {total_grasps_processed}")
    print(f"Correct Grasps (Top-{EVAL_TOP_K}): {correct_grasps}")
    print(f"Accuracy: {accuracy * 100:.2f}% (IoU > {IOU_THRESHOLD} & Angle Error < {ANGLE_THRESHOLD_DEG}°)")
    
    if all_iou_scores:
        correct_iou_scores = [iou for iou, err in zip(all_iou_scores, all_angle_errors) 
                              if iou > IOU_THRESHOLD and err < ANGLE_THRESHOLD_DEG]
        correct_angle_errors = [err for iou, err in zip(all_iou_scores, all_angle_errors) 
                                if iou > IOU_THRESHOLD and err < ANGLE_THRESHOLD_DEG]
        
        if correct_iou_scores:
            print(f"Average IoU (for correct grasps): {np.mean(correct_iou_scores):.3f}")
        if correct_angle_errors:
            print(f"Average Angle Error (for correct grasps): {np.mean(correct_angle_errors):.2f}°")
    
    # Plot results
    plot_results(accuracy, all_iou_scores, all_angle_errors, model_name)


def main():
    print(f"Using device: {DEVICE}")

    # Load Dataset and Test Split Indices (Done once)
    if not os.path.exists(TEST_INDICES_PATH):
        print(f"Error: Test split file not found at '{TEST_INDICES_PATH}'.")
        print("Please run train.py first to generate data splits.")
        return
        
    try:
        # augment=False is correct for evaluation
        full_dataset = GraspDataset(DATA_DIR, augment=False)
        if len(full_dataset) == 0:
            print("Dataset is empty. Please check the data directory.")
            return
            
        test_indices = np.load(TEST_INDICES_PATH)
        test_dataset = Subset(full_dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"Loaded {len(test_dataset)} test samples.")
    except FileNotFoundError as e:
        print(e)
        return

    # --- Run Evaluation for both models ---
    
    # 1. Main Model
    run_evaluation(
        model_path=MODEL_PATH_MAIN,
        model_name="Main Model",
        test_loader=test_loader,
        full_dataset=full_dataset,
        test_indices=test_indices
    )
    
    # 2. Fine-Tuned Model
    run_evaluation(
        model_path=MODEL_PATH_FINETUNE,
        model_name="Fine-Tuned Model",
        test_loader=test_loader,
        full_dataset=full_dataset,
        test_indices=test_indices
    )

if __name__ == '__main__':
    main()
