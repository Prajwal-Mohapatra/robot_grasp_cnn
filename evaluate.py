# ===================== evaluate.py =====================
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from shapely.geometry import Polygon
import math

# Import from your existing project files
from loader import CornellGraspDataset
from model import GraspCNN
from visualize import show_rgb_depth_grasps # For saving qualitative examples

# --- Configuration ---
MODEL_PATH = "outputs/saved_models/grasp_cnn_final_v2.pth" # Use the final, fine-tuned model
DATA_ROOT = "./data/cornell-grasp"
OUTPUT_DIR = "outputs/evaluation_results"
BATCH_SIZE = 16 
IOU_THRESHOLD = 0.25 
ANGLE_THRESHOLD_DEG = 30 

# --- FIXED: Updated custom collate function ---
# This now correctly handles the data format from the latest loader.py
def custom_collate(batch):
    batch = [item for item in batch if item is not None and item['pos_grasps'].shape[0] > 0]
    if not batch:
        return None
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'pos_grasps': [item['pos_grasps'] for item in batch]
    }

# --- FIXED: Updated Helper Functions for sin/cos angle representation ---

def rect_from_grasp_param(pred_params):
    """
    Converts a 6-element prediction vector [x, y, sin(2θ), cos(2θ), w, h] 
    into a 4x2 rectangle of corner points.
    """
    if isinstance(pred_params, torch.Tensor):
        pred_params = pred_params.detach().cpu().numpy()
    
    x, y, sin2a, cos2a, w, h = pred_params
    
    # Denormalize
    x, y, w, h = x * 224, y * 224, w * 224, h * 224
    
    # Recover the angle from sin(2a) and cos(2a)
    angle = np.arctan2(sin2a, cos2a) / 2.0
    
    # Calculate corner points
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    p1 = [x - w/2 * cos_a + h/2 * sin_a, y - w/2 * sin_a - h/2 * cos_a]
    p2 = [x + w/2 * cos_a + h/2 * sin_a, y + w/2 * sin_a - h/2 * cos_a]
    p3 = [x + w/2 * cos_a - h/2 * sin_a, y + w/2 * sin_a + h/2 * cos_a]
    p4 = [x - w/2 * cos_a - h/2 * sin_a, y - w/2 * sin_a + h/2 * cos_a]
    
    return np.array([p1, p2, p3, p4])

def get_target_vector(rect):
    """Converts a 4x2 grasp rectangle into a 6-element target vector."""
    center = rect.mean(axis=0)
    dx = rect[1] - rect[0]
    dy = rect[2] - rect[1]
    
    width = np.linalg.norm(dx)
    height = np.linalg.norm(dy)
    angle = np.arctan2(dx[1], dx[0])

    return np.array([
        center[0] / 224.0, center[1] / 224.0,
        np.sin(2 * angle), np.cos(2 * angle),
        width / 224.0, height / 224.0
    ])

def calculate_iou(rect1, rect2):
    """Calculates Intersection over Union (IoU) for two 4x2 rectangles."""
    poly1 = Polygon(rect1)
    poly2 = Polygon(rect2)
    
    if not poly1.is_valid or not poly2.is_valid: return 0.0
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    if union_area == 0: return 0.0
    return intersection_area / union_area

# --- Main Evaluation Script ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GraspCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    val_dataset = CornellGraspDataset(root=DATA_ROOT, split='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    print(f"✅ Validation dataset loaded with {len(val_dataset)} samples.")

    print("\nStarting evaluation...")
    all_gt_vectors, all_pred_vectors, all_ious = [], [], []
    successful_grasps, total_grasps = 0, 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None: continue

            rgb, depth = batch['rgb'].to(device), batch['depth'].to(device)
            gt_rects_batch = batch['pos_grasps']
            
            preds = model(rgb, depth).cpu().numpy()

            for j in range(len(preds)):
                if gt_rects_batch[j].shape[0] == 0: continue
                
                # Evaluate against the first ground truth grasp for simplicity
                gt_rect = gt_rects_batch[j][0].numpy()
                pred_params = preds[j]
                
                pred_rect = rect_from_grasp_param(pred_params)
                iou = calculate_iou(gt_rect, pred_rect)
                all_ious.append(iou)
                
                gt_vector = get_target_vector(gt_rect)
                all_gt_vectors.append(gt_vector)
                all_pred_vectors.append(pred_params)
                
                # Calculate angle difference correctly
                gt_angle = np.arctan2(gt_vector[2], gt_vector[3])
                pred_angle = np.arctan2(pred_params[2], pred_params[3])
                angle_diff = abs((gt_angle - pred_angle + np.pi) % (2 * np.pi) - np.pi)
                
                if iou >= IOU_THRESHOLD and np.rad2deg(angle_diff) <= ANGLE_THRESHOLD_DEG:
                    successful_grasps += 1
                total_grasps += 1
            
            print(f"  Processed batch {i+1}/{len(val_loader)}...")

    print("✅ Evaluation complete.")

    # --- Metrics Calculation & Reporting ---
    all_gt_vectors = np.array(all_gt_vectors)
    all_pred_vectors = np.array(all_pred_vectors)
    
    mse = np.mean((all_gt_vectors - all_pred_vectors)**2)
    mae = np.mean(np.abs(all_gt_vectors - all_pred_vectors))
    success_rate = (successful_grasps / total_grasps) * 100 if total_grasps > 0 else 0
    avg_iou = np.mean(all_ious)

    report = f"""
    ==================================================
              GRASP EVALUATION REPORT
    ==================================================
    
    Metrics based on {total_grasps} validation samples.
    
    Grasp Success Rate (IoU > {IOU_THRESHOLD} & Angle < {ANGLE_THRESHOLD_DEG}°): {success_rate:.2f}%
    Average Intersection over Union (IoU): {avg_iou:.4f}
    
    --- Parameter Prediction Error ---
    Mean Squared Error (MSE): {mse:.6f}
    Mean Absolute Error (MAE): {mae:.6f}
    
    ==================================================
    """
    print(report)
    with open(os.path.join(OUTPUT_DIR, "evaluation_report.txt"), "w") as f:
        f.write(report)
    print(f"✅ Report saved to {os.path.join(OUTPUT_DIR, 'evaluation_report.txt')}")

    # --- Plotting ---
    print("Generating plots...")

    # IoU Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_ious, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Intersection over Union (IoU) Scores')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.axvline(IOU_THRESHOLD, color='r', linestyle='--', label=f'Success Threshold ({IOU_THRESHOLD})')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(os.path.join(OUTPUT_DIR, "iou_distribution.png"))
    plt.close()

    # Predicted vs. Ground Truth Scatter Plots
    param_names = ['Center X', 'Center Y', 'sin(2θ)', 'cos(2θ)', 'Width', 'Height']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Predicted vs. Ground Truth Grasp Parameters', fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.scatter(all_gt_vectors[:, i], all_pred_vectors[:, i], alpha=0.3)
        ax.plot([-1, 1], [-1, 1], 'r--') # Perfect prediction line
        ax.set_title(param_names[i])
        ax.set_xlabel('Ground Truth (Normalized)')
        ax.set_ylabel('Prediction (Normalized)')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_gt_scatter.png"))
    plt.close()

    print(f"✅ All plots saved to {OUTPUT_DIR}")

