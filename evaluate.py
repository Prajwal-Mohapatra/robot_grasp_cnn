# ===================== evaluate.py =====================
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from shapely.geometry import Polygon
from tqdm import tqdm

from loader import CornellGraspDataset
from model import GraspCNN

# --- Configuration ---
MODEL_PATH = "outputs/saved_models/grasp_cnn_final_v3.pth" # Path to the trained model
OUTPUT_DIR = "outputs/evaluation_results_v3"
BATCH_SIZE = 32
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30

# --- Helper Functions ---

def custom_collate(batch):
    """Custom collate function to filter out samples with no valid grasps."""
    batch = [item for item in batch if item is not None and item['pos_grasps'].shape[0] > 0]
    if not batch: return None
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'pos_grasps': [item['pos_grasps'] for item in batch]
    }

def rect_from_grasp_param(pred_params, image_size):
    """
    Converts a 6-element prediction vector [x, y, sin(2θ), cos(2θ), w, h]
    into a 4x2 rectangle of corner points, scaled to the image size.
    """
    if isinstance(pred_params, torch.Tensor):
        pred_params = pred_params.detach().cpu().numpy()
    
    x_norm, y_norm, sin2a, cos2a, w_norm, h_norm = pred_params
    
    # Denormalize using the provided image size (CRITICAL FIX)
    x, y, w, h = x_norm * image_size, y_norm * image_size, w_norm * image_size, h_norm * image_size
    
    angle = np.arctan2(sin2a, cos2a) / 2.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # Calculate corner points of the rectangle
    p1 = [x - w/2 * cos_a + h/2 * sin_a, y - w/2 * sin_a - h/2 * cos_a]
    p2 = [x + w/2 * cos_a + h/2 * sin_a, y + w/2 * sin_a - h/2 * cos_a]
    p3 = [x + w/2 * cos_a - h/2 * sin_a, y + w/2 * sin_a + h/2 * cos_a]
    p4 = [x - w/2 * cos_a - h/2 * sin_a, y - w/2 * sin_a + h/2 * cos_a]
    
    return np.array([p1, p2, p3, p4])

def get_angle_from_rect(rect):
    """Extracts the angle from a 4x2 rectangle."""
    dx = rect[1, 0] - rect[0, 0]
    dy = rect[1, 1] - rect[0, 1]
    return np.arctan2(dy, dx)

def calculate_iou(rect1, rect2):
    """Calculates Intersection over Union (IoU) for two 4x2 rectangles."""
    poly1 = Polygon(rect1)
    poly2 = Polygon(rect2)
    
    if not poly1.is_valid or not poly2.is_valid: return 0.0
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    if union_area == 0: return 0.0
    return intersection_area / union_area

def evaluate_model():
    """Main function to run the full evaluation process."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model = GraspCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    # --- Load Dataset ---
    val_dataset = CornellGraspDataset(split='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    image_size = val_dataset.target_size[0] # Get image size from dataset (CRITICAL FIX)
    print(f"✅ Validation dataset loaded with {len(val_dataset)} samples. Image size: {image_size}x{image_size}")

    # --- Evaluation Loop ---
    print("\nStarting evaluation...")
    all_ious = []
    successful_grasps, total_grasps = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if batch is None: continue

            rgb, depth = batch['rgb'].to(device), batch['depth'].to(device)
            gt_rects_batch = batch['pos_grasps']
            preds = model(rgb, depth)

            for j in range(preds.shape[0]):
                pred_params = preds[j]
                gt_rects = gt_rects_batch[j].numpy()
                
                if gt_rects.shape[0] == 0: continue
                total_grasps += 1

                # Convert predicted parameters to a rectangle
                pred_rect = rect_from_grasp_param(pred_params, image_size)
                
                # --- Improved Evaluation: Find best GT match for the prediction ---
                best_iou = 0
                best_gt_angle = 0
                for gt_rect in gt_rects:
                    iou = calculate_iou(gt_rect, pred_rect)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_angle = get_angle_from_rect(gt_rect)
                
                all_ious.append(best_iou)
                
                # Check for success
                pred_angle = get_angle_from_rect(pred_rect)
                angle_diff_rad = abs(best_gt_angle - pred_angle)
                angle_diff_deg = np.rad2deg(min(angle_diff_rad, 2 * np.pi - angle_diff_rad))

                if best_iou >= IOU_THRESHOLD and angle_diff_deg <= ANGLE_THRESHOLD_DEG:
                    successful_grasps += 1

    print("✅ Evaluation complete.")

    # --- Metrics Calculation & Reporting ---
    success_rate = (successful_grasps / total_grasps) * 100 if total_grasps > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0

    report = f"""
    ==================================================
              GRASP EVALUATION REPORT (v3)
    ==================================================
    
    Metrics based on {total_grasps} validation samples.
    
    Grasp Success Rate (IoU > {IOU_THRESHOLD} & Angle < {ANGLE_THRESHOLD_DEG}°): {success_rate:.2f}%
    Average Intersection over Union (IoU): {avg_iou:.4f}
    
    ==================================================
    """
    print(report)
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Report saved to {report_path}")

    # --- Plotting ---
    print("Generating plots...")
    
    # IoU Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_ious, bins=50, color='skyblue', edgecolor='black', range=(0, 1))
    plt.title('Distribution of Best Intersection over Union (IoU) Scores')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.axvline(IOU_THRESHOLD, color='r', linestyle='--', label=f'Success Threshold ({IOU_THRESHOLD})')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "iou_distribution.png"))
    plt.close()

    print(f"✅ All plots saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    evaluate_model()
