import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from shapely.geometry import Polygon
from tqdm import tqdm
import torch.nn.functional as F

from loader import CornellGraspDataset
from model import GraspCNN

def get_best_grasp(quality_map, angle_map, width_map):
    quality_map = quality_map.squeeze()
    angle_map = angle_map.squeeze()
    width_map = width_map.squeeze()

    # Find the pixel with the highest quality score
    coords = np.unravel_index(torch.argmax(quality_map).cpu().numpy(), quality_map.shape)
    y, x = coords

    # Get angle and width at that pixel
    sin2a, cos2a = angle_map[0, y, x], angle_map[1, y, x]
    angle = np.arctan2(sin2a, cos2a) / 2.0
    width = width_map[y, x] * 300 # Denormalize width

    return [x, y, angle, width]

def grasp_to_rect(grasp_params):
    x, y, angle, width = grasp_params
    h = width / 2 # Assume a fixed aspect ratio for the gripper
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    p1 = [x - width/2 * cos_a + h/2 * sin_a, y - width/2 * sin_a - h/2 * cos_a]
    p2 = [x + width/2 * cos_a + h/2 * sin_a, y + width/2 * sin_a - h/2 * cos_a]
    p3 = [x + width/2 * cos_a - h/2 * sin_a, y + width/2 * sin_a + h/2 * cos_a]
    p4 = [x - width/2 * cos_a - h/2 * sin_a, y - width/2 * sin_a + h/2 * cos_a]
    
    return np.array([p1, p2, p3, p4])

def calculate_iou(rect1, rect2_list):
    poly1 = Polygon(rect1)
    if not poly1.is_valid: return 0.0

    best_iou = 0
    for rect2 in rect2_list:
        poly2 = Polygon(rect2)
        if not poly2.is_valid: continue
        
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - intersection_area
        if union_area == 0: continue
        
        iou = intersection_area / union_area
        if iou > best_iou:
            best_iou = iou
            
    return best_iou

def evaluate_model():
    MODEL_PATH = "outputs/saved_models/grasp_cnn_final_v5.pth"
    OUTPUT_DIR = "outputs/evaluation_results_v5"
    IOU_THRESHOLD = 0.25

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GraspCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    val_dataset = CornellGraspDataset(split='val')
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    print("\nStarting evaluation...")
    all_ious = []
    successful_grasps = 0
    total_grasps = len(val_loader)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            rgb, depth = batch['rgb'].to(device), batch['depth'].to(device)
            gt_bbs = batch['gt_bbs']
            
            pred_quality, pred_angle, pred_width = model(rgb, depth)
            
            # Resize predictions to match input size
            pred_quality = F.interpolate(pred_quality, size=val_dataset.target_size, mode='bilinear', align_corners=True)
            pred_angle = F.interpolate(pred_angle, size=val_dataset.target_size, mode='bilinear', align_corners=True)
            pred_width = F.interpolate(pred_width, size=val_dataset.target_size, mode='bilinear', align_corners=True)
            
            best_grasp = get_best_grasp(pred_quality, pred_angle, pred_width)
            pred_rect = grasp_to_rect(best_grasp)
            
            iou = calculate_iou(pred_rect, gt_bbs)
            all_ious.append(iou)
            
            if iou >= IOU_THRESHOLD:
                successful_grasps += 1

    success_rate = (successful_grasps / total_grasps) * 100
    avg_iou = np.mean(all_ious)

    report = f"""
    ==================================================
              GRASP EVALUATION REPORT (v5)
    ==================================================
    Metrics based on {total_grasps} validation samples.
    Success Rate (IoU > {IOU_THRESHOLD}): {success_rate:.2f}%
    Average Intersection over Union (IoU): {avg_iou:.4f}
    ==================================================
    """
    print(report)
    with open(os.path.join(OUTPUT_DIR, "evaluation_report.txt"), "w") as f: f.write(report)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_ious, bins=50, color='skyblue', edgecolor='black', range=(0, 1))
    plt.title('Distribution of Best Intersection over Union (IoU) Scores')
    plt.xlabel('IoU Score'); plt.ylabel('Frequency')
    plt.axvline(IOU_THRESHOLD, color='r', linestyle='--', label=f'Success Threshold ({IOU_THRESHOLD})')
    plt.legend(); plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "iou_distribution.png"))
    plt.close()
    print(f"✅ Report and plots saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    evaluate_model()

