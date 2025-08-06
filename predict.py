import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

from loader import CornellGraspDataset
from model import GraspCNN

def get_best_grasp(quality_map, angle_map, width_map):
    # Same as in evaluate.py
    quality_map = quality_map.squeeze()
    angle_map = angle_map.squeeze()
    width_map = width_map.squeeze()
    coords = np.unravel_index(torch.argmax(quality_map).cpu().numpy(), quality_map.shape)
    y, x = coords
    sin2a, cos2a = angle_map[0, y, x], angle_map[1, y, x]
    angle = np.arctan2(sin2a, cos2a) / 2.0
    width = width_map[y, x] * 300
    return [x, y, angle, width]

def grasp_to_rect(grasp_params):
    # Same as in evaluate.py
    x, y, angle, width = grasp_params
    h = width / 2
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    p1 = [x - width/2 * cos_a + h/2 * sin_a, y - width/2 * sin_a - h/2 * cos_a]
    p2 = [x + width/2 * cos_a + h/2 * sin_a, y + width/2 * sin_a - h/2 * cos_a]
    p3 = [x + width/2 * cos_a - h/2 * sin_a, y + width/2 * sin_a + h/2 * cos_a]
    p4 = [x - width/2 * cos_a - h/2 * sin_a, y - width/2 * sin_a + h/2 * cos_a]
    return np.array([p1, p2, p3, p4])

def predict_and_visualize():
    MODEL_PATH = "outputs/saved_models/grasp_cnn_final_v5.pth"
    OUTPUT_DIR = "outputs/predictions_v5"
    N_SAMPLES = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraspCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    val_dataset = CornellGraspDataset(split='val')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Will predict {N_SAMPLES} sample(s) and save to '{OUTPUT_DIR}/'")

    with torch.no_grad():
        for i in range(N_SAMPLES):
            idx = random.randint(0, len(val_dataset) - 1)
            sample = val_dataset[idx]
            
            rgb_tensor = sample['rgb'].unsqueeze(0).to(device)
            depth_tensor = sample['depth'].unsqueeze(0).to(device)
            
            pred_quality, pred_angle, pred_width = model(rgb_tensor, depth_tensor)
            
            # Un-normalize RGB for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            rgb_display = (rgb_tensor * std + mean).clamp(0, 1).squeeze().cpu().permute(1, 2, 0).numpy()

            # Resize maps and get best grasp
            q_map = F.interpolate(pred_quality, size=val_dataset.target_size, mode='bilinear', align_corners=True).squeeze().cpu()
            best_grasp = get_best_grasp(pred_quality, pred_angle, pred_width)
            pred_rect = grasp_to_rect(best_grasp)
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(rgb_display)
            axs[0].set_title('RGB with Predicted Grasp')
            rect_closed = np.vstack([pred_rect, pred_rect[0]])
            axs[0].plot(rect_closed[:, 0], rect_closed[:, 1], 'r--', linewidth=2.5)
            axs[0].axis('off')

            axs[1].imshow(rgb_display)
            im = axs[1].imshow(q_map, cmap='viridis', alpha=0.7)
            axs[1].set_title('Grasp Quality Heatmap')
            axs[1].axis('off')
            fig.colorbar(im, ax=axs[1])
            
            save_path = os.path.join(OUTPUT_DIR, f"prediction_{i+1:02d}_idx{idx}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"✅ Saved visualization to: {save_path}")

if __name__ == '__main__':
    predict_and_visualize()
