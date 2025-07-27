# ===================== grasp_visualization.py =====================
import torch
import random
from loader import CornellGraspDataset 
from model import GraspCNN
from visualize import show_rgb_depth_grasps
import os

def visualize_grasps():
    MODEL_PATH = "outputs/saved_models/grasp_cnn_final_v2.pth"
    OUTPUT_DIR = "outputs/final_visualizations_v2"
    N_SAMPLES = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraspCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully from {MODEL_PATH}")

    val_dataset = CornellGraspDataset(split='val')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Will save {N_SAMPLES} visualization(s) to '{OUTPUT_DIR}/'")

    for i in range(N_SAMPLES):
        idx = random.randint(0, len(val_dataset) - 1)
        sample = val_dataset[idx]
        
        rgb_tensor = sample['rgb'].unsqueeze(0).to(device)
        depth_tensor = sample['depth'].unsqueeze(0).to(device)
        
        print(f"\nProcessing sample {i+1}/{N_SAMPLES} (index: {idx})...")

        with torch.no_grad():
            pred = model(rgb_tensor, depth_tensor)
            
        save_path = os.path.join(OUTPUT_DIR, f"grasp_visualization_{i+1:03d}.png")
        
        show_rgb_depth_grasps(
            rgb=sample['rgb'], 
            depth=sample['depth'], 
            gt_grasps=sample['pos_grasps'],
            pred_params=pred[0],
            image_size=val_dataset.target_size[0]
        )
        print(f"✅ Saved visualization to: {save_path}")

if __name__ == '__main__':
    visualize_grasps()
