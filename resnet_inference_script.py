# ===================== inference.py =====================
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
import os
from model import create_grasp_model, grasp_params_to_rectangle
#from dataset_loader import CornellGraspDataset
from loader import CornellGraspDataset

def load_trained_model(model_path, model_type='resnet34', device='cuda'):
    """Load trained model from checkpoint"""
    model = create_grasp_model(model_type, pretrained=False).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"   - Model type: {checkpoint.get('model_type', model_type)}")
    print(f"   - Best success rate: {checkpoint.get('best_success_rate', 'N/A')}")
    
    return model

def preprocess_image(rgb_image, depth_image, target_size=(224, 224)):
    """Preprocess RGB and depth images for inference"""
    # Resize images
    rgb_resized = cv2.resize(rgb_image, target_size)
    depth_resized = cv2.resize(depth_image, target_size)
    
    # Convert to tensors and normalize
    rgb_tensor = transforms.ToTensor()(rgb_resized).unsqueeze(0)  # [1, 3, 224, 224]
    
    # Normalize depth image
    depth_normalized = depth_resized.astype(np.float32) / 255.0
    depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]
    
    return rgb_tensor, depth_tensor

def predict_grasp(model, rgb_tensor, depth_tensor, device='cuda'):
    """Predict grasp parameters using trained model"""
    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = depth_tensor.to(device)
    
    with torch.no_grad():
        grasp_params = model(rgb_tensor, depth_tensor)
    
    return grasp_params.cpu()

def visualize_grasp_prediction(rgb_image, depth_image, grasp_params, 
                             original_size=None, save_path=None):
    """Visualize grasp prediction on original images"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Scale factor if original size is different
    if original_size is not None:
        scale_x = original_size[1] / 224.0
        scale_y = original_size[0] / 224.0
    else:
        scale_x = scale_y = 1.0
    
    # Extract parameters
    x, y, theta, width = grasp_params[0]
    
    # Scale coordinates back to original size
    x_scaled = x * scale_x
    y_scaled = y * scale_y
    width_scaled = width * scale_x
    
    # Convert to rectangle
    scaled_params = torch.tensor([[x_scaled, y_scaled, theta, width_scaled]])
    rectangle = grasp_params_to_rectangle(scaled_params, height=30 * scale_y)[0]
    
    # RGB image with grasp
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image with Predicted Grasp')
    
    # Draw grasp rectangle
    rect_points = rectangle.numpy()
    grasp_patch = patches.Polygon(rect_points, linewidth=3, edgecolor='red', 
                                facecolor='none', alpha=0.8)
    axes[0].add_patch(grasp_patch)
    
    # Add center point
    axes[0].plot(x_scaled, y_scaled, 'ro', markersize=8)
    
    # Add grasp parameters text
    axes[0].text(10, 30, f'x: {x:.1f}, y: {y:.1f}\nθ: {theta:.3f} rad\nwidth: {width:.1f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10, verticalalignment='top')
    
    # Depth image with grasp
    axes[1].imshow(depth_image, cmap='gray')
    axes[1].set_title('Depth Image with Predicted Grasp')
    
    # Draw grasp rectangle on depth image
    grasp_patch_depth = patches.Polygon(rect_points, linewidth=3, edgecolor='yellow', 
                                       facecolor='none', alpha=0.8)
    axes[1].add_patch(grasp_patch_depth)
    axes[1].plot(x_scaled, y_scaled, 'yo', markersize=8)
    
    # Combined visualization
    axes[2].imshow(rgb_image)
    axes[2].imshow(depth_image, alpha=0.3, cmap='hot')
    axes[2].set_title('Combined RGB-Depth with Grasp')
    
    grasp_patch_combined = patches.Polygon(rect_points, linewidth=3, edgecolor='lime', 
                                          facecolor='none', alpha=0.9)
    axes[2].add_patch(grasp_patch_combined)
    axes[2].plot(x_scaled, y_scaled, 'go', markersize=8)
    
    # Remove axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    
    plt.show()
    
    return fig

def batch_inference(model, dataset, num_samples=8, device='cuda'):
    """Run inference on a batch of samples from dataset"""
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        if sample is None:
            continue
            
        rgb_tensor = sample['rgb'].unsqueeze(0)
        depth_tensor = sample['depth'].unsqueeze(0)
        
        # Predict grasp
        grasp_params = predict_grasp(model, rgb_tensor, depth_tensor, device)
        
        # Convert tensors to numpy for visualization
        rgb_np = rgb_tensor.squeeze().permute(1, 2, 0).numpy()
        depth_np = depth_tensor.squeeze().numpy()
        
        # Convert to rectangle
        rectangle = grasp_params_to_rectangle(grasp_params)[0]
        
        # Display
        axes[i].imshow(rgb_np)
        
        # Draw predicted grasp
        rect_points = rectangle.numpy()
        grasp_patch = patches.Polygon(rect_points, linewidth=2, edgecolor='red', 
                                    facecolor='none', alpha=0.8)
        axes[i].add_patch(grasp_patch)
        
        # Draw ground truth grasps if available
        if 'pos_grasps' in sample and len(sample['pos_grasps']) > 0:
            for gt_grasp in sample['pos_grasps'][:3]:  # Show first 3 GT grasps
                gt_points = gt_grasp.numpy()
                gt_patch = patches.Polygon(gt_points, linewidth=1, edgecolor='green', 
                                         facecolor='none', alpha=0.6)
                axes[i].add_patch(gt_patch)
        
        axes[i].set_title(f'Sample {idx}\nRed: Predicted, Green: GT')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return fig

def evaluate_model_performance(model, dataset, num_samples=100, device='cuda'):
    """Evaluate model performance on dataset samples"""
    model.eval()
    
    from shapely.geometry import Polygon
    
    def calculate_iou(rect1, rect2):
        try:
            poly1 = Polygon(rect1)
            poly2 = Polygon(rect2)
            
            if not poly1.is_valid or not poly2.is_valid:
                return 0.0
                
            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            
            if union == 0:
                return 0.0
                
            return intersection / union
        except:
            return 0.0
    
    success_rates = []
    iou_thresholds = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    all_ious = []
    
    for idx in indices:
        sample = dataset[idx]
        if sample is None:
            continue
            
        rgb_tensor = sample['rgb'].unsqueeze(0)
        depth_tensor = sample['depth'].unsqueeze(0)
        
        # Predict grasp
        grasp_params = predict_grasp(model, rgb_tensor, depth_tensor, device)
        pred_rectangle = grasp_params_to_rectangle(grasp_params)[0].numpy()
        
        # Calculate IoU with ground truth grasps
        max_iou = 0.0
        if 'pos_grasps' in sample and len(sample['pos_grasps']) > 0:
            for gt_grasp in sample['pos_grasps']:
                gt_rectangle = gt_grasp.numpy()
                iou = calculate_iou(pred_rectangle, gt_rectangle)
                max_iou = max(max_iou, iou)
        
        all_ious.append(max_iou)
    
    # Calculate success rates at different thresholds
    for threshold in iou_thresholds:
        success_rate = np.mean([iou >= threshold for iou in all_ious])
        success_rates.append(success_rate)
        print(f"Success rate @ IoU {threshold}: {success_rate:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(iou_thresholds, success_rates, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('IoU Threshold')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs IoU Threshold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(all_ious, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('IoU with Ground Truth')
    plt.ylabel('Frequency')
    plt.title('Distribution of IoU Scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return success_rates, all_ious

def main():
    """Main inference function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    model_path = "saved_models/resnet_resnet34_grasp.pth"
    model_type = 'resnet34'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please make sure you have trained the model first.")
        return
    
    model = load_trained_model(model_path, model_type, device)
    
    # Load dataset for testing
    dataset = CornellGraspDataset(root='./data/cornell-grasp', split='test')
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Run batch inference
    print("\n📊 Running batch inference...")
    batch_inference(model, dataset, num_samples=8, device=device)
    
    # Evaluate model performance
    print("\n🔍 Evaluating model performance...")
    success_rates, all_ious = evaluate_model_performance(model, dataset, num_samples=100, device=device)
    
    print(f"\n📈 Performance Summary:")
    print(f"   - Mean IoU: {np.mean(all_ious):.4f}")
    print(f"   - Std IoU: {np.std(all_ious):.4f}")
    print(f"   - Success @ IoU 0.25: {success_rates[2]:.4f}")
    
    # Test on a specific sample
    if len(dataset) > 0:
        print("\n🎯 Testing on specific sample...")
        sample = dataset[0]
        if sample is not None:
            rgb_tensor = sample['rgb'].unsqueeze(0)
            depth_tensor = sample['depth'].unsqueeze(0)
            
            grasp_params = predict_grasp(model, rgb_tensor, depth_tensor, device)
            
            # Convert to numpy for visualization
            rgb_np = rgb_tensor.squeeze().permute(1, 2, 0).numpy()
            depth_np = depth_tensor.squeeze().numpy()
            
            # Visualize
            visualize_grasp_prediction(
                rgb_np, depth_np, grasp_params, 
                save_path="sample_prediction.png"
            )

if __name__ == "__main__":
    main()
