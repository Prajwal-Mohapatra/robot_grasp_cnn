# ===================== visualize.py =====================
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MPLPolygon
from shapely.geometry import Polygon
import random

def calculate_iou(rect1, rect2):
    """Calculate IoU between two rectangles defined by 4 corner points"""
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

def safe_len(obj):
    """Safely get length of object, handling tensors and lists"""
    if obj is None:
        return 0
    elif torch.is_tensor(obj):
        if obj.numel() == 0:
            return 0
        elif obj.dim() == 1:
            return 1
        else:
            return obj.shape[0]
    elif hasattr(obj, '__len__'):
        return len(obj)
    else:
        return 0

def show_rgb_depth_grasps(rgb, depth, pos_grasps=None, neg_grasps=None, pred_grasp=None, 
                          save_path=None, show_metrics=True):
    """
    Visualize RGB, depth, ground truth grasps, and predicted grasp with metrics.

    Args:
        rgb: Tensor [3, H, W] or np array [H, W, 3]
        depth: Tensor [1, H, W] or np array [H, W]
        pos_grasps: List of positive grasp tensors [4, 2]
        neg_grasps: List of negative grasp tensors [4, 2]
        pred_grasp: Tensor [4, 2] (predicted grasp rectangle)
        save_path: If set, saves the figure to the path
        show_metrics: If True, displays IoU and success metrics
    """

    # Convert tensors to numpy
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(rgb, 0, 1)

    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot Depth
    axes[1].imshow(depth, cmap='gray')
    axes[1].set_title("Depth Image", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Plot RGB with all grasps
    axes[2].imshow(rgb)
    axes[2].set_title("Grasp Visualization", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    legend_handles = []
    metrics_text = []

    # Plot positive grasps (ground truth)
    if pos_grasps is not None:
        for i, grasp in enumerate(pos_grasps):
            if isinstance(grasp, torch.Tensor):
                grasp = grasp.cpu().numpy()
            
            # Create closed polygon
            grasp_closed = np.vstack([grasp, grasp[0]])
            axes[2].plot(grasp_closed[:, 0], grasp_closed[:, 1], 
                        'g-', linewidth=2.5, alpha=0.8)
            
            # Add grasp number
            center = grasp.mean(axis=0)
            axes[2].text(center[0], center[1], f'P{i+1}', 
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.1', facecolor='green', alpha=0.7))
        
        legend_handles.append(Line2D([0], [0], color='green', lw=2.5, 
                                   label=f'Positive Grasps ({len(pos_grasps)})'))

    # Plot negative grasps
    if neg_grasps is not None and len(neg_grasps) > 0:
        for i, grasp in enumerate(neg_grasps):
            if isinstance(grasp, torch.Tensor):
                grasp = grasp.cpu().numpy()
            
            # Create closed polygon
            grasp_closed = np.vstack([grasp, grasp[0]])
            axes[2].plot(grasp_closed[:, 0], grasp_closed[:, 1], 
                        'r--', linewidth=2, alpha=0.6)
            
            # Add grasp number
            center = grasp.mean(axis=0)
            axes[2].text(center[0], center[1], f'N{i+1}', 
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.1', facecolor='red', alpha=0.6))
        
        legend_handles.append(Line2D([0], [0], color='red', lw=2, linestyle='--',
                                   label=f'Negative Grasps ({len(neg_grasps)})'))

    # Plot predicted grasp
    if pred_grasp is not None:
        if isinstance(pred_grasp, torch.Tensor):
            pred_grasp = pred_grasp.detach().cpu().numpy()
        
        # Create closed polygon
        pred_closed = np.vstack([pred_grasp, pred_grasp[0]])
        axes[2].plot(pred_closed[:, 0], pred_closed[:, 1], 
                    'yellow', linewidth=3, alpha=0.9)
        
        # Add prediction marker
        center = pred_grasp.mean(axis=0)
        axes[2].plot(center[0], center[1], 'yo', markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        legend_handles.append(Line2D([0], [0], color='yellow', lw=3, 
                                   label='Predicted Grasp'))

        # Calculate metrics if we have positive grasps
        if pos_grasps is not None and show_metrics:
            max_iou = 0.0
            best_match_idx = -1
            
            for i, pos_grasp in enumerate(pos_grasps):
                if isinstance(pos_grasp, torch.Tensor):
                    pos_grasp = pos_grasp.cpu().numpy()
                
                iou = calculate_iou(pred_grasp, pos_grasp)
                if iou > max_iou:
                    max_iou = iou
                    best_match_idx = i
            
            # Success determination
            success = max_iou >= 0.25
            success_text = "✅ SUCCESS" if success else "❌ FAILED"
            
            metrics_text.extend([
                f"Best IoU: {max_iou:.3f}",
                f"Best Match: P{best_match_idx+1}" if best_match_idx >= 0 else "Best Match: None",
                f"Success (IoU>0.25): {success_text}",
                f"Pos Grasps: {len(pos_grasps)}",
                f"Neg Grasps: {safe_len(neg_grasps)}"
            ])

    # Add legend
    if legend_handles:
        axes[2].legend(handles=legend_handles, loc='upper right', 
                      bbox_to_anchor=(1.0, 1.0), fontsize=10)

    # Add metrics text box
    if metrics_text:
        metrics_str = '\n'.join(metrics_text)
        axes[2].text(0.02, 0.98, metrics_str, transform=axes[2].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()

def show_batch_predictions(model, dataset, device, n_samples=6, save_dir="grasp_outputs"):
    """
    Show predictions for a batch of samples from the dataset
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            if sample is None:
                continue
            
            # Get model prediction
            rgb_batch = sample['rgb'].unsqueeze(0).to(device)
            depth_batch = sample['depth'].unsqueeze(0).to(device)
            pred = model(rgb_batch, depth_batch)
            
            # Convert back to original image space for visualization
            rgb_vis = sample['rgb'].clone()
            depth_vis = sample['depth'].clone()
            
            # Denormalize RGB for visualization
            rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_vis = rgb_vis * rgb_std + rgb_mean
            rgb_vis = torch.clamp(rgb_vis, 0, 1)
            
            # Denormalize depth for visualization
            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
            
            # Extract grasps
            pos_grasps = sample['pos_grasps'] if safe_len(sample['pos_grasps']) > 0 else None
            neg_grasps = sample['neg_grasps'] if safe_len(sample['neg_grasps']) > 0 else None
            pred_grasp = pred[0]
            
            # Create save path
            save_path = os.path.join(save_dir, f"prediction_{i+1:03d}.png")
            
            # Visualize
            show_rgb_depth_grasps(
                rgb_vis, depth_vis, pos_grasps, neg_grasps, pred_grasp,
                save_path=save_path, show_metrics=True
            )
            
            print(f"✅ Saved prediction {i+1}/{len(indices)} to: {save_path}")

def evaluate_model_performance(model, dataset, device, n_samples=100):
    """
    Evaluate model performance on a subset of the dataset
    """
    model.eval()
    
    if n_samples > len(dataset):
        n_samples = len(dataset)
    
    indices = random.sample(range(len(dataset)), n_samples)
    
    total_samples = 0
    successful_grasps = 0
    total_iou = 0.0
    iou_threshold = 0.25
    
    iou_scores = []
    success_flags = []
    
    print(f"Evaluating model on {n_samples} samples...")
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            if sample is None:
                continue
            
            # Get model prediction
            rgb_batch = sample['rgb'].unsqueeze(0).to(device)
            depth_batch = sample['depth'].unsqueeze(0).to(device)
            pred = model(rgb_batch, depth_batch)
            pred_grasp = pred[0].cpu().numpy()
            
            # Calculate IoU with all positive grasps
            pos_grasps = sample['pos_grasps']
            if len(pos_grasps) == 0:
                continue
            
            max_iou = 0.0
            for pos_grasp in pos_grasps:
                iou = calculate_iou(pred_grasp, pos_grasp.cpu().numpy())
                max_iou = max(max_iou, iou)
            
            # Record metrics
            iou_scores.append(max_iou)
            success = max_iou >= iou_threshold
            success_flags.append(success)
            
            if success:
                successful_grasps += 1
            
            total_iou += max_iou
            total_samples += 1
    
    # Calculate final metrics
    success_rate = successful_grasps / total_samples if total_samples > 0 else 0
    avg_iou = total_iou / total_samples if total_samples > 0 else 0
    
    # Calculate IoU distribution
    iou_scores = np.array(iou_scores)
    
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Samples Evaluated: {total_samples}")
    print(f"Success Rate (IoU > {iou_threshold}): {success_rate:.4f} ({successful_grasps}/{total_samples})")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Max IoU: {iou_scores.max():.4f}")
    print(f"Min IoU: {iou_scores.min():.4f}")
    print(f"IoU Std: {iou_scores.std():.4f}")
    print(f"{'='*50}")
    
    return {
        'success_rate': success_rate,
        'avg_iou': avg_iou,
        'max_iou': iou_scores.max(),
        'min_iou': iou_scores.min(),
        'iou_std': iou_scores.std(),
        'total_samples': total_samples,
        'successful_grasps': successful_grasps,
        'iou_scores': iou_scores,
        'success_flags': success_flags
    }
