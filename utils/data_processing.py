import numpy as np
import cv2
from skimage.draw import polygon, line

def get_grasp_rect_angle(rect_pts):
    """
    Calculate the angle and width of a grasp rectangle.
    """
    p1, p2, p3, p4 = rect_pts
    
    # Calculate edge vectors
    edge1 = p2 - p1
    edge2 = p4 - p1
    
    # Determine which edge is the width (shorter side)
    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        # edge1 is the length, edge2 is the width
        angle = np.arctan2(edge1[1], edge1[0])
        width = np.linalg.norm(edge2)
        center = p1 + edge2 / 2 + edge1 / 2
    else:
        # edge2 is the length, edge1 is the width
        angle = np.arctan2(edge2[1], edge2[0])
        width = np.linalg.norm(edge1)
        center = p1 + edge1 / 2 + edge2 / 2
        
    return angle, width, center

def draw_gaussian(img, pt, sigma=5):
    """
    Draw a 2D gaussian at a specific point on an image.
    """
    h, w = img.shape
    x, y = int(pt[0]), int(pt[1])

    if x < 0 or x >= w or y < 0 or y >= h:
        return img

    # Create a meshgrid
    x_ax = np.arange(w)
    y_ax = np.arange(h)
    xx, yy = np.meshgrid(x_ax, y_ax)

    # Calculate gaussian
    g = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    
    # Combine with existing image using maximum
    img = np.maximum(img, g)
    return img

def generate_grasp_maps_gaussian(grasps, img_size, sigma=5):
    """
    Generate ground-truth maps for quality, angle, and width using Gaussian quality.

    Args:
        grasps (list of np.ndarray): List of grasp rectangles, each a (4, 2) array.
        img_size (tuple): The (height, width) of the output maps.
        sigma (int): The standard deviation of the Gaussian for the quality map.

    Returns:
        tuple: (q_map, cos_map, sin_map, width_map)
    """
    height, width = img_size
    
    # Initialize empty maps
    q_map = np.zeros(img_size, dtype=np.float32)
    cos_map = np.zeros(img_size, dtype=np.float32)
    sin_map = np.zeros(img_size, dtype=np.float32)
    width_map = np.zeros(img_size, dtype=np.float32)
    
    for rect in grasps:
        # Get angle, width, and center for the current rectangle
        angle, grasp_width, center = get_grasp_rect_angle(rect)
        
        # Draw the Gaussian at the center
        q_map = draw_gaussian(q_map, center, sigma=sigma)
        
        # Create a binary mask for the polygon to fill angle/width
        rr, cc = polygon(rect[:, 1], rect[:, 0])
        
        # Clamp coordinates to be within image bounds
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width - 1)
        
        # Fill the maps within the polygon area
        cos_map[rr, cc] = np.cos(2 * angle)
        sin_map[rr, cc] = np.sin(2 * angle)
        
        # Normalize width (e.g., max grasp width of 150 pixels for Cornell)
        MAX_GRASP_WIDTH = 150.0
        width_map[rr, cc] = min(grasp_width, MAX_GRASP_WIDTH) / MAX_GRASP_WIDTH
        
    return q_map, cos_map, sin_map, width_map

def normalize_depth(depth_img, max_depth=1000.0):
    """
    Normalize depth image to [0, 1] and handle missing values.
    """
    depth_img = np.array(depth_img)
    # Inpaint missing depth values (often represented as 0)
    depth_img = cv2.inpaint(depth_img, (depth_img == 0).astype(np.uint8), 3, cv2.INPAINT_NS)
    
    # Clip and normalize
    depth_img = np.clip(depth_img, 0, max_depth)
    depth_img /= max_depth
    
    return depth_img

def normalize_rgb(rgb_img):
    """
    Normalize RGB image to [0, 1].
    """
    return np.array(rgb_img).astype(np.float32) / 255.0

if __name__ == '__main__':
    # Example Usage
    print("Testing Gaussian Ground-Truth Map Generation...")
    img_size = (224, 224)
    
    # Define a sample grasp rectangle
    grasp_rect = np.array([
        [100, 100], [150, 100], [150, 120], [100, 120]
    ])
    
    grasps = [grasp_rect]
    
    q_map, cos_map, sin_map, width_map = generate_grasp_maps_gaussian(grasps, img_size, sigma=10)
    
    print(f"Generated maps of size: {q_map.shape}")
    print(f"Max value in Q map: {np.max(q_map):.2f} (Expected: 1.0)")
    print(f"Min value in Q map: {np.min(q_map):.2f} (Expected: 0.0)")
    
    # Check the center point
    angle, width, center = get_grasp_rect_angle(grasp_rect)
    center_y, center_x = int(center[1]), int(center[0])
    print(f"Center point {center_x, center_y} Q-value: {q_map[center_y, center_x]:.2f}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Sample Grasp")
    plt.imshow(np.zeros(img_size), cmap='gray')
    plt.plot(np.append(grasp_rect[:, 0], grasp_rect[0, 0]), np.append(grasp_rect[:, 1], grasp_rect[0, 1]), 'r-')
    plt.plot(center[0], center[1], 'g+')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Gaussian Quality (Q) Map")
    plt.imshow(q_map, cmap='viridis')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
