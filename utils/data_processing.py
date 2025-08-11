import numpy as np
import cv2
from skimage.draw import polygon

def get_grasp_rect_angle(rect_pts):
    """
    Calculate the angle of a grasp rectangle.
    The angle is the orientation of the longer side.
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
    else:
        # edge2 is the length, edge1 is the width
        angle = np.arctan2(edge2[1], edge2[0])
        width = np.linalg.norm(edge1)
        
    return angle, width

def generate_grasp_maps(grasps, img_size):
    """
    Generate ground-truth maps for quality, angle, and width.

    Args:
        grasps (list of np.ndarray): List of grasp rectangles, each a (4, 2) array.
        img_size (tuple): The (height, width) of the output maps.

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
        # Get angle and width for the current rectangle
        angle, grasp_width = get_grasp_rect_angle(rect)
        
        # Create a binary mask for the polygon
        rr, cc = polygon(rect[:, 1], rect[:, 0])
        
        # Clamp coordinates to be within image bounds
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width - 1)
        
        # Fill the maps within the polygon area
        q_map[rr, cc] = 1.0
        cos_map[rr, cc] = np.cos(2 * angle)
        sin_map[rr, cc] = np.sin(2 * angle)
        
        # Normalize width (e.g., max grasp width of 150 pixels for Cornell)
        # This value should be tuned based on the dataset
        MAX_GRASP_WIDTH = 150.0
        width_map[rr, cc] = min(grasp_width, MAX_GRASP_WIDTH) / MAX_GRASP_WIDTH
        
    return q_map, cos_map, sin_map, width_map

def normalize_depth(depth_img, max_depth=1000.0):
    """
    Normalize depth image to [0, 1] and handle missing values.
    """
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
    return rgb_img.astype(np.float32) / 255.0

if __name__ == '__main__':
    # Example Usage
    print("Testing Ground-Truth Map Generation...")
    img_size = (224, 224)
    
    # Define a sample grasp rectangle (e.g., a 45-degree rectangle)
    grasp_rect = np.array([
        [100, 100],
        [150, 150],
        [130, 170],
        [80, 120]
    ])
    
    grasps = [grasp_rect]
    
    q_map, cos_map, sin_map, width_map = generate_grasp_maps(grasps, img_size)
    
    print(f"Generated maps of size: {q_map.shape}")
    print(f"Max value in Q map: {np.max(q_map):.2f} (Expected: 1.0)")
    print(f"Min value in Q map: {np.min(q_map):.2f} (Expected: 0.0)")
    
    # Check a point inside the rectangle
    point_inside_y, point_inside_x = 120, 120
    print(f"\nValues at a point inside the grasp polygon ({point_inside_y}, {point_inside_x}):")
    print(f"  Quality: {q_map[point_inside_y, point_inside_x]:.2f}")
    print(f"  Cos(2θ): {cos_map[point_inside_y, point_inside_x]:.2f}")
    print(f"  Sin(2θ): {sin_map[point_inside_y, point_inside_x]:.2f}")
    print(f"  Width: {width_map[point_inside_y, point_inside_x]:.2f}")
    
    # Visualize the Q map
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Sample Grasp Rectangle")
    plt.imshow(np.zeros(img_size), cmap='gray')
    plt.plot(np.append(grasp_rect[:, 0], grasp_rect[0, 0]), np.append(grasp_rect[:, 1], grasp_rect[0, 1]), 'r-')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Quality (Q) Map")
    plt.imshow(q_map, cmap='viridis')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
