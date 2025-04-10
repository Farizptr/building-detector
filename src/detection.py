import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

def load_model(model_path="best.pt"):
    """
    Load YOLOv8 model from the specified path
    
    Args:
        model_path: Path to the YOLOv8 model file (.pt)
        
    Returns:
        Loaded YOLOv8 model
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Load YOLOv8 model
    try:
        model = YOLO(model_path)
        print(f"YOLOv8 model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def detect_buildings(model, image_path, conf=0.25, filter_edge_buildings=True, edge_buffer_percent=5, boundary_edges=None):
    """
    Detect buildings in an image using YOLOv8
    
    Args:
        model: Loaded YOLOv8 model
        image_path: Path to the image file
        conf: Confidence threshold for detections
        filter_edge_buildings: Whether to filter out buildings on the edge of the image
        edge_buffer_percent: Percentage of image width/height to consider as edge buffer
        boundary_edges: List of edges that are on the boundary of the polygon ['top', 'right', 'bottom', 'left']
        
    Returns:
        Dictionary containing detection results
    """
    # Run inference
    try:
        results = model(image_path, conf=conf, verbose=False)[0]
    except Exception as e:
        print(f"Error during inference: {e}")
        return {"boxes": [], "confidences": []}
    
    # Extract bounding boxes and confidences
    boxes = []
    confidences = []
    
    # Get image dimensions
    if isinstance(image_path, str):
        img = Image.open(image_path)
        img_width, img_height = img.size
    else:
        # Assume it's a numpy array or PIL Image
        if hasattr(image_path, 'shape'):
            img_height, img_width = image_path.shape[:2]
        else:
            img_width, img_height = image_path.size
    
    # Calculate edge buffer in pixels
    buffer_x = int(img_width * edge_buffer_percent / 100)
    buffer_y = int(img_height * edge_buffer_percent / 100)
    
    # Process results
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        
        # Skip buildings on the edge if filter_edge_buildings is True
        if filter_edge_buildings:
            # Skip if the building touches any edge that is on the boundary
            if boundary_edges:
                if ('top' in boundary_edges and y1 < buffer_y) or \
                   ('right' in boundary_edges and x2 > img_width - buffer_x) or \
                   ('bottom' in boundary_edges and y2 > img_height - buffer_y) or \
                   ('left' in boundary_edges and x1 < buffer_x):
                    continue
        
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(results.boxes.conf.cpu().numpy()[len(confidences)]))
    
    return {"boxes": boxes, "confidences": confidences}

def visualize_detections(image_path, boxes, confidences=None, output_path=None):
    """
    Visualize building detections on an image
    
    Args:
        image_path: Path to the image file
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        confidences: List of confidence scores for each box
        output_path: Path to save the visualization (optional)
        
    Returns:
        None
    """
    # Load image
    img = plt.imread(image_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # Display image
    ax.imshow(img)
    
    # Add bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add rectangle to the plot
        ax.add_patch(rect)
        
        # Add confidence score if available
        if confidences and i < len(confidences):
            ax.text(x1, y1, f"{confidences[i]:.2f}", color='white', fontsize=8, 
                    bbox=dict(facecolor='red', alpha=0.5))
    
    # Remove axis
    ax.axis('off')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()