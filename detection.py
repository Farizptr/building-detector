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

def detect_buildings(model, image_path, conf=0.25):
    """
    Detect buildings in an image using the loaded YOLOv8 model
    
    Args:
        model: Loaded YOLOv8 model
        image_path: Path to the image file
        conf: Confidence threshold
        
    Returns:
        results: Model detection results
        img: Original image as numpy array
    """
    # Check if image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Run detection
    results = model.predict(image_path, conf=conf)
    
    return results[0], img_array

def visualize_detections(img, results, output_path=None):
    """
    Visualize building detections on the image
    
    Args:
        img: Original image as numpy array
        results: YOLOv8 model detection results
        output_path: Path to save the output image (optional)
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 12))
    
    # Display image
    ax.imshow(img)
    
    # Get detections (boxes, confidence scores, and class IDs)
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    
    # Get class names
    class_names = results.names
    
    # Draw bounding boxes
    for i, box in enumerate(boxes):
        # Get coordinates
        x1, y1, x2, y2 = box
        
        # Create rectangle
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                edgecolor='r', facecolor='none')
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add label with confidence
        class_id = int(class_ids[i])
        class_name = class_names[class_id] if class_id in class_names else f"Class {class_id}"
        ax.text(x1, y1-5, f"{class_name}: {confidences[i]:.2f}", 
                color='white', fontsize=10, backgroundcolor='red')
    
    # Set title
    ax.set_title(f"Building Detection Results ({len(boxes)} buildings found)")
    
    # Remove axis
    ax.axis('off')
    
    # Save output if path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Detection results saved to {output_path}")
    
    # Show plot
    plt.show()
    
    return boxes, confidences, class_ids

def process_tiles(model, input_dir, output_dir="detection_results", conf=0.25):
    """
    Process all tiles in a directory and save detection results
    
    Args:
        model: Loaded YOLOv8 model
        input_dir: Directory containing tiles
        output_dir: Directory to save detection results
        conf: Confidence threshold
        
    Returns:
        List of processed files and their detection counts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    results_summary = []
    
    # Process each image
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"detected_{img_file}")
        
        print(f"Processing {img_file}...")
        
        # Detect buildings
        results, img = detect_buildings(model, input_path, conf=conf)
        
        # Save and visualize detections
        boxes, confidences, _ = visualize_detections(img, results, output_path)
        
        # Add to summary
        results_summary.append({
            'file': img_file,
            'detections': len(boxes),
            'avg_confidence': np.mean(confidences) if len(confidences) > 0 else 0
        })
        
    return results_summary

if __name__ == "__main__":
    # Path to the model and tiles
    model_path = "best.pt"
    tiles_dir = "osm_tiles"
    output_dir = "detection_results"
    
    # Load the YOLOv8 model
    model = load_model(model_path)
    
    # Process all tiles
    results = process_tiles(model, tiles_dir, output_dir, conf=0.25)
    
    # Print summary
    print("\nDetection Summary:")
    for result in results:
        print(f"File: {result['file']} - Buildings: {result['detections']} - Avg Confidence: {result['avg_confidence']:.2f}")