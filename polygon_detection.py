import os
import json
import sys
import tempfile
import time
from tqdm import tqdm
import concurrent.futures
from functools import partial

from detection import load_model, detect_buildings
from geojson_utils import load_geojson, extract_polygon, create_example_geojson
from tile_utils import get_tile_bounds, get_tiles_for_polygon, get_tile_image, process_tile_detections, identify_boundary_tiles
from visualization import visualize_polygon_detections
from building_export import save_buildings_to_json

def process_tile_batch(tile_batch, model, conf, filter_edge_buildings=True, edge_buffer_percent=5, boundary_tiles=None):
    """Process a batch of tiles and return their detection results"""
    batch_results = []
    
    for tile in tile_batch:
        try:
            # Get tile image (in memory)
            tile_image = get_tile_image(tile)
            
            # Check if this is a boundary tile
            tile_key = f"{tile.z}/{tile.x}/{tile.y}"
            boundary_edges = boundary_tiles.get(tile_key, []) if boundary_tiles else []
            
            # Create a temporary file for detection
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                # Save the image to the temporary file
                tile_image.save(temp_path, format='PNG')
            
            try:
                # Detect buildings using the temporary file path
                results, img = detect_buildings(model, temp_path, conf=conf)
                
                # Process detection results with filtered buildings tracking
                boxes, confidences, class_ids, filtered_boxes, filtered_confidences, filtered_class_ids = process_tile_detections(
                    results, 
                    filter_edge_buildings=filter_edge_buildings,
                    edge_buffer_percent=edge_buffer_percent,
                    boundary_edges=boundary_edges
                )
                
                # Add to results
                tile_bounds = get_tile_bounds(tile)
                tile_detections = {
                    'tile': tile_key,
                    'bounds': tile_bounds,
                    'detections': len(boxes),
                    'boxes': boxes.tolist() if len(boxes) > 0 else [],
                    'confidences': confidences.tolist() if len(confidences) > 0 else [],
                    'class_ids': class_ids.tolist() if len(class_ids) > 0 else [],
                    'filtered_boxes': [box.tolist() for box in filtered_boxes] if filtered_boxes else [],
                    'filtered_confidences': filtered_confidences if filtered_confidences else [],
                    'filtered_class_ids': filtered_class_ids if filtered_class_ids else [],
                    'filtered_count': len(filtered_boxes),
                    'image': tile_image,  # Store the image in memory
                    'is_boundary': len(boundary_edges) > 0,
                    'boundary_edges': boundary_edges
                }
                batch_results.append(tile_detections)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error processing tile {tile}: {e}")
    
    return batch_results

def create_batches(items, batch_size):
    """Split a list of items into batches of the specified size"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def detect_buildings_in_polygon(model, geojson_path, output_dir="polygon_detection_results", zoom=18, conf=0.25, batch_size=5, filter_edge_buildings=True, edge_buffer_percent=5, preserve_boundary_buildings=True):
    """
    Detect buildings within a polygon defined in a GeoJSON file using optimized batch processing with 2 workers
    
    Args:
        model: Loaded YOLOv8 model
        geojson_path: Path to the GeoJSON file
        output_dir: Directory to save detection results
        zoom: Zoom level for tiles
        conf: Confidence threshold
        batch_size: Number of tiles per batch
        filter_edge_buildings: Whether to filter out buildings at tile edges
        edge_buffer_percent: Percentage of tile edges to consider as buffer zone (default 5%)
        preserve_boundary_buildings: Whether to preserve buildings at the boundary of the area of interest
        
    Returns:
        Dictionary with detection results and execution time
    """
    # Fixed optimal worker count based on previous benchmarks
    num_workers = 2
    
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load GeoJSON
    geojson_data = load_geojson(geojson_path)
    
    # Extract polygon
    polygon = extract_polygon(geojson_data)
    
    # Get tiles that intersect with the polygon
    tiles = get_tiles_for_polygon(polygon, zoom=zoom)
    print(f"Found {len(tiles)} tiles that intersect with the polygon")
    
    # Identify boundary tiles if needed
    boundary_tiles = {}
    if preserve_boundary_buildings:
        boundary_tiles = identify_boundary_tiles(tiles, polygon)
        boundary_tile_count = len(boundary_tiles)
        print(f"Found {boundary_tile_count} boundary tiles")
    
    # Create batches of tiles
    tile_batches = create_batches(tiles, batch_size)
    print(f"Created {len(tile_batches)} batches with batch size {batch_size}")
    
    # Process batches in parallel with 2 workers
    all_detections = []
    total_buildings = 0
    total_filtered_buildings = 0
    
    # Create a partial function with fixed arguments
    process_batch = partial(process_tile_batch, model=model, conf=conf, 
                           filter_edge_buildings=filter_edge_buildings, 
                           edge_buffer_percent=edge_buffer_percent,
                           boundary_tiles=boundary_tiles)
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches and get a list of futures
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(tile_batches)}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                          total=len(tile_batches), 
                          desc=f"Processing tile batches"):
            batch_idx = future_to_batch[future]
            try:
                batch_detections = future.result()
                if batch_detections:
                    all_detections.extend(batch_detections)
                    total_buildings += sum(d['detections'] for d in batch_detections)
                    total_filtered_buildings += sum(d.get('filtered_count', 0) for d in batch_detections)
            except Exception as exc:
                print(f"Batch {batch_idx} generated an exception: {exc}")
    
    # Count boundary tiles with detections
    boundary_tiles_with_detections = sum(1 for d in all_detections if d.get('is_boundary', False) and d['detections'] > 0)
    
    # Save results to JSON (without images)
    results_path = os.path.join(output_dir, "detection_results.json")
    
    # Create a copy of the results without the images for JSON serialization
    json_results = {
        'total_buildings': total_buildings,
        'total_filtered_buildings': total_filtered_buildings,
        'total_tiles': len(tiles),
        'zoom': zoom,
        'confidence_threshold': conf,
        'filter_edge_buildings': filter_edge_buildings,
        'edge_buffer_percent': edge_buffer_percent,
        'preserve_boundary_buildings': preserve_boundary_buildings,
        'boundary_tiles_with_detections': boundary_tiles_with_detections,
        'detections': [{k: v for k, v in d.items() if k != 'image'} for d in all_detections]
    }
    
    # Try to convert numpy types to Python types
    try:
        # Convert all detections to serializable format
        for detection in json_results['detections']:
            if 'filtered_confidences' in detection and detection['filtered_confidences']:
                detection['filtered_confidences'] = [float(conf) for conf in detection['filtered_confidences']]
            if 'filtered_class_ids' in detection and detection['filtered_class_ids']:
                detection['filtered_class_ids'] = [int(cls) for cls in detection['filtered_class_ids']]
    except Exception as e:
        print(f"Error converting data to JSON serializable format: {e}")
    
    # Save results to file
    try:
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except TypeError as e:
        print(f"JSON serialization error: {e}")
        # Find problematic values
        for detection in json_results['detections']:
            for key, value in detection.items():
                try:
                    json.dumps(value)
                except TypeError:
                    print(f"Non-serializable value in key '{key}': {type(value)}")
                    if isinstance(value, list):
                        for i, item in enumerate(value):
                            try:
                                json.dumps(item)
                            except TypeError:
                                print(f"  - Non-serializable item at index {i}: {type(item)}")
        raise
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Add execution time to results
    json_results['execution_time'] = execution_time
    
    print(f"Processing completed in {execution_time:.2f} seconds")
    print(f"Detection results saved to {results_path}")
    print(f"Total buildings detected: {total_buildings}")
    print(f"Total buildings filtered out: {total_filtered_buildings}")
    if preserve_boundary_buildings:
        print(f"Boundary tiles with detections: {boundary_tiles_with_detections}")
    
    # Create a visualization of all detections with in-memory images
    visualization_path = os.path.join(output_dir, "polygon_visualization.png")
    
    # Create results_data with images for visualization
    results_data = {
        'total_buildings': total_buildings,
        'total_filtered_buildings': total_filtered_buildings,
        'total_tiles': len(tiles),
        'zoom': zoom,
        'confidence_threshold': conf,
        'detections': all_detections  # This includes the images
    }
    
    visualize_polygon_detections(geojson_path, results_data, visualization_path)
    
    # Save buildings to JSON
    buildings_json_path = os.path.join(output_dir, "buildings.json")
    save_buildings_to_json(json_results, buildings_json_path)
    
    return json_results

if __name__ == "__main__":
    # Check if shapely and mercantile are installed
    try:
        import shapely
        import mercantile
    except ImportError:
        print("This script requires shapely and mercantile packages.")
        print("Please install them with: pip install shapely mercantile")
        sys.exit(1)
    
    # Path to the model
    model_path = "best.pt"
    
    # Create example GeoJSON if needed
    if len(sys.argv) > 1:
        geojson_path = sys.argv[1]
    else:
        print("No GeoJSON file provided, creating an example...")
        geojson_path = create_example_geojson()
    
    # Output directory
    output_dir = "polygon_detection_results"
    
    # Set batch size (can be specified as command line argument)
    batch_size = 5
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except:
            pass
    
    # Set edge building filtering parameters
    filter_edge_buildings = True          # Default to filtering edge buildings
    edge_buffer_percent = 0.5               # Zero buffer - only filter buildings exactly at the edge
    preserve_boundary_buildings = False   # Set to False to filter buildings at all edges, including boundaries
    
    # Load the YOLOv8 model
    model = load_model(model_path)
    
    # Detect buildings in the polygon with optimized batch processing
    results = detect_buildings_in_polygon(
        model, 
        geojson_path, 
        output_dir, 
        zoom=18, 
        conf=0.25, 
        batch_size=batch_size,
        filter_edge_buildings=filter_edge_buildings,
        edge_buffer_percent=edge_buffer_percent,
        preserve_boundary_buildings=preserve_boundary_buildings
    )
    
    print("\nDetection Summary:")
    print(f"Total buildings detected: {results['total_buildings']}")
    print(f"Total buildings filtered out: {results['total_filtered_buildings']}")
    print(f"Total tiles processed: {results['total_tiles']}")
    print(f"Edge building filtering: {'Enabled' if filter_edge_buildings else 'Disabled'}")
    if filter_edge_buildings:
        print(f"Edge buffer percentage: {edge_buffer_percent}%")
        print(f"Preserve boundary buildings: {'Enabled' if preserve_boundary_buildings else 'Disabled'}")
        if preserve_boundary_buildings:
            print(f"Boundary tiles with detections: {results.get('boundary_tiles_with_detections', 0)}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print(f"Results saved to {output_dir}/detection_results.json")
    print(f"Visualization saved to {output_dir}/polygon_visualization.png")
    print(f"Building data saved to {output_dir}/buildings.json")