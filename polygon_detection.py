import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point, shape
from shapely.ops import unary_union
import mercantile
from tqdm import tqdm
import requests
from io import BytesIO
import sys
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from detection import load_model, detect_buildings
import tempfile

def load_geojson(geojson_path):
    """
    Load a GeoJSON file
    
    Args:
        geojson_path: Path to the GeoJSON file
        
    Returns:
        GeoJSON data as a Python dictionary
    """
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        print(f"GeoJSON loaded successfully from {geojson_path}")
        return geojson_data
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        raise

def extract_polygon(geojson_data):
    """
    Extract polygon(s) from GeoJSON data
    
    Args:
        geojson_data: GeoJSON data as a Python dictionary
        
    Returns:
        Shapely polygon or multipolygon
    """
    polygons = []
    
    # Handle different GeoJSON types
    if geojson_data['type'] == 'FeatureCollection':
        for feature in geojson_data['features']:
            geom = shape(feature['geometry'])
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                polygons.append(geom)
    elif geojson_data['type'] == 'Feature':
        geom = shape(geojson_data['geometry'])
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            polygons.append(geom)
    else:
        # Direct geometry
        geom = shape(geojson_data)
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            polygons.append(geom)
    
    if not polygons:
        raise ValueError("No valid polygons found in GeoJSON")
    
    # Combine all polygons into one
    if len(polygons) == 1:
        return polygons[0]
    else:
        return unary_union(polygons)

def get_tile_bounds(tile):
    """
    Get the bounds of a tile in [west, south, east, north] format
    
    Args:
        tile: A mercantile Tile object
        
    Returns:
        Bounds as [west, south, east, north]
    """
    bounds = mercantile.bounds(tile)
    return [bounds.west, bounds.south, bounds.east, bounds.north]

def get_tiles_for_polygon(polygon, zoom=18):
    """
    Get all tiles that intersect with a polygon
    
    Args:
        polygon: Shapely polygon
        zoom: Zoom level
        
    Returns:
        List of mercantile Tile objects
    """
    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Get all tiles that intersect with the bounds
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zoom))
    
    # Filter tiles to only those that intersect with the polygon
    intersecting_tiles = []
    for tile in tiles:
        tile_bounds = get_tile_bounds(tile)
        tile_polygon = Polygon([
            (tile_bounds[0], tile_bounds[1]),  # SW
            (tile_bounds[2], tile_bounds[1]),  # SE
            (tile_bounds[2], tile_bounds[3]),  # NE
            (tile_bounds[0], tile_bounds[3]),  # NW
            (tile_bounds[0], tile_bounds[1])   # SW (close the polygon)
        ])
        
        if polygon.intersects(tile_polygon):
            intersecting_tiles.append(tile)
    
    return intersecting_tiles

def get_tile_image(tile):
    """
    Get an OSM tile image
    
    Args:
        tile: A mercantile Tile object
        
    Returns:
        PIL Image object of the tile
    """
    # Create URL for the tile
    url = f"https://tile.openstreetmap.org/{tile.z}/{tile.x}/{tile.y}.png"
    
    # Download tile
    headers = {'User-Agent': 'BuildingDetectionBot/1.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Convert response to RGB image
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    else:
        raise Exception(f"Failed to download tile: {response.status_code}")

def process_tile_detections(results):
    """
    Process detection results without visualizing or saving images
    
    Args:
        results: YOLOv8 model detection results
        
    Returns:
        Tuple of (boxes, confidences, class_ids)
    """
    # Get detections (boxes, confidence scores, and class IDs)
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    
    return boxes, confidences, class_ids

def create_stitched_image(tile_detections):
    """
    Create a stitched image from individual tile images stored in memory
    
    Args:
        tile_detections: List of tile detection results with bounds information and images
        
    Returns:
        Tuple of (stitched_image, transform_params)
        - stitched_image: PIL Image of the stitched tiles
        - transform_params: Parameters for transforming geo coordinates to pixel coordinates
    """
    if not tile_detections:
        raise ValueError("No tile detections provided")
    
    # Get the bounds of all tiles
    all_bounds = [td['bounds'] for td in tile_detections]
    
    # Calculate the overall bounds
    min_west = min(bounds[0] for bounds in all_bounds)
    min_south = min(bounds[1] for bounds in all_bounds)
    max_east = max(bounds[2] for bounds in all_bounds)
    max_north = max(bounds[3] for bounds in all_bounds)
    
    # Calculate the width and height in degrees
    width_deg = max_east - min_west
    height_deg = max_north - min_south
    
    # Assume all tiles are 256x256 pixels
    tile_size = 256
    
    # Calculate the number of tiles in each direction
    num_tiles_x = len(set(bounds[0] for bounds in all_bounds))
    num_tiles_y = len(set(bounds[3] for bounds in all_bounds))
    
    # Calculate the size of the stitched image
    width_px = num_tiles_x * tile_size
    height_px = num_tiles_y * tile_size
    
    # Create a blank image
    stitched_image = Image.new('RGB', (width_px, height_px), (255, 255, 255))
    
    # Place each tile in the stitched image
    for td in tile_detections:
        # Get the tile bounds
        west, south, east, north = td['bounds']
        
        # Get the tile image from memory
        if 'image' not in td or td['image'] is None:
            print(f"Warning: Tile image for {td['tile']} not found, skipping")
            continue
        
        tile_image = td['image']
        
        # Calculate the position in the stitched image
        x_pos = int((west - min_west) / width_deg * width_px)
        y_pos = int((max_north - north) / height_deg * height_px)
        
        # Paste the tile image
        stitched_image.paste(tile_image, (x_pos, y_pos))
    
    # Create transform parameters for converting geo coordinates to pixel coordinates
    transform_params = {
        'min_west': min_west,
        'max_north': max_north,
        'width_deg': width_deg,
        'height_deg': height_deg,
        'width_px': width_px,
        'height_px': height_px
    }
    
    return stitched_image, transform_params

def visualize_polygon_detections(geojson_path, results_data, output_path=None):
    """
    Visualize all building detections across the entire GeoJSON area on a single map
    
    Args:
        geojson_path: Path to the GeoJSON file
        results_data: Detection results from detect_buildings_in_polygon
        output_path: Path to save the visualization (optional)
        
    Returns:
        None
    """
    # Load GeoJSON
    geojson_data = load_geojson(geojson_path)
    
    # Extract polygon
    polygon = extract_polygon(geojson_data)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(15, 15))
    
    # Set bounds based on polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Create a stitched image as background if images are available
    if 'detections' in results_data and results_data['detections'] and 'image' in results_data['detections'][0]:
        try:
            # Create stitched image from in-memory tiles
            stitched_image, transform_params = create_stitched_image(results_data['detections'])
            
            # Display the stitched image as background
            ax.imshow(stitched_image, extent=[
                transform_params['min_west'], 
                transform_params['min_west'] + transform_params['width_deg'],
                transform_params['max_north'] - transform_params['height_deg'],
                transform_params['max_north']
            ])
            
            # Set bounds based on the stitched image
            ax.set_xlim(transform_params['min_west'], transform_params['min_west'] + transform_params['width_deg'])
            ax.set_ylim(transform_params['max_north'] - transform_params['height_deg'], transform_params['max_north'])
        except Exception as e:
            print(f"Warning: Failed to create stitched image: {e}")
            print("Falling back to standard visualization")
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
    else:
        # Standard visualization without background image
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    
    # Plot the polygon(s)
    if geojson_data['type'] == 'FeatureCollection':
        for i, feature in enumerate(geojson_data['features']):
            geom = shape(feature['geometry'])
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                # Get a color from the tab20 colormap
                color = plt.cm.tab20(i % 20)
                # Plot the polygon
                x, y = geom.exterior.xy
                ax.plot(x, y, color=color, linewidth=2, alpha=0.7)
                # Add a label if name is available
                if 'properties' in feature and 'name' in feature['properties']:
                    ax.text(np.mean(x), np.mean(y), feature['properties']['name'], 
                            fontsize=12, ha='center', va='center', 
                            bbox=dict(facecolor='white', alpha=0.7))
    else:
        # Plot a single polygon
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='blue', linewidth=2, alpha=0.7)
    
    # Plot tile boundaries
    for tile_detection in results_data['detections']:
        bounds = tile_detection['bounds']
        tile_polygon = Polygon([
            (bounds[0], bounds[1]),  # SW
            (bounds[2], bounds[1]),  # SE
            (bounds[2], bounds[3]),  # NE
            (bounds[0], bounds[3]),  # NW
            (bounds[0], bounds[1])   # SW (close the polygon)
        ])
        x, y = tile_polygon.exterior.xy
        ax.plot(x, y, color='gray', linewidth=0.5, alpha=0.3)
    
    # Plot building detections
    building_patches = []
    confidence_values = []
    
    for tile_detection in results_data['detections']:
        # Get tile bounds
        bounds = tile_detection['bounds']
        west, south, east, north = bounds
        
        # Calculate tile width and height in degrees
        tile_width = east - west
        tile_height = north - south
        
        # Get boxes and confidences
        boxes = tile_detection['boxes']
        confidences = tile_detection['confidences']
        
        # Process each box
        for i, box in enumerate(boxes):
            # Get normalized coordinates (0-1) within the tile
            x1, y1, x2, y2 = box
            
            # Convert to image coordinates (assuming 256x256 images)
            img_width = 256
            img_height = 256
            x1_norm = x1 / img_width
            y1_norm = y1 / img_height
            x2_norm = x2 / img_width
            y2_norm = y2 / img_height
            
            # Convert to geo coordinates
            geo_x1 = west + x1_norm * tile_width
            geo_y1 = north - y1_norm * tile_height  # Flip y-axis
            geo_x2 = west + x2_norm * tile_width
            geo_y2 = north - y2_norm * tile_height  # Flip y-axis
            
            # Create rectangle
            rect = patches.Rectangle(
                (geo_x1, geo_y2),  # Lower left corner (x, y)
                geo_x2 - geo_x1,    # Width
                geo_y1 - geo_y2,    # Height
                linewidth=1,
                edgecolor='none',
                facecolor='none'
            )
            
            building_patches.append(rect)
            confidence_values.append(confidences[i] if i < len(confidences) else 0.5)
    
    # Add building patches to the plot with color based on confidence
    if building_patches:
        # Create a PatchCollection for better performance
        building_collection = PatchCollection(
            building_patches, 
            cmap=plt.cm.viridis,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Set the color array based on confidence values
        building_collection.set_array(np.array(confidence_values))
        
        # Add the collection to the plot
        ax.add_collection(building_collection)
        
        # Add a colorbar
        cbar = plt.colorbar(building_collection, ax=ax)
        cbar.set_label('Confidence Score')
    
    # Set title and labels
    ax.set_title(f'Building Detections in GeoJSON Area ({len(building_patches)} buildings)', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add a legend
    ax.plot([], [], color='blue', linewidth=2, label='GeoJSON Polygon')
    ax.plot([], [], color='gray', linewidth=0.5, label='Tile Boundaries')
    ax.legend(loc='upper right')
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
        plt.close()  # Close the figure to avoid displaying it
    else:
        # Only show the plot if no output path is provided
        plt.tight_layout()
        plt.show()


def save_buildings_to_json(results_data, output_path="buildings.json"):
    """
    Extract building coordinates from detection results and save to a regular JSON file
    
    Args:
        results_data: Detection results from detect_buildings_in_polygon
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    # Create a list to store building data
    buildings = []
    
    # Process each tile detection
    building_id = 1
    for tile_detection in results_data['detections']:
        # Get tile bounds
        bounds = tile_detection['bounds']
        west, south, east, north = bounds
        
        # Calculate tile width and height in degrees
        tile_width = east - west
        tile_height = north - south
        
        # Get boxes and confidences
        boxes = tile_detection['boxes']
        confidences = tile_detection['confidences']
        
        # Process each box
        for i, box in enumerate(boxes):
            # Get normalized coordinates (0-1) within the tile
            x1, y1, x2, y2 = box
            
            # Convert to image coordinates (assuming 256x256 images)
            img_width = 256
            img_height = 256
            x1_norm = x1 / img_width
            y1_norm = y1 / img_height
            x2_norm = x2 / img_width
            y2_norm = y2 / img_height
            
            # Convert to geo coordinates
            geo_x1 = west + x1_norm * tile_width
            geo_y1 = north - y1_norm * tile_height  # Flip y-axis
            geo_x2 = west + x2_norm * tile_width
            geo_y2 = north - y2_norm * tile_height  # Flip y-axis
            
            # Calculate center point
            center_lon = (geo_x1 + geo_x2) / 2
            center_lat = (geo_y1 + geo_y2) / 2
            
            # Create a building entry
            building = {
                "building_id": building_id,
                "latitude": center_lat,
                "longitude": center_lon,
                "confidence": confidences[i] if i < len(confidences) else 0.5
            }
            
            # Add to buildings list
            buildings.append(building)
            building_id += 1
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(buildings, f, indent=2)
    
    print(f"Building coordinates saved to {output_path}")
    print(f"Total buildings saved: {len(buildings)}")
    
    return output_path

def detect_buildings_in_polygon(model, geojson_path, output_dir="polygon_detection_results", zoom=18, conf=0.25):
    """
    Detect buildings within a polygon defined in a GeoJSON file
    
    Args:
        model: Loaded YOLOv8 model
        geojson_path: Path to the GeoJSON file
        output_dir: Directory to save detection results
        zoom: Zoom level for tiles
        conf: Confidence threshold
        
    Returns:
        Dictionary with detection results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load GeoJSON
    geojson_data = load_geojson(geojson_path)
    
    # Extract polygon
    polygon = extract_polygon(geojson_data)
    
    # Get tiles that intersect with the polygon
    tiles = get_tiles_for_polygon(polygon, zoom=zoom)
    print(f"Found {len(tiles)} tiles that intersect with the polygon")
    
    # Download tiles and detect buildings
    all_detections = []
    total_buildings = 0
    
    for tile in tqdm(tiles, desc="Processing tiles"):
        try:
            # Get tile image (in memory)
            tile_image = get_tile_image(tile)
            
            # Create a temporary file for detection
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                # Save the image to the temporary file
                tile_image.save(temp_path, format='PNG')
            
            try:
                # Detect buildings using the temporary file path
                results, img = detect_buildings(model, temp_path, conf=conf)
                
                # Process detection results
                boxes, confidences, class_ids = process_tile_detections(results)
                
                # Add to results
                tile_bounds = get_tile_bounds(tile)
                tile_detections = {
                    'tile': f"{tile.z}/{tile.x}/{tile.y}",
                    'bounds': tile_bounds,
                    'detections': len(boxes),
                    'boxes': boxes.tolist() if len(boxes) > 0 else [],
                    'confidences': confidences.tolist() if len(confidences) > 0 else [],
                    'class_ids': class_ids.tolist() if len(class_ids) > 0 else [],
                    'image': tile_image  # Store the image in memory
                }
                all_detections.append(tile_detections)
                total_buildings += len(boxes)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error processing tile {tile}: {e}")
    
    # Save results to JSON (without images)
    results_path = os.path.join(output_dir, "detection_results.json")
    
    # Create a copy of the results without the images for JSON serialization
    json_results = {
        'total_buildings': total_buildings,
        'total_tiles': len(tiles),
        'zoom': zoom,
        'confidence_threshold': conf,
        'detections': [{k: v for k, v in d.items() if k != 'image'} for d in all_detections]
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Detection results saved to {results_path}")
    print(f"Total buildings detected: {total_buildings}")
    
    # Create a visualization of all detections with in-memory images
    visualization_path = os.path.join(output_dir, "polygon_visualization.png")
    
    # Create results_data with images for visualization
    results_data = {
        'total_buildings': total_buildings,
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

def create_example_geojson(output_path="example_area.geojson"):
    """
    Create an example GeoJSON file with a polygon
    
    Args:
        output_path: Path to save the GeoJSON file
        
    Returns:
        Path to the created GeoJSON file
    """
    # Example polygon (Jakarta area)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Jakarta Example Area"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [106.8456, -6.2088],  # Center point
                            [106.8476, -6.2088],  # East
                            [106.8476, -6.2068],  # Northeast
                            [106.8456, -6.2068],  # North
                            [106.8436, -6.2068],  # Northwest
                            [106.8436, -6.2088],  # West
                            [106.8436, -6.2108],  # Southwest
                            [106.8456, -6.2108],  # South
                            [106.8476, -6.2108],  # Southeast
                            [106.8476, -6.2088],  # Back to East
                            [106.8456, -6.2088]   # Close the polygon
                        ]
                    ]
                }
            }
        ]
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Example GeoJSON saved to {output_path}")
    return output_path

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
    
    # Load the YOLOv8 model
    model = load_model(model_path)
    
    # Detect buildings in the polygon
    results = detect_buildings_in_polygon(model, geojson_path, output_dir, zoom=18, conf=0.25)
    
    print("\nDetection Summary:")
    print(f"Total buildings detected: {results['total_buildings']}")
    print(f"Total tiles processed: {results['total_tiles']}")
    print(f"Results saved to {output_dir}/detection_results.json")
    print(f"Visualization saved to {output_dir}/polygon_visualization.png") 