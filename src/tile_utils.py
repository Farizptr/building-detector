import mercantile
import requests
from io import BytesIO
from PIL import Image
from shapely.geometry import Polygon

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

def process_tile_detections(results, filter_edge_buildings=True, edge_buffer_percent=5, boundary_edges=None):
    """
    Process detection results without visualizing or saving images
    
    Args:
        results: YOLOv8 model detection results
        filter_edge_buildings: Whether to filter out buildings at tile edges
        edge_buffer_percent: Percentage of image size to use as edge buffer
        boundary_edges: List of edges that are at the boundary of the area of interest
                        and should preserve buildings
        
    Returns:
        Tuple of (boxes, confidences, class_ids, filtered_boxes, filtered_confidences, filtered_class_ids)
        where the filtered_* components are the buildings that were filtered out
    """
    # Get detections (boxes, confidence scores, and class IDs)
    all_boxes = results.boxes.xyxy.cpu().numpy()
    all_confidences = results.boxes.conf.cpu().numpy()
    all_class_ids = results.boxes.cls.cpu().numpy()
    
    # Initialize filtered boxes
    filtered_boxes = []
    filtered_confidences = []
    filtered_class_ids = []
    
    # Filter out buildings at edges if requested
    if filter_edge_buildings and len(all_boxes) > 0:
        # Create masks for buildings to keep
        keep_mask = []
        for box in all_boxes:
            # Check if building is NOT at edge (or is at a boundary edge we want to preserve)
            keep = not is_building_at_edge(box, edge_buffer_percent, boundary_edges)
            keep_mask.append(keep)
        
        # Convert mask to numpy array
        keep_mask = [bool(x) for x in keep_mask]  # Ensure boolean values
        
        # Save filtered out buildings
        for i in range(len(all_boxes)):
            if not keep_mask[i]:
                filtered_boxes.append(all_boxes[i])
                filtered_confidences.append(all_confidences[i])
                filtered_class_ids.append(all_class_ids[i])
        
        # Apply filtering to keep only non-edge buildings
        boxes = all_boxes[keep_mask]
        confidences = all_confidences[keep_mask]
        class_ids = all_class_ids[keep_mask]
    else:
        boxes = all_boxes
        confidences = all_confidences
        class_ids = all_class_ids
    
    return boxes, confidences, class_ids, filtered_boxes, filtered_confidences, filtered_class_ids

def is_building_at_edge(box, buffer_percent=5, boundary_edges=None):
    """
    Check if a building is at the edge of a tile
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
        buffer_percent: Percentage of image size to use as edge buffer
        boundary_edges: List of edges that are at the boundary of the area of interest
                        and should preserve buildings
    
    Returns:
        True if building is at edge, False otherwise
    """
    # Initialize boundary edges if not provided
    if boundary_edges is None:
        boundary_edges = []
    
    # Image dimensions (assuming 256x256 images)
    img_width = 256
    img_height = 256
    
    # Calculate buffer distance in pixels
    buffer_x = img_width * (buffer_percent / 100)
    buffer_y = img_height * (buffer_percent / 100)
    
    # Check if building is at left edge (not boundary)
    if 'left' not in boundary_edges and box[0] < buffer_x:
        return True
    
    # Check if building is at right edge (not boundary)
    if 'right' not in boundary_edges and box[2] > (img_width - buffer_x):
        return True
    
    # Check if building is at top edge (not boundary)
    if 'top' not in boundary_edges and box[1] < buffer_y:
        return True
    
    # Check if building is at bottom edge (not boundary)
    if 'bottom' not in boundary_edges and box[3] > (img_height - buffer_y):
        return True
    
    # Building is not at edge or is at a boundary edge we want to preserve
    return False

def identify_boundary_tiles(tiles, polygon):
    """
    Identify which tiles are at the boundary of the polygon
    
    Args:
        tiles: List of mercantile Tile objects
        polygon: Shapely polygon
        
    Returns:
        Dictionary mapping tile keys to list of boundary edges
    """
    boundary_tiles = {}
    
    for tile in tiles:
        # Get tile bounds
        tile_bounds = get_tile_bounds(tile)
        west, south, east, north = tile_bounds
        
        # Create tile polygon
        tile_polygon = Polygon([
            (west, south),  # SW
            (east, south),  # SE
            (east, north),  # NE
            (west, north),  # NW
            (west, south)   # SW (close the polygon)
        ])
        
        # Check if this tile is at the boundary of the polygon
        boundary_edges = []
        
        # Check each edge of the tile
        # Left edge
        left_edge = Polygon([
            (west, south),
            (west + 0.00001, south),
            (west + 0.00001, north),
            (west, north),
            (west, south)
        ])
        if polygon.intersects(left_edge):
            boundary_edges.append('left')
        
        # Right edge
        right_edge = Polygon([
            (east - 0.00001, south),
            (east, south),
            (east, north),
            (east - 0.00001, north),
            (east - 0.00001, south)
        ])
        if polygon.intersects(right_edge):
            boundary_edges.append('right')
        
        # Top edge
        top_edge = Polygon([
            (west, north - 0.00001),
            (east, north - 0.00001),
            (east, north),
            (west, north),
            (west, north - 0.00001)
        ])
        if polygon.intersects(top_edge):
            boundary_edges.append('top')
        
        # Bottom edge
        bottom_edge = Polygon([
            (west, south),
            (east, south),
            (east, south + 0.00001),
            (west, south + 0.00001),
            (west, south)
        ])
        if polygon.intersects(bottom_edge):
            boundary_edges.append('bottom')
        
        # If any edges are at the boundary, add to results
        if boundary_edges:
            tile_key = f"{tile.z}/{tile.x}/{tile.y}"
            boundary_tiles[tile_key] = boundary_edges
    
    return boundary_tiles

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