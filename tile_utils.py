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

def is_building_at_edge(box, buffer_percent=5, boundary_edges=None):
    """
    Determine if a building is at the edge of a tile
    
    Args:
        box: Bounding box as [x1, y1, x2, y2] in pixel coordinates
        buffer_percent: Percentage of image size to use as edge buffer
        boundary_edges: List of edges ('left', 'right', 'top', 'bottom') that are at the boundary
                        of the area of interest and should not be filtered
        
    Returns:
        True if the building is at a non-boundary edge, False otherwise
    """
    # Assuming 256x256 pixel tiles
    tile_size = 256
    buffer = int(tile_size * (buffer_percent / 100))
    
    x1, y1, x2, y2 = box
    
    # If no boundary_edges provided, check all edges
    if boundary_edges is None:
        boundary_edges = []
    
    # Check if building touches each edge buffer zone, but ignore boundary edges
    # Left edge
    if 'left' not in boundary_edges and x1 < buffer:
        return True
    
    # Top edge
    if 'top' not in boundary_edges and y1 < buffer:
        return True
    
    # Right edge
    if 'right' not in boundary_edges and x2 > (tile_size - buffer):
        return True
    
    # Bottom edge
    if 'bottom' not in boundary_edges and y2 > (tile_size - buffer):
        return True
    
    return False

def identify_boundary_tiles(tiles, polygon):
    """
    Identify which tiles are at the boundary of the polygon and which edges are outside
    
    Args:
        tiles: List of mercantile Tile objects that intersect with the polygon
        polygon: Shapely polygon representing the area of interest
        
    Returns:
        Dictionary mapping tile keys to a list of boundary edges ('left', 'right', 'top', 'bottom')
    """
    try:
        boundary_tiles = {}
        
        for tile in tiles:
            # Get tile bounds and create tile polygon
            tile_bounds = get_tile_bounds(tile)
            west, south, east, north = tile_bounds
            
            # Create polygons for each edge
            edge_width = 0.00001  # Small buffer for edge detection
            
            left_edge = Polygon([
                (west, south),
                (west + edge_width, south),
                (west + edge_width, north),
                (west, north),
                (west, south)
            ])
            
            right_edge = Polygon([
                (east - edge_width, south),
                (east, south),
                (east, north),
                (east - edge_width, north),
                (east - edge_width, south)
            ])
            
            top_edge = Polygon([
                (west, north - edge_width),
                (east, north - edge_width),
                (east, north),
                (west, north),
                (west, north - edge_width)
            ])
            
            bottom_edge = Polygon([
                (west, south),
                (east, south),
                (east, south + edge_width),
                (west, south + edge_width),
                (west, south)
            ])
            
            # Check which edges are outside the polygon
            outside_edges = []
            
            if not polygon.intersects(left_edge) or polygon.intersection(left_edge).area < left_edge.area * 0.5:
                outside_edges.append('left')
            
            if not polygon.intersects(right_edge) or polygon.intersection(right_edge).area < right_edge.area * 0.5:
                outside_edges.append('right')
            
            if not polygon.intersects(top_edge) or polygon.intersection(top_edge).area < top_edge.area * 0.5:
                outside_edges.append('top')
            
            if not polygon.intersects(bottom_edge) or polygon.intersection(bottom_edge).area < bottom_edge.area * 0.5:
                outside_edges.append('bottom')
            
            # If any edges are outside, this is a boundary tile
            if outside_edges:
                tile_key = f"{tile.z}/{tile.x}/{tile.y}"
                boundary_tiles[tile_key] = outside_edges
        
        return boundary_tiles
    except Exception as e:
        print(f"Error in identify_boundary_tiles: {e}")
        return {} 