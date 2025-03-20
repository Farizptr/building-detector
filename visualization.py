import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, shape

from geojson_utils import load_geojson, extract_polygon
from tile_utils import create_stitched_image

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
    
    # Plot tile boundaries with buffer zones
    for tile_detection in results_data['detections']:
        bounds = tile_detection['bounds']
        # Check if this is a boundary tile
        is_boundary = tile_detection.get('is_boundary', False)
        boundary_edges = tile_detection.get('boundary_edges', [])
        
        # Create the tile polygon
        tile_polygon = Polygon([
            (bounds[0], bounds[1]),  # SW
            (bounds[2], bounds[1]),  # SE
            (bounds[2], bounds[3]),  # NE
            (bounds[0], bounds[3]),  # NW
            (bounds[0], bounds[1])   # SW (close the polygon)
        ])
        
        # Choose colors based on boundary status
        if is_boundary:
            edge_color = 'red'
            line_width = 2.0
            alpha = 0.8
            # Add translucent fill to boundary tiles
            tile_patch = patches.Polygon(
                list(zip(*tile_polygon.exterior.xy)),
                facecolor='yellow',
                edgecolor=edge_color,
                linewidth=line_width,
                alpha=0.2
            )
            ax.add_patch(tile_patch)
            
            # Draw special markers for boundary edges
            for edge in boundary_edges:
                if edge == 'left':
                    ax.plot([bounds[0], bounds[0]], [bounds[1], bounds[3]], color='orange', linewidth=3, alpha=0.9)
                elif edge == 'right':
                    ax.plot([bounds[2], bounds[2]], [bounds[1], bounds[3]], color='orange', linewidth=3, alpha=0.9)
                elif edge == 'top':
                    ax.plot([bounds[0], bounds[2]], [bounds[3], bounds[3]], color='orange', linewidth=3, alpha=0.9)
                elif edge == 'bottom':
                    ax.plot([bounds[0], bounds[2]], [bounds[1], bounds[1]], color='orange', linewidth=3, alpha=0.9)
        else:
            edge_color = 'gray'
            line_width = 0.5
            alpha = 0.3
        
        # Draw the tile boundary
        x, y = tile_polygon.exterior.xy
        ax.plot(x, y, color=edge_color, linewidth=line_width, alpha=alpha)
        
        # Draw buffer zones for edge filtering
        if 'edge_buffer_percent' in results_data:
            buffer_percent = results_data.get('edge_buffer_percent', 5)
            west, south, east, north = bounds
            tile_width = east - west
            tile_height = north - south
            
            # Calculate buffer distance in geo coordinates
            buffer_x = tile_width * (buffer_percent / 100)
            buffer_y = tile_height * (buffer_percent / 100)
            
            # Don't show buffer for boundary edges
            left_buffer = buffer_x if 'left' not in boundary_edges else 0
            right_buffer = buffer_x if 'right' not in boundary_edges else 0
            top_buffer = buffer_y if 'top' not in boundary_edges else 0
            bottom_buffer = buffer_y if 'bottom' not in boundary_edges else 0
            
            # Create inner buffer polygon
            buffer_polygon = Polygon([
                (west + left_buffer, south + bottom_buffer),
                (east - right_buffer, south + bottom_buffer),
                (east - right_buffer, north - top_buffer),
                (west + left_buffer, north - top_buffer),
                (west + left_buffer, south + bottom_buffer)
            ])
            
            # Draw the buffer zone
            buffer_x, buffer_y = buffer_polygon.exterior.xy
            ax.plot(buffer_x, buffer_y, color='purple', linewidth=1, linestyle='--', alpha=0.5)
    
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
        
        # Also draw filtered buildings with dashed red lines
        filtered_boxes = tile_detection.get('filtered_boxes', [])
        for i, box in enumerate(filtered_boxes):
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
            
            # Draw dashed rectangle for filtered buildings
            rect = patches.Rectangle(
                (geo_x1, geo_y2),  # Lower left corner (x, y)
                geo_x2 - geo_x1,    # Width
                geo_y1 - geo_y2,    # Height
                linewidth=1.5,
                edgecolor='red',
                linestyle='--',
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
    
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
    total_buildings = results_data.get('total_buildings', 0)
    total_filtered = results_data.get('total_filtered_buildings', 0)
    ax.set_title(f'Building Detections ({total_buildings} kept, {total_filtered} filtered)', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add a legend
    ax.plot([], [], color='blue', linewidth=2, label='GeoJSON Polygon')
    ax.plot([], [], color='gray', linewidth=0.5, label='Tile Boundaries')
    ax.plot([], [], color='red', linewidth=2, label='Boundary Tiles')
    ax.plot([], [], color='orange', linewidth=3, label='Boundary Edges')
    ax.plot([], [], color='purple', linewidth=1, linestyle='--', label='Edge Buffer Zone')
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.2, edgecolor='red', label='Boundary Tile Area'))
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, edgecolor='red', linestyle='--', facecolor='none', label='Filtered Buildings'))
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