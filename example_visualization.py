#!/usr/bin/env python3
"""
Example script demonstrating how to detect buildings in a complex GeoJSON area
and visualize the results
"""

import os
import sys
from polygon_detection import load_model, detect_buildings_in_polygon, create_example_geojson

def main():
    """
    Main function to demonstrate building detection and visualization
    """
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
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        print("Please download the YOLOv8 model file and place it in the project root directory")
        sys.exit(1)
    
    # Use the complex example GeoJSON
    geojson_path = "complex_example.geojson"
    
    # Check if GeoJSON exists
    if not os.path.exists(geojson_path):
        print(f"Complex example GeoJSON not found, creating it...")
        # Create a more complex example with multiple polygons
        with open(geojson_path, 'w') as f:
            f.write('''{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Area 1",
        "description": "First area of interest"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [106.8456, -6.2088],
            [106.8476, -6.2088],
            [106.8476, -6.2068],
            [106.8456, -6.2068],
            [106.8456, -6.2088]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Area 2",
        "description": "Second area of interest"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [106.8436, -6.2088],
            [106.8456, -6.2088],
            [106.8456, -6.2108],
            [106.8436, -6.2108],
            [106.8436, -6.2088]
          ]
        ]
      }
    }
  ]
}''')
    
    # Output directory
    output_dir = "complex_example_results"
    
    # Load the YOLOv8 model
    print("Loading YOLOv8 model...")
    model = load_model(model_path)
    
    # Detect buildings in the polygon
    print(f"Detecting buildings in {geojson_path}...")
    results = detect_buildings_in_polygon(model, geojson_path, output_dir, zoom=18, conf=0.25)
    
    print("\nDetection Summary:")
    print(f"Total buildings detected: {results['total_buildings']}")
    print(f"Total tiles processed: {results['total_tiles']}")
    print(f"Results saved to {output_dir}/detection_results.json")
    print(f"Visualization saved to {output_dir}/polygon_visualization.png")
    
    print("\nYou can also visualize the results again using:")
    print(f"python visualize_results.py {geojson_path} {output_dir}/detection_results.json")

if __name__ == "__main__":
    main() 