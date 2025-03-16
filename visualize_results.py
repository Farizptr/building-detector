#!/usr/bin/env python3
"""
Visualize building detection results from a GeoJSON polygon
"""

import os
import sys
import json
import argparse
from polygon_detection import visualize_polygon_detections

def main():
    """
    Main function to visualize building detection results
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize building detection results from a GeoJSON polygon')
    parser.add_argument('geojson_path', help='Path to the GeoJSON file')
    parser.add_argument('results_path', help='Path to the detection results JSON file')
    parser.add_argument('--output', '-o', help='Path to save the visualization (optional)')
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.geojson_path):
        print(f"Error: GeoJSON file {args.geojson_path} not found")
        sys.exit(1)
    
    if not os.path.exists(args.results_path):
        print(f"Error: Results file {args.results_path} not found")
        sys.exit(1)
    
    # Load results data
    try:
        with open(args.results_path, 'r') as f:
            results_data = json.load(f)
        print(f"Results loaded successfully from {args.results_path}")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    # Set default output path if not provided
    output_path = args.output
    if not output_path:
        output_dir = os.path.dirname(args.results_path)
        output_path = os.path.join(output_dir, "polygon_visualization.png")
    
    # Visualize results
    print(f"Visualizing {results_data['total_buildings']} buildings across {results_data['total_tiles']} tiles...")
    visualize_polygon_detections(args.geojson_path, results_data, output_path)
    
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main() 