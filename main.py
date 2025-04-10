import sys
import argparse
from src.detection import load_model
from src.polygon_detection import detect_buildings_in_polygon
from src.validate import create_validation_map
from src.geojson_utils import create_example_geojson

def main():
    """Main entry point for the building detector"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect buildings in a GeoJSON polygon')
    parser.add_argument('--geojson', '-g', type=str, required=False,
                        help='Path to a GeoJSON file containing a polygon')
    parser.add_argument('--model', '-m', type=str, default='best.pt',
                        help='Path to the YOLOv8 model file (.pt)')
    parser.add_argument('--zoom', '-z', type=int, default=18,
                        help='Zoom level for tiles (default: 18)')
    parser.add_argument('--confidence', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--batch-size', '-b', type=int, default=5,
                        help='Number of tiles per batch (default: 5)')
    parser.add_argument('--output-dir', '-o', type=str, default='polygon_detection_results',
                        help='Output directory for results (default: polygon_detection_results)')
    parser.add_argument('--filter-edge', action='store_true', default=True,
                        help='Filter out buildings at tile edges (default: True)')
    parser.add_argument('--edge-buffer', type=float, default=5.0,
                        help='Percentage of tile edges to consider as buffer zone (default: 5%)')
    parser.add_argument('--preserve-boundary', action='store_true', default=True,
                        help='Preserve buildings at polygon boundary (default: True)')
    parser.add_argument('--create-example', action='store_true',
                        help='Create an example GeoJSON polygon before detection')
    
    args = parser.parse_args()
    
    # Create example GeoJSON if requested
    if args.create_example:
        example_path = create_example_geojson()
        print(f"Created example GeoJSON: {example_path}")
        if not args.geojson:
            args.geojson = example_path
    
    # Ensure required parameters are provided
    if not args.geojson:
        parser.error("Please provide a GeoJSON file path with --geojson or use --create-example")
    
    # Load model
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Detect buildings
    try:
        results = detect_buildings_in_polygon(
            model=model,
            geojson_path=args.geojson,
            output_dir=args.output_dir,
            zoom=args.zoom,
            conf=args.confidence,
            batch_size=args.batch_size,
            filter_edge_buildings=args.filter_edge,
            edge_buffer_percent=args.edge_buffer,
            preserve_boundary_buildings=args.preserve_boundary
        )
        
        # Create validation map
        buildings_json_path = f"{args.output_dir}/buildings.json"
        output_html = "building_validation_map.html"
        create_validation_map(buildings_json_path, output_html)
        
        print("\nProcessing complete!")
        print(f"Total buildings detected: {results['total_buildings']}")
        print(f"Total filtered buildings: {results['total_filtered_buildings']}")
        print(f"Execution time: {results['execution_time']:.2f} seconds")
        print(f"Results saved to directory: {args.output_dir}")
        print(f"Interactive map saved to: {output_html}")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 