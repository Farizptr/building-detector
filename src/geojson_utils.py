import json
from shapely.geometry import shape
from shapely.ops import unary_union

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
        Shapely polygon
    """
    # Check if it's a FeatureCollection
    if geojson_data.get('type') == 'FeatureCollection':
        # Extract all polygons from features
        polygons = []
        for feature in geojson_data.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') in ['Polygon', 'MultiPolygon']:
                polygons.append(shape(geom))
        
        # Combine all polygons into one
        if polygons:
            return unary_union(polygons)
        else:
            raise ValueError("No polygon features found in GeoJSON")
    
    # Check if it's a Feature
    elif geojson_data.get('type') == 'Feature':
        geom = geojson_data.get('geometry', {})
        if geom.get('type') in ['Polygon', 'MultiPolygon']:
            return shape(geom)
        else:
            raise ValueError("Feature does not contain a polygon geometry")
    
    # Check if it's a direct Geometry
    elif geojson_data.get('type') in ['Polygon', 'MultiPolygon']:
        return shape(geojson_data)
    
    else:
        raise ValueError("Invalid GeoJSON format or no polygon found")

def create_example_geojson(output_path="example_polygon.geojson"):
    """
    Create an example GeoJSON file with a polygon
    
    Args:
        output_path: Path to save the GeoJSON file
        
    Returns:
        Path to the created GeoJSON file
    """
    # Example polygon (small area in San Francisco)
    example_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.4308, 37.7759],
                            [-122.4308, 37.7769],
                            [-122.4298, 37.7769],
                            [-122.4298, 37.7759],
                            [-122.4308, 37.7759]
                        ]
                    ]
                }
            }
        ]
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(example_geojson, f, indent=2)
    
    print(f"Example GeoJSON created at {output_path}")
    return output_path