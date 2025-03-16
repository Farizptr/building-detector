import folium
import geopandas as gpd
import json
from shapely.geometry import shape

def visualize_geojson(geojson_path, output_path='map.html', zoom_start=10):
    """
    Create an OpenStreetMap visualization from a GeoJSON file
    
    Parameters:
    -----------
    geojson_path : str
        Path to the GeoJSON file
    output_path : str
        Path to save the HTML map output
    zoom_start : int
        Initial zoom level of the map
    """
    # Load GeoJSON data
    try:
        # Try using geopandas first
        gdf = gpd.read_file(geojson_path)
        
        # Get the center of the data for the map
        center_lat = gdf.geometry.centroid.y.mean()
        center_lon = gdf.geometry.centroid.x.mean()
        
        # Create a Folium map centered on the data
        m = folium.Map(location=[center_lat, center_lon], 
                       zoom_start=zoom_start,
                       tiles='OpenStreetMap')
        
        # Add the GeoJSON data to the map with tooltip
        folium.GeoJson(
            gdf,
            name='geojson',
            style_function=lambda x: {
                'fillColor': '#3388ff',
                'color': '#3388ff',
                'weight': 2,
                'fillOpacity': 0.2
            },
            tooltip=folium.GeoJsonTooltip(
                fields=list(gdf.columns),
                aliases=list(gdf.columns),
                localize=True
            )
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save the map
        m.save(output_path)
        print(f"Map saved to {output_path}")
        
    except Exception as e:
        print(f"Error with geopandas approach: {e}")
        print("Trying alternate method...")
        
        # Alternate method using folium directly
        try:
            # Load GeoJSON data
            with open(geojson_path) as f:
                data = json.load(f)
            
            # Find the center of the features
            features = data.get('features', [])
            if not features:
                raise ValueError("No features found in GeoJSON")
                
            # Get first feature to find a center point
            first_geom = shape(features[0]['geometry'])
            center_point = first_geom.centroid
            center = [center_point.y, center_point.x]
            
            # Create a map
            m = folium.Map(location=center, 
                          zoom_start=zoom_start,
                          tiles='OpenStreetMap')
            
            # Add GeoJSON data to map
            folium.GeoJson(
                data,
                name='geojson',
                style_function=lambda x: {
                    'fillColor': '#3388ff',
                    'color': '#3388ff',
                    'weight': 2,
                    'fillOpacity': 0.2
                }
            ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save the map
            m.save(output_path)
            print(f"Map saved to {output_path}")
            
        except Exception as e:
            print(f"Failed to create map: {e}")

# Example usage
if __name__ == "__main__":
    # Replace this with your GeoJSON file path
    geojson_file = "example_area.geojson"
    
    visualize_geojson(geojson_file, output_path="osm_map.html", zoom_start=12)