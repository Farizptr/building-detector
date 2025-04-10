import folium
import json

def create_validation_map(buildings_json_path="polygon_detection_results/buildings.json", output_html="building_validation_map.html"):
    """
    Create an interactive validation map for the detected buildings
    
    Args:
        buildings_json_path: Path to the buildings.json file
        output_html: Path to save the HTML map
        
    Returns:
        Path to the saved HTML map
    """
    # Load buildings
    with open(buildings_json_path, 'r') as f:
        buildings = json.load(f)
    
    # Create a map centered on the first building
    center_lat = buildings[0]['latitude']
    center_lon = buildings[0]['longitude']
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='OpenStreetMap')
    
    # Add satellite view option
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                    attr='Esri', name='Satellite').add_to(m)
    
    # Add building points
    for building in buildings:
        folium.CircleMarker(
            location=[building['latitude'], building['longitude']],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=f"ID: {building['building_id']}, Conf: {building['confidence']:.2f}"
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save the map
    m.save(output_html)
    print(f"Map saved to {output_html}")
    
    return output_html

if __name__ == "__main__":
    create_validation_map() 