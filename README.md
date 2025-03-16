# Building Detector

A tool for detecting buildings in satellite/aerial imagery using YOLOv8.

## Features

- Detect buildings in individual images
- Process multiple tiles in a directory
- Detect buildings within a polygon defined in a GeoJSON file
- Visualize detection results for individual images
- Visualize all building detections across an entire GeoJSON area

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLOv8 model file (`best.pt`) and place it in the project root directory

## Usage

### Detecting Buildings in an Image

```python
from detection import load_model, detect_buildings, visualize_detections

# Load model
model = load_model("best.pt")

# Detect buildings
results, img = detect_buildings(model, "path/to/image.jpg", conf=0.25)

# Visualize detections
boxes, confidences, class_ids = visualize_detections(img, results, "output.jpg")
```

### Processing Multiple Tiles

```python
from detection import load_model, process_tiles

# Load model
model = load_model("best.pt")

# Process tiles
results = process_tiles(model, "path/to/tiles", "detection_results", conf=0.25)
```

### Detecting Buildings in a GeoJSON Polygon

```python
from polygon_detection import load_model, detect_buildings_in_polygon

# Load model
model = load_model("best.pt")

# Detect buildings in polygon
results = detect_buildings_in_polygon(model, "path/to/polygon.geojson", "polygon_detection_results", zoom=18, conf=0.25)
```

### Visualizing All Detections in a GeoJSON Area

```python
from polygon_detection import visualize_polygon_detections

# Visualize all detections
visualize_polygon_detections("path/to/polygon.geojson", results, "visualization.png")
```

### Creating an Example GeoJSON

```python
from polygon_detection import create_example_geojson

# Create example GeoJSON
geojson_path = create_example_geojson("example_area.geojson")
```

## Command Line Usage

### Detecting Buildings in a GeoJSON Polygon

```bash
python polygon_detection.py path/to/polygon.geojson
```

If no GeoJSON file is provided, an example will be created automatically.

### Visualizing Existing Detection Results

```bash
python visualize_results.py path/to/polygon.geojson path/to/detection_results.json --output visualization.png
```

## Examples

The repository includes example GeoJSON files:
- `example_area.geojson`: A simple polygon around Jakarta
- `complex_example.geojson`: Multiple polygons demonstrating more complex areas

## Output

The detection results are saved in the specified output directory:
- Images with bounding boxes around detected buildings for each tile
- A comprehensive visualization showing all buildings across the entire GeoJSON area
- JSON file with detection details including coordinates, confidence scores, and class IDs

## Visualization Features

The comprehensive visualization includes:
- The original GeoJSON polygon(s)
- Tile boundaries
- All detected buildings colored by confidence score
- A colorbar showing the confidence scale
- Total building count in the title
