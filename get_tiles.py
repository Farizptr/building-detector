import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os

def get_osm_tile(latitude: float, longitude: float, zoom: int = 18):
    """Mengambil tile OSM untuk koordinat yang diberikan"""
    
    # Konversi lat/lon ke koordinat tile
    lat_rad = np.radians(latitude)
    n = 2.0 ** zoom
    x_tile = int((longitude + 180.0) / 360.0 * n)
    y_tile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
    
    # Buat URL tile
    url = f"https://tile.openstreetmap.org/{zoom}/{x_tile}/{y_tile}.png"
    
    # Download tile
    headers = {'User-Agent': 'BuildingDetectionBot/1.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Konversi response ke gambar RGB
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return np.array(img), (x_tile, y_tile, zoom)
    else:
        raise Exception(f"Gagal mengambil tile: {response.status_code}")

def resize_and_save_image(img_array, filename, size=(256, 256)):
    """
    Mengubah ukuran gambar dan menyimpan ke file
    
    Args:
        img_array: Array NumPy dari gambar
        filename: Nama file untuk menyimpan gambar
        size: Tuple (width, height) untuk ukuran gambar baru
    """
    # Konversi array NumPy kembali ke gambar PIL
    img = Image.fromarray(img_array)
    
    # Ubah ukuran gambar
    img_resized = img.resize(size, Image.LANCZOS)
    
    # Pastikan direktori ada
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Simpan gambar
    img_resized.save(filename)
    print(f"Gambar berhasil disimpan ke {filename} dengan ukuran {size}")
    
    return np.array(img_resized)

def display_tile(img, title=None):
    """Menampilkan tile gambar"""
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Contoh koordinat (Jakarta)
    latitude = -6.2088
    longitude = 106.8456
    zoom = 18
    
    # Buat direktori untuk menyimpan gambar
    output_dir = "osm_tiles"
    os.makedirs(output_dir, exist_ok=True)
    
    # Ambil tile OSM
    osm_img, osm_coords = get_osm_tile(latitude, longitude, zoom=zoom)
    x_tile, y_tile, z = osm_coords
    
    # Simpan gambar OSM asli
    osm_filename = f"{output_dir}/osm_{latitude}_{longitude}_z{zoom}.png"
    Image.fromarray(osm_img).save(osm_filename)
    print(f"Gambar OSM asli disimpan ke {osm_filename}")
    
    # Ubah ukuran dan simpan gambar OSM 256x256
    osm_resized_filename = f"{output_dir}/osm_{latitude}_{longitude}_z{zoom}_256x256.png"
    osm_resized = resize_and_save_image(osm_img, osm_resized_filename)
    
    # Tampilkan gambar yang sudah di-resize
    display_tile(osm_resized, f"OSM Tile Resized (256x256): {osm_coords}")
    
    print(f"Berhasil mengambil dan menyimpan tile OSM untuk koordinat {latitude}, {longitude}")