import pandas as pd
import geopandas as gpd
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

load_dotenv()

class GoogleMapsExporter:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.maps_api_base = "https://www.google.com/maps/d/create"
        
    def create_new_map(self, title="Street Ends Analysis"):
        """Create a new Google My Maps"""
        url = f"https://www.googleapis.com/mymaps/v1/maps?key={self.api_key}"
        payload = {
            "name": title,
            "description": f"Street ends analysis generated on {datetime.now().strftime('%Y-%m-%d')}"
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            map_data = response.json()
            return map_data['id']
        else:
            logger.error(f"Failed to create map: {response.text}")
            return None
            
    def add_points_to_map(self, map_id, points_df):
        """Add points to the map"""
        url = f"https://www.googleapis.com/mymaps/v1/maps/{map_id}/features/batch?key={self.api_key}"
        
        features = []
        for _, row in points_df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['Longitude'], row['Latitude']]
                },
                "properties": {
                    "name": row['Name'],
                    "description": f"Street end in {row['Name']}"
                }
            }
            features.append(feature)
            
        payload = {"features": features}
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Failed to add points: {response.text}")
            return False
        return True
        
    def export_cities(self, cities):
        """Process multiple cities and combine their street ends"""
        all_points = []
        
        for city in cities:
            # Read the city's GeoJSON file
            filename = f"global_analysis/{city.split(',')[0].strip()}/street_ends.geojson"
            if os.path.exists(filename):
                gdf = gpd.read_file(filename)
                gdf['city'] = city.split(',')[0].strip()
                all_points.append(gdf)
        
        if all_points:
            # Combine all cities' data
            combined_gdf = pd.concat(all_points)
            
            # Convert to DataFrame with lat/lon columns
            export_df = pd.DataFrame({
                'Name': combined_gdf['city'],
                'Latitude': combined_gdf.geometry.y,
                'Longitude': combined_gdf.geometry.x,
            })
            
            # Create new map
            map_id = self.create_new_map()
            if not map_id:
                return "Failed to create map"
                
            # Add points to map
            if self.add_points_to_map(map_id, export_df):
                return f"https://www.google.com/maps/d/edit?mid={map_id}"
            else:
                return "Failed to add points to map"
        
        return "No data to export"

# Example usage:
if __name__ == "__main__":
    exporter = GoogleMapsExporter()
    cities = ["Chicago, USA", "Boston, USA", "Seattle, USA"]
    map_link = exporter.export_cities(cities)
    print(f"Google Maps link: {map_link}") 