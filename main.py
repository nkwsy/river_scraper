#main.py

from utils.osmnx_load import get_ox
ox = get_ox()

import geopandas as gpd
from shapely.geometry import Point

from enrich_data import LocationEnricher
from render_map import StreetEndRenderer
from utils.logging_config import setup_logger

class StreetEndFinder:
    def __init__(self, location, threshold_distance=10):
        self.logger = setup_logger(__name__)
        self.location = location
        self.threshold_distance = threshold_distance
        self.water_features = None
        self.streets = None
        self.street_ends = []
        self.near_water = []
        
    def get_data(self):
        """Download street and water data"""
        self.logger.info(f"Processing {self.location}")
        
        # Get street network
        graph = ox.graph_from_place(self.location, network_type='all')
        self.streets = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        
        # Get water features
        water_tags = {
            "natural": [ "stream", "riverbank"],
            "water": ["river", "stream", "canal"]
        }
        self.water_features = ox.features_from_place(self.location, tags=water_tags)
        
        self.logger.info(f"Found {len(self.water_features)} water features")
        self.logger.info(f"Found {len(self.streets)} streets")
        
    def find_street_ends(self):
        """Identify all street endpoints"""
        self.logger.info("Collecting street ends...")
        for _, row in self.streets.iterrows():
            start_point = Point(row['geometry'].coords[0])
            end_point = Point(row['geometry'].coords[-1])
            self.street_ends.append(start_point)
            self.street_ends.append(end_point)
        self.logger.info(f"Collected {len(self.street_ends)} street ends")
        
    def find_near_water(self):
        """Find street ends near water"""
        river_geom = self.water_features.geometry.unary_union
        seen_coords = set()
        self.logger.info(f"\nChecking proximity to water (threshold: {self.threshold_distance}m)...")
        
        for i, point in enumerate(self.street_ends):
            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(self.street_ends)} points...")
            
            distance = river_geom.distance(point) * 111000
            if distance < self.threshold_distance:
                coord = (round(point.x, 6), round(point.y, 6))
                if coord not in seen_coords:
                    self.near_water.append(point)
                    seen_coords.add(coord)
        
        self.logger.info("\nResults:")
        self.logger.info(f"Total street ends: {len(self.street_ends)}")
        self.logger.info(f"Unique street ends near water: {len(self.near_water)}")
        self.logger.info(f"Threshold distance: {self.threshold_distance} meters")
        
    def save_results(self, filename='street_ends_near_river.geojson'):
        """Save results to GeoJSON"""
        points_gdf = gpd.GeoDataFrame(geometry=self.near_water, crs="EPSG:4326")
        points_gdf.to_file(filename, driver='GeoJSON')
        self.logger.info(f"\nResults saved to {filename}")
        
    def process(self):
        """Run the complete analysis"""
        self.get_data()
        self.find_street_ends()
        self.find_near_water()
        self.save_results()
        return self.water_features, self.near_water

# Example usage:
if __name__ == "__main__":
    # Can be run for different cities
    cities = ["Skokie, USA"]
    for city in cities:
        # Find street ends
        finder = StreetEndFinder(city, threshold_distance=10)
        finder.process()
        
        # Render the results
        output_file = f"street_ends_{city.split(',')[0].lower()}.html"
        renderer = StreetEndRenderer('street_ends_near_river.geojson', city)
        renderer.render(output_file)

        # enrich data
        enricher = LocationEnricher()
        enricher.process_locations('street_ends_near_river.geojson')
