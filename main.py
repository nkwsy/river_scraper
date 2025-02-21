#main.py

from utils.osmnx_load import get_ox
ox = get_ox()

import geopandas as gpd
from shapely.geometry import Point

from enrich_data import LocationEnricher
from render_map import StreetEndRenderer
from utils.logging_config import setup_logger

class StreetEndFinder:
    def __init__(self, location, threshold_distance=10, **kwargs):
        self.logger = setup_logger(__name__)
        self.geojson_file = kwargs.get('geojson_file', 'street_ends_near_river.geojson')
        self.location = location
        self.threshold_distance = threshold_distance
        self.water_features = None
        self.streets = None
        self.street_ends = []
        self.near_water = []
        self.should_find_street_ends = kwargs.get('find_street_ends', True)
        self.should_enrich_data = kwargs.get('enrich_data', True)
        
    def get_data(self):
        """Download street and water data"""
        self.logger.info(f"Processing {self.location}")
        
        # Get water features first
        water_tags = {
            # "natural": ["stream", "riverbank"],
            "water": ["river", "stream", "canal"]
        }
        self.water_features = ox.features_from_place(self.location, tags=water_tags)
        
        if self.water_features.empty:
            self.logger.warning("No water features found")
            return
            
        # Create a buffer around water features (e.g., 100 meters)
        # Convert to projected CRS for accurate buffer distance
        water_features_proj = self.water_features.to_crs('EPSG:3857')
        buffer_distance = 100  # meters
        water_buffer = water_features_proj.geometry.buffer(buffer_distance).unary_union
        
        # Convert back to WGS84 for OSMnx
        water_buffer_wgs = gpd.GeoSeries([water_buffer], crs='EPSG:3857').to_crs('EPSG:4326')[0]
        
        # Get street network only within the buffered area
        graph = ox.graph_from_polygon(
            water_buffer_wgs,
            network_type='all',
            simplify=True,
            truncate_by_edge=True
        )
        self.streets = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        
        self.logger.info(f"Found {len(self.water_features)} water features")
        self.logger.info(f"Found {len(self.streets)} streets within {buffer_distance}m of water")
        
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
        seen_points = []  # List of Points instead of coords
        self.logger.info(f"\nChecking proximity to water (threshold: {self.threshold_distance}m)...")
        
        # Convert to projected CRS for accurate distance measurements
        river_geom_proj = gpd.GeoSeries([river_geom], crs='EPSG:4326').to_crs('EPSG:3857')[0]
        
        for i, point in enumerate(self.street_ends):
            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(self.street_ends)} points...")
            
            # Convert point to projected CRS
            point_proj = gpd.GeoSeries([point], crs='EPSG:4326').to_crs('EPSG:3857')[0]
            
            # Check distance to water
            distance = river_geom_proj.distance(point_proj)
            if distance < self.threshold_distance:
                # Check if point is too close to any existing points
                is_too_close = False
                for seen_point in seen_points:
                    if point_proj.distance(seen_point) < 25:  # 25 meters
                        is_too_close = True
                        break
                
                if not is_too_close:
                    # Create a new Point with properties
                    point_with_props = {
                        'type': 'Feature',
                        'geometry': point,  # Original WGS84 point for storage
                        'properties': {'distance_to_water': round(float(distance), 2)}
                    }
                    self.near_water.append(point_with_props)
                    seen_points.append(point_proj)  # Store projected point for distance checks
        
        self.logger.info("\nResults:")
        self.logger.info(f"Total street ends: {len(self.street_ends)}")
        self.logger.info(f"Unique street ends near water: {len(self.near_water)}")
        self.logger.info(f"Threshold distance: {self.threshold_distance} meters")
        
    def save_results(self, filename=None):
        """Save results to GeoJSON"""
        if filename is None:
            filename = self.geojson_file
        # Create GeoDataFrame from features with properties
        points_gdf = gpd.GeoDataFrame.from_features(
            self.near_water,
            crs="EPSG:4326"
        )
        points_gdf.to_file(filename, driver='GeoJSON')
        self.logger.info(f"\nResults saved to {filename}")
        
    def process(self):
        """Run the complete analysis"""
        self.get_data()
        if self.should_find_street_ends:
            self.find_street_ends()
        if self.should_enrich_data:
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
        
        # enrich data
        enricher = LocationEnricher()
        enricher.process_locations('street_ends_near_river.geojson')
        # Render the results
        output_file = f"street_ends_{city.split(',')[0].lower()}.html"
        renderer = StreetEndRenderer('street_ends_near_river.geojson', city)
        renderer.render(output_file)


