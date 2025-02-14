import os
import json
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from datetime import datetime
from dotenv import load_dotenv
from utils.logging_config import setup_logger
from utils.osmnx_load import get_ox
ox = get_ox()

# Create logger for this module
logger = setup_logger(__name__)

load_dotenv()

class LocationEnricher:
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.census_api_key = os.getenv('CENSUS_API_KEY')
        self.radius_miles = float(os.getenv('RADIUS_MILES', 1))
        self.output_dir = os.getenv('OUTPUT_DIR', 'location_data')
        self.max_locations = int(os.getenv('MAX_LOCATIONS', -1))
        self.save_images = os.getenv('SAVE_IMAGES', 'true').lower() == 'true'
        self.save_detailed_json = os.getenv('SAVE_DETAILED_JSON', 'true').lower() == 'true'
        
        if not self.google_api_key or not self.census_api_key:
            raise ValueError("Missing required API keys in .env file")
            
        os.makedirs(self.output_dir, exist_ok=True)

    def download_streetview(self, lat, lon, location_id):
        """Download Google Street View image"""
        url = f"https://maps.googleapis.com/maps/api/streetview?size=600x300&fov=120&location={lat},{lon}&key={self.google_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            filename = f"{self.output_dir}/streetview_{location_id}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
        return None

    def download_aerial(self, lat, lon, location_id):
        """Download Google Maps Static aerial image"""
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=18&size=600x300&maptype=satellite&key={self.google_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            filename = f"{self.output_dir}/aerial_{location_id}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
        return None

    def get_census_data(self, lat, lon):
        """Get demographic data from Census API"""
        try:
            # First get FIPS codes using Census Geocoding API
            geocode_url = f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
            params = {
                "x": lon,
                "y": lat,
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "layers": "Census Tracts",
                "format": "json"
            }
            
            response = requests.get(geocode_url, params=params)
            if response.status_code != 200:
                return None
            
            result = response.json()
            tract_info = result['result']['geographies']['Census Tracts'][0]
            
            # Get census data using FIPS codes
            census_url = "https://api.census.gov/data/2020/acs/acs5"
            census_params = {
                "get": "NAME,B19013_001E,B01003_001E",  # Name, Median Income, Population
                "for": f"tract:{tract_info['TRACT']}",
                "in": f"state:{tract_info['STATE']} county:{tract_info['COUNTY']}",
                "key": self.census_api_key
            }
            
            census_response = requests.get(census_url, params=census_params)
            if census_response.status_code == 200:
                data = census_response.json()
                # Convert array response to dictionary
                headers = data[0]
                values = data[1]
                return dict(zip(headers, values))
            
        except Exception as e:
            logger.error(f"Census API error: {str(e)}")
        
        return None

    def calculate_greenspace(self, lat, lon):
        """Calculate available greenspace within 1 mile radius"""
        try:
            # Convert 1 mile to meters
            radius = self.radius_miles * 1609.34
            
            # Create a point and buffer
            point = Point(lon, lat)
            buffer_area = point.buffer(radius / 111000)  # approximate degree conversion
            
            # Get parks and green areas
            tags = {
                'leisure': ['park', 'garden', 'nature_reserve'],
                'landuse': ['grass', 'forest', 'recreation_ground'],
                'natural': ['wood', 'grassland', 'water']
            }
            
            green_areas = ox.features_from_polygon(buffer_area, tags=tags)
            if len(green_areas) > 0:
                # Convert to projected CRS for accurate area calculation
                green_areas = green_areas.to_crs('EPSG:3857')  # Web Mercator projection
                buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_area], crs='EPSG:4326').to_crs('EPSG:3857')
                
                total_area = green_areas.geometry.area.sum()
                buffer_total_area = buffer_gdf.geometry.area[0]
                green_percentage = (total_area / buffer_total_area) * 100
                
                return {
                    'green_percentage': green_percentage,
                    'num_parks': len(green_areas[green_areas['leisure'] == 'park']),
                    'total_green_areas': len(green_areas)
                }
            return None
        except Exception as e:
            logger.error(f"Error calculating greenspace: {str(e)}")
            return None

    def get_waterway_info(self, lat, lon):
        """Get information about nearby waterways"""
        point = Point(lon, lat)
        buffer_area = point.buffer(0.001)  # Small buffer to find closest waterway
        
        tags = {
            'waterway': True,
            'natural': ['water', 'stream', 'riverbank']
        }
        
        waterways = ox.features_from_polygon(buffer_area, tags=tags)
        if len(waterways) > 0:
            closest = waterways.iloc[0]
            return {
                'name': closest.get('name', 'Unknown'),
                'type': closest.get('waterway', closest.get('natural', 'Unknown')),
                'width': closest.get('width', 'Unknown')
            }
        return None

    def calculate_desirability_score(self, data):
        """Calculate a desirability score based on various factors, prioritizing high density with low greenspace"""
        score = 0
        weights = {
            'population_density': 0.7,  # Increased weight for density
            'green_percentage': 0.3,    # Lower weight for greenspace, and we'll invert it
        }
        
        # Normalize and weight each factor
        if 'population' in data and 'population_density' in data['population']:
            # Normalize density (assuming 15000 people/km² as upper bound)
            # Higher density = higher score
            density_score = min(data['population']['population_density'] / 15000, 1)
            score += density_score * weights['population_density']

        if 'greenspace' in data and 'green_percentage' in data['greenspace']:
            # Invert green percentage so less green = higher score
            # 100% green = 0 score, 0% green = 1 score
            green_score = 1 - (data['greenspace']['green_percentage'] / 100)
            score += green_score * weights['green_percentage']
        
        return min(score * 100, 100)  # Return score out of 100

    def get_population_density(self, lat, lon):
        """Get population density data using WorldPop and OSM data for international coverage"""
        try:
            # Create a small buffer around the point (approximately 1km)
            point = Point(lon, lat)
            buffer_degrees = 0.01  # roughly 1km at equator
            buffer_area = point.buffer(buffer_degrees)
            
            # Get built-up area from OSM to understand urban density
            tags = {
                'building': True,
                'landuse': ['residential', 'commercial', 'mixed']
            }
            
            logger.info(f"Fetching OSM data for location: {lat}, {lon}")
            
            # Get area features from OSM
            area_data = ox.features_from_polygon(buffer_area, tags=tags)
            
            # Calculate built-up percentage
            if len(area_data) > 0:
                # Convert to projected CRS for accurate area calculation
                area_data = area_data.to_crs('EPSG:3857')
                buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_area], crs='EPSG:4326').to_crs('EPSG:3857')
                
                built_up_area = area_data.geometry.area.sum()
                total_area = buffer_gdf.geometry.area[0]
                built_up_percentage = (built_up_area / total_area) * 100
                logger.info(f"Built-up percentage: {built_up_percentage:.2f}%")
            else:
                built_up_percentage = 0
                logger.info("No buildings found in area")
            
            # Estimate population density based on built-up percentage and OSM data
            if built_up_percentage > 80:
                population_density = 15000  # high density urban
                density_category = "high density urban"
            elif built_up_percentage > 50:
                population_density = 8000   # medium density urban
                density_category = "medium density urban"
            elif built_up_percentage > 20:
                population_density = 3000   # low density urban
                density_category = "low density urban"
            else:
                population_density = 500    # rural/suburban
                density_category = "rural/suburban"
            
            logger.info(f"Estimated density category: {density_category}")
            
            # Try to get place context, but don't fail if not found
            place_type = 'unknown'
            try:
                context_tags = ox.features_from_point(
                    (lat, lon), 
                    tags={'place': True}, 
                    dist=1000
                )
                
                if len(context_tags) > 0:
                    place_types = context_tags['place'].dropna().unique()
                    if len(place_types) > 0:
                        place_type = place_types[0]
                        logger.info(f"Identified place type: {place_type}")
            except Exception as e:
                logger.warning(f"Could not determine place type: {str(e)}")
            
            return {
                'population_density': round(population_density, 2),  # people per square km
                'built_up_percentage': round(built_up_percentage, 2),
                'density_category': density_category,
                'place_type': place_type,
                'data_source': 'OSM',
                'location': {
                    'lat': lat,
                    'lon': lon
                }
            }
            
        except Exception as e:
            logger.error(f"Population density estimation error: {str(e)}", exc_info=True)
            return None

    def enrich_location(self, lat, lon, location_id):
        """Enrich a single location with all available data"""
        data = {
            'location_id': location_id,
            'latitude': lat,
            'longitude': lon,
            'timestamp': datetime.now().isoformat(),
            'images': {}
        }
        
        # Download images if enabled
        if self.save_images:
            streetview = self.download_streetview(lat, lon, location_id)
            aerial = self.download_aerial(lat, lon, location_id)
            if streetview:
                data['images']['streetview'] = streetview
            if aerial:
                data['images']['aerial'] = aerial
            
        # Get population density data
        population_data = self.get_population_density(lat, lon)
        if population_data:
            data['population'] = population_data
            
        # Get greenspace data
        greenspace = self.calculate_greenspace(lat, lon)
        if greenspace:
            data['greenspace'] = greenspace
            
        # Get waterway info
        waterway = self.get_waterway_info(lat, lon)
        if waterway:
            data['waterway'] = waterway
            
        # Calculate desirability score
        data['desirability_score'] = self.calculate_desirability_score(data)
        
        # Save to JSON if enabled
        if self.save_detailed_json:
            output_file = f"{self.output_dir}/location_{location_id}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        return data

    def process_locations(self, geojson_file):
        """Process all locations from a GeoJSON file"""
        points = gpd.read_file(geojson_file)
        results = []
        
        for idx, point in points.iterrows():
            if self.max_locations > 0 and idx >= self.max_locations:
                break
                
            lat, lon = point.geometry.y, point.geometry.x
            logger.info(f"Processing location {idx}: {lat}, {lon}")
            
            result = self.enrich_location(lat, lon, idx)
            if result:
                results.append(result)
            
        # Save summary
        with open(f"{self.output_dir}/summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

# Example usage:
if __name__ == "__main__":
    google_api_key = "YOUR_GOOGLE_API_KEY"
    census_api_key = "YOUR_CENSUS_API_KEY"
    
    enricher = LocationEnricher()
    results = enricher.process_locations('street_ends_near_river.geojson') 