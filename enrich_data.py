import os
import json
import asyncio
import time

from dotenv import load_dotenv
import aiohttp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from utils.logging_config import setup_logger
from utils.osmnx_load import get_ox
ox = get_ox()
load_dotenv()

logger = setup_logger(__name__)

@dataclass
class LocationConfig:
    radius_km: float
    output_dir: str
    max_locations: int
    save_images: bool
    save_detailed_json: bool

class LocationEnricher:
    def __init__(self, config: LocationConfig):
        self.config = config
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.session = None
        self._setup_dirs()
        self.logger = setup_logger('location_enricher')
        
    def _setup_dirs(self):
        """Create necessary directories with proper permissions"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, 'data'), exist_ok=True)

    async def _init_session(self):
        """Initialize aiohttp session for concurrent requests"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def download_image(self, url: str, filename: str) -> Optional[str]:
        """Download image asynchronously"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    path = os.path.join(self.config.output_dir, 'images', filename)
                    with open(path, 'wb') as f:
                        f.write(await response.read())
                    return path
        except Exception as e:
            self.logger.error(f"Image download error: {str(e)}")
        return None

    async def get_location_images(self, lat: float, lon: float, location_id: str) -> Dict:
        """Get both street view and aerial images concurrently"""
        if not self.config.save_images:
            return {}

        street_url = f"https://maps.googleapis.com/maps/api/streetview?size=600x300&fov=120&location={lat},{lon}&key={self.google_api_key}"
        aerial_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=18&size=600x300&maptype=satellite&key={self.google_api_key}"
        
        tasks = [
            self.download_image(street_url, f"street_{location_id}.jpg"),
            self.download_image(aerial_url, f"aerial_{location_id}.jpg")
        ]
        results = await asyncio.gather(*tasks)
        
        return {
            'streetview': results[0],
            'aerial': results[1]
        }

    def calculate_population_metrics(self, buildings_gdf: gpd.GeoDataFrame, buffer_area: float) -> Dict:
        """Calculate advanced population and density metrics"""
        building_types = {
            'residential': {'floors': 2, 'density': 40},
            'apartments': {'floors': 4, 'density': 30},
            'commercial': {'floors': 3, 'density': 60},
            'mixed': {'floors': 3, 'density': 35},
            'default': {'floors': 2, 'density': 40}
        }

        def get_building_metrics(row):
            btype = row['building']
            metrics = building_types.get(btype, building_types['default'])
            area = row.geometry.area
            return {
                'floors': metrics['floors'],
                'area': area,
                'density': metrics['density']
            }

        building_metrics = buildings_gdf.apply(get_building_metrics, axis=1)
        
        total_floor_area = sum(m['area'] * m['floors'] for m in building_metrics)
        avg_density = sum(m['density'] for m in building_metrics) / len(building_metrics)
        
        estimated_population = total_floor_area / avg_density
        population_density = estimated_population / (buffer_area / 1000000)

        return {
            'population_density': round(population_density, 0),
            'total_floor_area': round(total_floor_area, 2),
            'building_count': len(buildings_gdf),
            'avg_building_density': round(avg_density, 2)
        }

    def calculate_greenspace_score(self, green_areas: gpd.GeoDataFrame, buffer_area: float) -> Dict:
        """Calculate greenspace score from nearby areas"""
        try:
            # Ensure we have the required columns, with safe defaults
            if 'leisure' not in green_areas.columns:
                self.logger.warning("No 'leisure' column found in green_areas, adding empty column")
                green_areas['leisure'] = None
            
            if 'landuse' not in green_areas.columns:
                self.logger.warning("No 'landuse' column found in green_areas, adding empty column")
                green_areas['landuse'] = None

            # Filter for specific green space types
            parks = green_areas[green_areas['leisure'].fillna(0).str.lower() == 'park']
            nature_reserves = green_areas[green_areas['leisure'].fillna(0).str.lower() == 'nature_reserve']
            gardens = green_areas[green_areas['leisure'].fillna(0).str.lower() == 'garden']
            
            # Calculate total area of specific green spaces only
            total_green_area = (
                parks.geometry.area.sum() +
                nature_reserves.geometry.area.sum() +
                gardens.geometry.area.sum()
            )
            green_percentage = (total_green_area / buffer_area) * 100  # Convert to percentage

            # Calculate average distance to nearest park (in meters)
            # Buffer radius is typically 1km = 1000m
            if not parks.empty:
                # Convert to projected CRS for accurate distance calculation
                parks_proj = parks.to_crs('EPSG:3857')
                center_point = Point(0, 0)  # Buffer is centered at 0,0
                center_gdf = gpd.GeoDataFrame(geometry=[center_point], crs='EPSG:4326').to_crs('EPSG:3857')
                distances = parks_proj.geometry.distance(center_gdf.geometry.iloc[0])
                nearest_park_distance = distances.min()
                
                # Score based on walking distance (higher score for closer parks)
                # Assumes 5 minutes walking = ~400m
                # Max score (100) for parks within 400m
                # Min score (0) for parks at or beyond 1000m
                distance_score = max(0, min(100, (1000 - nearest_park_distance) / 6))
            else:
                distance_score = 0

            # Additional filtering for other counts
            green_spaces = green_areas[green_areas['landuse'].fillna(0).str.lower().isin(['grass', 'recreation_ground'])]

            return {
                'park_count': len(parks),
                'nature_reserve_count': len(nature_reserves),
                'garden_count': len(gardens),
                'green_space_count': len(green_spaces),
                'total_green_areas': len(green_areas),
                'green_percentage': round(green_percentage, 2),
                'nearest_park_distance': round(nearest_park_distance if 'nearest_park_distance' in locals() else -1, 2),
                'park_distance_score': round(distance_score, 2),
                'greenspace_score': round((green_percentage + distance_score) / 2, 2)  # Combined score
            }
        except Exception as e:
            self.logger.error(f"Error calculating greenspace score: {str(e)}")
            return {
                'park_count': 0,
                'nature_reserve_count': 0,
                'garden_count': 0,
                'green_space_count': 0,
                'total_green_areas': 0,
                'greenspace_score': 0
            }

    def _get_waterway_info(self, point: Point, buffer_area=None) -> Dict:
        """
        Get basic information about the nearest waterway (name, type, width),
        prioritizing named water features.
        """
        try:
            # Create a buffer for water features (500m)
            water_buffer = point.buffer(0.5 / 111)  # 0.5km converted to degrees
            
            # More inclusive water feature tags
            waterway_tags = {
                'waterway': True,
                'natural': ['water', 'stream', 'river'],
                'water': True
            }
            
            logger.info(f"Searching for water features at {point.coords[0]}")
            
            try:
                water_features = ox.features_from_polygon(water_buffer, tags=waterway_tags)
            except Exception as e:
                logger.warning(f"First attempt failed, trying with simplified buffer: {str(e)}")
                water_buffer = water_buffer.simplify(0.0001)
                water_features = ox.features_from_polygon(water_buffer, tags=waterway_tags)
            
            if water_features.empty:
                logger.warning(f"No waterway features found at {point.coords[0]}")
                return {
                    'name': 'Unknown',
                    'type': 'Unknown',
                    'width': 'Unknown'
                }
            
            # Convert to projected CRS for accurate distance calculation
            water_features = water_features.to_crs('EPSG:3857')
            point_proj = gpd.GeoDataFrame(
                geometry=[point], 
                crs='EPSG:4326'
            ).to_crs('EPSG:3857')
            
            # Calculate distances to all features
            distances = water_features.geometry.distance(point_proj.geometry.iloc[0])
            
            # Check if 'name' column exists, if not add it with None values
            if 'name' not in water_features.columns:
                water_features['name'] = None
            
            # First try to find named features within the buffer
            named_features = water_features[water_features['name'].notna()]
            
            if not named_features.empty:
                # Get distances to named features only
                named_distances = distances[named_features.index]
                min_distance_idx = named_distances.astype(float).idxmin()
                nearest_feature = water_features.loc[min_distance_idx]
                logger.info("Found named water feature")
            else:
                # Fall back to nearest feature if no named features found
                min_distance_idx = distances.astype(float).idxmin()
                nearest_feature = water_features.loc[min_distance_idx]
                logger.info("No named features found, using nearest water feature")
            
            try:
                # Determine the water feature type
                feature_type = 'Unknown'
                for type_col in ['waterway', 'natural', 'water']:
                    if type_col in nearest_feature and pd.notna(nearest_feature[type_col]):
                        feature_type = nearest_feature[type_col]
                        break
                
                # Handle name - replace nan with 'Unknown'
                name = nearest_feature.get('name', 'Unknown')
                if pd.isna(name):
                    name = 'Unknown'
                
                # Extract width if available
                width = nearest_feature.get('width', 'Unknown')
                if pd.isna(width):
                    width = 'Unknown'
                
                waterway_info = {
                    'name': name,
                    'type': feature_type,
                    'width': width
                }
                
                logger.info(f"Found waterway: {waterway_info}")
                return waterway_info
                
            except Exception as e:
                logger.error(f"Error accessing nearest feature: {str(e)}")
                return {
                    'name': 'Unknown',
                    'type': 'Unknown',
                    'width': 'Unknown'
                }
                
        except Exception as e:
            logger.error(f"Error getting waterway info: {str(e)}", exc_info=True)
            return {
                'name': 'Unknown',
                'type': 'Unknown',
                'width': 'Unknown'
            }

    async def enrich_location(self, lat: float, lon: float, location_id: str) -> Dict:
        """Enrich location data with concurrent processing"""
        start_time = time.time()
        self.logger.info(f"Starting enrichment for location {location_id}")
        
        await self._init_session()
        
        point = Point(lon, lat)
        buffer_radius = self.config.radius_km * 1000
        buffer_area = point.buffer(buffer_radius / 111000)

        # Log each major step
        self.logger.info(f"Fetching images for location {location_id}")
        images_start = time.time()
        images_task = self.get_location_images(lat, lon, location_id)
        
        self.logger.info(f"Fetching OSM data for location {location_id}")
        osm_start = time.time()
        with ThreadPoolExecutor() as executor:
            osm_future = executor.submit(self._get_osm_data, buffer_area)
            waterway_future = executor.submit(self._get_waterway_info, point, buffer_area)

        images = await images_task
        self.logger.info(f"Images fetched in {time.time() - images_start:.2f} seconds")
        
        osm_data = osm_future.result()
        waterway_info = waterway_future.result()
        self.logger.info(f"OSM data fetched in {time.time() - osm_start:.2f} seconds")

        # Calculate metrics
        self.logger.info(f"Calculating metrics for location {location_id}")
        metrics_start = time.time()
        
        population_metrics = self.calculate_population_metrics(
            osm_data['buildings'],
            osm_data['buffer_area']
        )
        
        greenspace_score = self.calculate_greenspace_score(
            osm_data['green_areas'],
            osm_data['buffer_area']
        )
        
        self.logger.info(f"Metrics calculated in {time.time() - metrics_start:.2f} seconds")

        total_time = time.time() - start_time
        self.logger.info(f"Location {location_id} enriched in {total_time:.2f} seconds")

        data = {
            'location_id': location_id,
            'coordinates': {'lat': lat, 'lon': lon},
            'timestamp': datetime.now().isoformat(),
            'images': images,
            'population': population_metrics,
            'greenspace': greenspace_score,
            'waterway': waterway_info,
            'desirability_score': self._calculate_final_score(
                population_metrics,
                greenspace_score
            )
        }

        if self.config.save_detailed_json:
            self._save_location_data(data, location_id)

        return data

    def _calculate_final_score(
        self,
        population_metrics: Dict,
        greenspace_score: Dict
    ) -> float:
        """
        Calculate need-based score. Higher scores indicate greater need for waterfront access.
        Areas with high population density and low greenspace get higher scores.
        """
        # Normalize population density (0-100)
        # Assuming anything above 150 people per sq km is high density
        pop_density_score = min(100, (population_metrics['population_density'] / 10000) * 100)
        
        # Invert greenspace score (0-100)
        # Less greenspace = higher need score
        green_need_score = 100 - greenspace_score['greenspace_score']
        
        # Calculate combined need score
        # Higher weight on population density as it indicates more people would benefit
        weights = {
            'population': 0.6,  # Increased from 0.4
            'greenspace_need': 0.4  # Decreased from 0.6
        }
        
        need_score = (
            pop_density_score * weights['population'] +
            green_need_score * weights['greenspace_need']
        )
        
        # Add bonus points for very high density + very low greenspace combinations
        if pop_density_score > 80 and green_need_score > 80:
            need_score *= 1.2  # 20% bonus for high-need areas
        
        # Log the scoring components for debugging
        self.logger.info(
            f"Location scoring - "
            f"Population Density Score: {pop_density_score:.2f}, "
            f"Green Need Score: {green_need_score:.2f}, "
            f"Final Need Score: {need_score:.2f}"
        )
        
        return round(min(100, need_score), 2)  # Cap at 100

    async def process_locations(self, geojson_file: str) -> Dict:
        """Process locations with concurrent execution"""
        with open(geojson_file) as f:
            geojson = json.load(f)

        features = geojson['features'][:self.config.max_locations] if self.config.max_locations > 0 else geojson['features']
        
        # Create a semaphore to limit concurrent operations
        sem = asyncio.Semaphore(20)  # Adjust this number based on your needs (e.g., 5-20)

        async def process_feature(feature, idx):
            async with sem:  # This ensures only N operations run at once
                lon, lat = feature['geometry']['coordinates']
                return await self.enrich_location(lat, lon, str(idx))

        # Create tasks with controlled concurrency
        tasks = [
            process_feature(feature, idx)
            for idx, feature in enumerate(features)
        ]

        results = await asyncio.gather(*tasks)
        
        for feature, result in zip(features, results):
            feature['properties'].update(result)

        self._save_summary(geojson)
        return geojson

    def _get_osm_data(self, buffer_area) -> Dict:
        """Get OSM data for buildings and green areas"""
        # Get building data
        building_tags = {'building': True}
        buildings = ox.features_from_polygon(buffer_area, tags=building_tags)
        
        # Get green areas
        green_tags = {
            'leisure': ['park', 'garden', 'nature_reserve'],
            'landuse': ['grass', 'forest', 'recreation_ground'],
            'natural': ['wood', 'grassland', 'water']
        }
        green_areas = ox.features_from_polygon(buffer_area, tags=green_tags)
        
        # Convert to projected CRS for accurate area calculation
        buildings = buildings.to_crs('EPSG:3857') if not buildings.empty else buildings
        green_areas = green_areas.to_crs('EPSG:3857') if not green_areas.empty else green_areas
        buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_area], crs='EPSG:4326').to_crs('EPSG:3857')
        
        return {
            'buildings': buildings,
            'green_areas': green_areas,
            'buffer_area': buffer_gdf.geometry.area[0]
        }

    def _save_location_data(self, data: Dict, location_id: str):
        """Save location data with proper error handling"""
        try:
            filepath = os.path.join(self.config.output_dir, 'data', f'location_{location_id}.json')
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving location data: {str(e)}")

    def _save_summary(self, data: Dict):
        """Save summary with error handling"""
        try:
            filepath = os.path.join(self.config.output_dir, 'summary.json')
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving summary: {str(e)}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

if __name__ == "__main__":
    config = LocationConfig(
        radius_km=1.0,
        output_dir="location_data",
        max_locations=-1,
        save_images=True,
        save_detailed_json=True
    )

    enricher = LocationEnricher(config)
    asyncio.run(enricher.process_locations("global_analysis/US/Skokie/street_ends.geojson"))
    enricher.cleanup()