import os
import json
import asyncio
import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

from dotenv import load_dotenv
import aiohttp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

from utils.logging_config import setup_logger
from utils.osmnx_load import get_ox
ox = get_ox()
load_dotenv()

logger = setup_logger(__name__)

def get_city_info(city_name: str, country_code: str = None) -> Dict:
    """
    Get detailed information about a city using GeoNames database.
    
    Args:
        city_name (str): Name of the city
        country_code (str, optional): Two-letter country code (ISO-3166)
    
    Returns:
        Dict: City information including:
            - name: Official name
            - latitude: City center latitude
            - longitude: City center longitude
            - population: Population count
            - country_code: Two-letter country code
            - timezone: City timezone
            - admin_codes: Administrative region codes
            - geometry: City boundary polygon (if available via OSMnx)
    """
    try:
        # Download and read GeoNames data
        url = "http://download.geonames.org/export/dump/cities15000.zip"
        df = pd.read_csv(url, compression='zip', sep='\t', header=None,
                        names=['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 
                               'longitude', 'feature_class', 'feature_code', 'country_code', 
                               'cc2', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code',
                               'population', 'elevation', 'dem', 'timezone', 'modification_date'])
        
        # Filter for the city
        city_filter = df['name'].str.lower() == city_name.lower()
        if country_code:
            city_filter &= df['country_code'] == country_code.upper()
        city_filter &= df['feature_class'] == 'P'  # Populated places only
        
        city_data = df[city_filter]
        
        if city_data.empty:
            logger.warning(f"City not found: {city_name}" + 
                         f" (country: {country_code})" if country_code else "")
            return None
            
        # Get the most populous match if multiple exist
        city_info = city_data.sort_values('population', ascending=False).iloc[0]
        
        # Try to get city geometry from OSMnx
        try:
            gdf = ox.geocode_to_gdf(f"{city_name}, {city_info['country_code']}")
            geometry = gdf.geometry.iloc[0] if not gdf.empty else None
        except Exception as e:
            logger.warning(f"Could not fetch city geometry: {str(e)}")
            geometry = None
        
        # Construct result dictionary
        result = {
            'name': city_info['name'],
            'city_name': f"{city_info['name']}, {city_info['country_code']}",
            'latitude': float(city_info['latitude']),
            'longitude': float(city_info['longitude']),
            'population': int(city_info['population']),
            'country_code': city_info['country_code'],
            'timezone': city_info['timezone'],
            'admin_codes': {
                'admin1': city_info['admin1_code'],
                'admin2': city_info['admin2_code'],
                'admin3': city_info['admin3_code'],
                'admin4': city_info['admin4_code']
            },
            'analysis_date': datetime.now().isoformat(),
            'geometry': geometry
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting city information: {str(e)}")
        return None

@dataclass
class LocationConfig:
    radius_km: float
    output_dir: str
    max_locations: int
    save_images: bool
    save_detailed_json: bool
    batch_size: int = 10  # Process locations in batches
    cache_osm_data: bool = True  # Enable OSM data caching
    concurrency_limit: int = 5  # Number of concurrent location processings
    use_multithreading: bool = False  # Whether to use multithreading for data fetching


class LocationEnricher:
    def __init__(self, config: LocationConfig):
        self.config = config
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.session = None
        self._setup_dirs()
        self.logger = setup_logger('location_enricher')
        
        # Cache for OSM data to avoid redundant queries
        self.osm_cache = {}
        self.cache_hits = 0
        
    def _setup_dirs(self):
        """Create necessary directories with proper permissions"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, 'cache'), exist_ok=True)

    async def _init_session(self):
        """Initialize aiohttp session for concurrent requests"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self.config.concurrency_limit)
            )

    async def download_image(self, url: str, filename: str) -> Optional[str]:
        """Download image asynchronously"""
        try:
            cache_path = os.path.join(self.config.output_dir, 'images', filename)
            # Skip if already exists
            if os.path.exists(cache_path):
                return cache_path
                
            async with self.session.get(url) as response:
                if response.status == 200:
                    with open(cache_path, 'wb') as f:
                        f.write(await response.read())
                    return cache_path
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
        if buildings_gdf.empty:
            return {
                'population_density': 0,
                'total_floor_area': 0,
                'building_count': 0,
                'avg_building_density': 0
            }
            
        building_types = {
            'residential': {'floors': 2, 'density': 40},
            'apartments': {'floors': 4, 'density': 30},
            'commercial': {'floors': 3, 'density': 60},
            'mixed': {'floors': 3, 'density': 35},
            'default': {'floors': 2, 'density': 40}
        }

        def get_building_metrics(row):
            btype = row.get('building', 'default')
            if not isinstance(btype, str):
                btype = 'default'
                
            metrics = building_types.get(btype, building_types['default'])
            area = row.geometry.area
            return {
                'floors': metrics['floors'],
                'area': area,
                'density': metrics['density']
            }

        building_metrics = buildings_gdf.apply(get_building_metrics, axis=1)
        
        total_floor_area = sum(m['area'] * m['floors'] for m in building_metrics)
        avg_density = sum(m['density'] for m in building_metrics) / max(1, len(building_metrics))
        
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
            if green_areas.empty:
                return {
                    'park_count': 0,
                    'nature_reserve_count': 0,
                    'garden_count': 0,
                    'green_space_count': 0,
                    'total_green_areas': 0,
                    'green_percentage': 0,
                    'nearest_park_distance': -1,
                    'park_distance_score': 0,
                    'greenspace_score': 0
                }
            
            # Ensure we have the required columns, with safe defaults
            if 'leisure' not in green_areas.columns:
                green_areas['leisure'] = None
            
            if 'landuse' not in green_areas.columns:
                green_areas['landuse'] = None

            # Filter for specific green space types
            parks = green_areas[green_areas['leisure'].fillna('').astype(str).str.lower() == 'park']
            nature_reserves = green_areas[green_areas['leisure'].fillna('').astype(str).str.lower() == 'nature_reserve']
            gardens = green_areas[green_areas['leisure'].fillna('').astype(str).str.lower() == 'garden']
            
            # Calculate total area of specific green spaces only
            total_green_area = (
                parks.geometry.area.sum() +
                nature_reserves.geometry.area.sum() +
                gardens.geometry.area.sum()
            )
            green_percentage = (total_green_area / buffer_area) * 100  # Convert to percentage

            # Calculate average distance to nearest park (in meters)
            if not parks.empty:
                parks_proj = parks.to_crs('EPSG:3857')
                center_point = Point(0, 0)  # Buffer is centered at 0,0
                center_gdf = gpd.GeoDataFrame(geometry=[center_point], crs='EPSG:4326').to_crs('EPSG:3857')
                distances = parks_proj.geometry.distance(center_gdf.geometry.iloc[0])
                nearest_park_distance = distances.min()
                
                # Score based on walking distance
                distance_score = max(0, min(100, (1000 - nearest_park_distance) / 6))
            else:
                nearest_park_distance = -1
                distance_score = 0

            # Additional filtering for other counts
            green_spaces = green_areas[green_areas['landuse'].fillna('').astype(str).str.lower().isin(['grass', 'recreation_ground'])]

            return {
                'park_count': len(parks),
                'nature_reserve_count': len(nature_reserves),
                'garden_count': len(gardens),
                'green_space_count': len(green_spaces),
                'total_green_areas': len(green_areas),
                'green_percentage': round(green_percentage, 2),
                'nearest_park_distance': round(nearest_park_distance, 2),
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

    def _get_waterway_info(self, point: Point, fetch_radius_km=0.25) -> Dict:
        """Get waterway info with improved error handling and caching"""
        try:
            # Use a stable cache key based on rounded coordinates (for nearby points)
            lat, lon = point.y, point.x
            cache_key = f"water_{round(lat, 4)}_{round(lon, 4)}"
            
            if cache_key in self.osm_cache:
                self.cache_hits += 1
                return self.osm_cache[cache_key]
            
            # Create a simplified buffer for water features
            buffer_degrees = fetch_radius_km / 111  # Convert km to approximate degrees
            water_buffer = point.buffer(buffer_degrees).simplify(0.0001)
            
            # More inclusive water feature tags
            waterway_tags = {
                'waterway': True,
                'natural': ['water', 'stream', 'river'],
                'water': True
            }
            
            try:
                water_features = ox.features_from_polygon(water_buffer, tags=waterway_tags)
            except Exception as e:
                logger.warning(f"Error fetching water features: {str(e)}")
                return {
                    'name': 'Unknown',
                    'type': 'Unknown',
                    'width': 'Unknown'
                }
            
            if water_features.empty:
                logger.warning(f"No waterway features found at {point.coords[0]}")
                result = {
                    'name': 'Unknown',
                    'type': 'Unknown',
                    'width': 'Unknown'
                }
                self.osm_cache[cache_key] = result
                return result
            
            # Convert to projected CRS for accurate distance calculation
            water_features = water_features.to_crs('EPSG:3857')
            point_proj = gpd.GeoDataFrame(
                geometry=[point], 
                crs='EPSG:4326'
            ).to_crs('EPSG:3857')
            
            # Calculate distances to all features
            distances = water_features.geometry.distance(point_proj.geometry.iloc[0])
            
            # Get the nearest feature first
            min_distance_idx = distances.astype(float).idxmin()
            nearest_feature = water_features.loc[min_distance_idx]
            
            # Try to find a named feature within 1.5x the distance to the nearest feature
            min_distance = distances[min_distance_idx]
            nearby_features = water_features[distances <= min_distance * 1.5]
            
            # Look for named features among nearby features
            named_features = nearby_features[
                nearby_features.apply(
                    lambda x: any(
                        isinstance(v, str) and len(v.strip()) > 0 
                        for v in [x.get('name'), x.get('waterway'), x.get('water')]
                        if v is not None
                    ),
                    axis=1
                )
            ]
            
            if not named_features.empty:
                # Use the nearest named feature
                named_distances = distances[named_features.index]
                min_named_idx = named_distances.astype(float).idxmin()
                feature_to_use = water_features.loc[min_named_idx]
            else:
                feature_to_use = nearest_feature
            
            # Determine the water feature type
            feature_type = 'Unknown'
            for type_col in ['waterway', 'natural', 'water']:
                if type_col in feature_to_use and pd.notna(feature_to_use[type_col]):
                    feature_type = feature_to_use[type_col]
                    break
            
            # Get name from various possible columns
            name = 'Unknown'
            for name_col in ['name', 'waterway', 'water']:
                if name_col in feature_to_use and pd.notna(feature_to_use[name_col]):
                    name_val = feature_to_use[name_col]
                    if isinstance(name_val, str) and len(name_val.strip()) > 0:
                        name = name_val
                        break
            
            # Extract width if available
            width = feature_to_use.get('width', 'Unknown')
            if pd.isna(width):
                width = 'Unknown'
            
            waterway_info = {
                'name': name,
                'type': feature_type,
                'width': width
            }
            
            # Cache the result
            self.osm_cache[cache_key] = waterway_info
            return waterway_info
                
        except Exception as e:
            logger.error(f"Error getting waterway info: {str(e)}", exc_info=True)
            return {
                'name': 'Unknown',
                'type': 'Unknown',
                'width': 'Unknown'
            }

    def _get_osm_data_cached(self, buffer_area, lat, lon) -> Dict:
        """Get OSM data for buildings and green areas with caching"""
        # Use a cache key based on the center of the buffer area
        cache_key = f"osm_{round(lat, 3)}_{round(lon, 3)}_{self.config.radius_km}"
        
        if cache_key in self.osm_cache and self.config.cache_osm_data:
            self.cache_hits += 1
            return self.osm_cache[cache_key]
        
        # Simplify the buffer geometry for OSM querying
        simplified_buffer = buffer_area.simplify(0.0001)
        
        try:
            # Get building data
            building_tags = {'building': True}
            buildings = ox.features_from_polygon(simplified_buffer, tags=building_tags)
            
            # Get green areas
            green_tags = {
                'leisure': ['park', 'garden', 'nature_reserve'],
                'landuse': ['grass', 'forest', 'recreation_ground'],
                'natural': ['wood', 'grassland', 'water']
            }
            green_areas = ox.features_from_polygon(simplified_buffer, tags=green_tags)
            
            # Convert to projected CRS for accurate area calculation
            buildings = buildings.to_crs('EPSG:3857') if not buildings.empty else gpd.GeoDataFrame(geometry=[], crs='EPSG:3857')
            green_areas = green_areas.to_crs('EPSG:3857') if not green_areas.empty else gpd.GeoDataFrame(geometry=[], crs='EPSG:3857')
            buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_area], crs='EPSG:4326').to_crs('EPSG:3857')
            
            result = {
                'buildings': buildings,
                'green_areas': green_areas,
                'buffer_area': buffer_gdf.geometry.area[0]
            }
            
            # Cache the result
            if self.config.cache_osm_data:
                self.osm_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching OSM data: {str(e)}")
            # Return empty data frames
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs='EPSG:3857')
            buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_area], crs='EPSG:4326').to_crs('EPSG:3857')
            
            result = {
                'buildings': empty_gdf,
                'green_areas': empty_gdf,
                'buffer_area': buffer_gdf.geometry.area[0]
            }
            
            return result

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
        pop_density_score = min(100, (population_metrics['population_density'] / 10000) * 100)
        
        # Invert greenspace score (0-100)
        green_need_score = 100 - greenspace_score['greenspace_score']
        
        # Calculate combined need score
        weights = {
            'population': 0.6,
            'greenspace_need': 0.4
        }
        
        need_score = (
            pop_density_score * weights['population'] +
            green_need_score * weights['greenspace_need']
        )
        
        # Add bonus points for very high density + very low greenspace combinations
        if pop_density_score > 80 and green_need_score > 80:
            need_score *= 1.2  # 20% bonus for high-need areas
        
        return round(min(100, need_score), 2)  # Cap at 100

    async def enrich_location(self, lat: float, lon: float, location_id: str) -> Dict:
        """Enrich location data with concurrent processing"""
        start_time = time.time()
        
        point = Point(lon, lat)
        buffer_radius = self.config.radius_km * 1000
        buffer_area = point.buffer(buffer_radius / 111000)  # Convert to degrees approximately

        # Launch tasks concurrently
        images_task = self.get_location_images(lat, lon, location_id)
        
        # Get OSM data - either with threading or sequentially
        if self.config.use_multithreading:
            # Use thread pool for concurrent processing
            with ThreadPoolExecutor() as executor:
                osm_future = executor.submit(self._get_osm_data_cached, buffer_area, lat, lon)
                waterway_future = executor.submit(self._get_waterway_info, point)
            
            # Wait for all tasks to complete
            images = await images_task
            osm_data = osm_future.result()
            waterway_info = waterway_future.result()
        else:
            # Process sequentially
            images = await images_task
            osm_data = self._get_osm_data_cached(buffer_area, lat, lon)
            waterway_info = self._get_waterway_info(point)
        
        # Calculate metrics
        population_metrics = self.calculate_population_metrics(
            osm_data['buildings'],
            osm_data['buffer_area']
        )
        
        greenspace_score = self.calculate_greenspace_score(
            osm_data['green_areas'],
            osm_data['buffer_area']
        )

        total_time = time.time() - start_time
        self.logger.info(f"Location {location_id} enriched in {total_time:.2f} seconds (cache hits: {self.cache_hits})")

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

    def _group_locations_by_proximity(self, features) -> List[List[Dict]]:
        """Group locations that are close to each other to optimize OSM queries"""
        # Convert km to approximate degrees (at the equator)
        proximity_threshold = self.config.radius_km * 2 / 111
        
        # Init groups
        groups = []
        used_indices = set()
        
        for i, feature in enumerate(features):
            if i in used_indices:
                continue
                
            lon1, lat1 = feature['geometry']['coordinates']
            point1 = Point(lon1, lat1)
            
            # Start a new group
            current_group = [feature]
            used_indices.add(i)
            
            # Find nearby points
            for j, other_feature in enumerate(features):
                if j in used_indices or i == j:
                    continue
                    
                lon2, lat2 = other_feature['geometry']['coordinates']
                point2 = Point(lon2, lat2)
                
                # If points are close, add to current group
                distance = point1.distance(point2)
                if distance < proximity_threshold:
                    current_group.append(other_feature)
                    used_indices.add(j)
                    
            groups.append(current_group)
            
        return groups

    async def process_group(self, group, start_idx):
        """Process a group of nearby locations"""
        tasks = []
        for i, feature in enumerate(group):
            lon, lat = feature['geometry']['coordinates']
            location_id = str(start_idx + i)
            tasks.append(self.enrich_location(lat, lon, location_id))
            
        results = await asyncio.gather(*tasks)
        return results

    async def process_locations(self, geojson_file: str) -> Dict:
        """
        Process locations from a GeoJSON file
        
        Args:
            geojson_file: Path to GeoJSON file with locations
            
        Returns:
            Dict: Updated GeoJSON with enriched properties
        """
        with open(geojson_file) as f:
            geojson = json.load(f)

        # Extract city name from the file path
        # Assumes path structure like "global_analysis/US/Chicago/street_ends.geojson"
        path_parts = os.path.normpath(geojson_file).split(os.sep)
        try:
            country_code = path_parts[-3]  # e.g., "US"
            city_name = path_parts[-2]     # e.g., "Chicago"
            
            # Get city information
            city_info = get_city_info(city_name, country_code)
            if city_info:
                # Add city info directly to FeatureCollection properties
                if 'properties' not in geojson:
                    geojson['properties'] = {}
                
                # Remove geometry from city info to avoid redundancy
                if 'geometry' in city_info:
                    del city_info['geometry']
                    
                geojson['properties'].update(city_info)
                
        except (IndexError, Exception) as e:
            self.logger.warning(f"Could not extract city information from path: {str(e)}")

        features = geojson['features'][:self.config.max_locations] if self.config.max_locations > 0 else geojson['features']
        self.logger.info(f"Processing {len(features)} locations from {geojson_file}")
        
        # Reset cache statistics
        self.cache_hits = 0
        
        # Initialize session
        await self._init_session()
        
        # Group nearby locations for batch processing
        location_groups = self._group_locations_by_proximity(features)
        self.logger.info(f"Grouped {len(features)} locations into {len(location_groups)} proximity groups")
        
        # Process groups with controlled concurrency
        all_results = []
        location_index = 0
        
        # Determine concurrency approach based on configuration
        if self.config.use_multithreading:
            self.logger.info(f"Using concurrent processing with limit of {self.config.concurrency_limit}")
            sem = asyncio.Semaphore(self.config.concurrency_limit)
            
            async def process_group_with_semaphore(group, start_idx):
                async with sem:
                    return await self.process_group(group, start_idx)
            
            # Create tasks for each group
            tasks = []
            for group in location_groups:
                tasks.append(process_group_with_semaphore(group, location_index))
                location_index += len(group)
            
            # Execute all tasks
            group_results = await asyncio.gather(*tasks)
            
            # Flatten results
            for group_result in group_results:
                all_results.extend(group_result)
        else:
            # Process groups sequentially
            self.logger.info("Using sequential processing")
            for group in tqdm(location_groups, desc="Processing location groups"):
                group_result = await self.process_group(group, location_index)
                all_results.extend(group_result)
                location_index += len(group)
        
        # Update feature properties with results
        for feature, result in zip(features, all_results):
            feature['properties'].update(result)
        # Preserve existing properties while adding new ones
        existing_props = geojson['properties'].copy()
        geojson['properties'] = {
            **existing_props,
            'total_street_ends': len(features),
            'analysis_date': datetime.now().isoformat()
        }
        self.logger.info(f"Processing complete. Cache hits: {self.cache_hits}")
        self._save_summary(geojson)
        return geojson

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
    # Example 1: Sequential processing (default)
    config = LocationConfig(
        radius_km=1.0,
        output_dir="global_analysis/US/Chicago",
        max_locations=2,
        save_images=False,
        save_detailed_json=True,
        batch_size=10,
        cache_osm_data=True,
        concurrency_limit=6,
        use_multithreading=True  # Default is False, but explicitly set here for clarity
    )

    enricher = LocationEnricher(config)
    asyncio.run(enricher.process_locations("global_analysis/US/Chicago/street_ends.geojson"))
    asyncio.run(enricher.cleanup())
    
    # Example 2: Concurrent processing (explicitly enabled)
    # config_mt = LocationConfig(
    #     radius_km=1.0,
    #     output_dir="location_data/US/Chicago",
    #     max_locations=10,
    #     save_images=False,
    #     save_detailed_json=True,
    #     batch_size=10,
    #     cache_osm_data=True,
    #     concurrency_limit=5,  # Higher limit for concurrent processing
    #     use_multithreading=True  # Enable multithreading
    # )
    #
    # enricher_mt = LocationEnricher(config_mt)
    # asyncio.run(enricher_mt.process_locations("location_data/US/Chicago/street_ends.geojson"))
    # asyncio.run(enricher_mt.cleanup())