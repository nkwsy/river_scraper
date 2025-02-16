import pandas as pd
import time
from tqdm import tqdm
import logging
from pathlib import Path
from main import StreetEndFinder
from render_map import StreetEndRenderer
from enrich_data import LocationEnricher, LocationConfig
import json
from google_maps_export import GoogleMapsExporter
import geopandas as gpd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils.logging_config import setup_logger
from datetime import datetime
from functools import wraps
from typing import Callable
import asyncio

def log_timing(func: Callable) -> Callable:
    """Decorator to log function execution time"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        self.logger.info(f"Starting {func.__name__}")
        
        result = func(self, *args, **kwargs)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Completed {func.__name__} in {elapsed_time:.2f} seconds")
        return result
    return wrapper

class GlobalCityAnalyzer:
    def __init__(self, **kwargs):
        self.output_dir = Path('global_analysis')
        self.output_dir.mkdir(exist_ok=True)
        self.logger = setup_logger('global_analyzer')
        self.maps_exporter = GoogleMapsExporter()
        self.analyzed_cities = []  # Add this to track cities
        self.find_street_ends = kwargs.get('find_street_ends', True)
        self.enrich_data = kwargs.get('enrich_data', True)
        self.start_time = None

    def get_top_cities(self):
        """Get list of top 1000 cities by population"""
        try:
            # Using geonames API for city data
            url = "http://download.geonames.org/export/dump/cities15000.zip"
            df = pd.read_csv(url, compression='zip', sep='\t', header=None,
                           names=['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 
                                'longitude', 'feature_class', 'feature_code', 'country_code', 
                                'cc2', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code',
                                'population', 'elevation', 'dem', 'timezone', 'modification_date'])
            
            # Filter and sort by population
            cities = df[df['feature_class'] == 'P'].sort_values('population', ascending=False)
            return cities.head(1000)
            
        except Exception as e:
            logging.error(f"Error getting city list: {str(e)}")
            return None

    def get_top_us_cities(self, num_cities=20):
        """Get list of top 20 US cities by population"""
        try:
            url = "http://download.geonames.org/export/dump/cities15000.zip"
            df = pd.read_csv(url, compression='zip', sep='\t', header=None,
                           names=['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 
                                'longitude', 'feature_class', 'feature_code', 'country_code', 
                                'cc2', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code',
                                'population', 'elevation', 'dem', 'timezone', 'modification_date'])
            
            # Filter for US cities and sort by population
            us_cities = df[(df['feature_class'] == 'P') & (df['country_code'] == 'US')].sort_values('population', ascending=False)
            return us_cities.head(20)
            
        except Exception as e:
            logging.error(f"Error getting US city list: {str(e)}")
            return None

    @log_timing
    def analyze_city_stage1(self, city_row):
        """
        Stage 1: Only generate the raw street ends near water for this city.
        No enrichment is done here.
        """
        try:
            city_name = f"{city_row['name']}, {city_row['country_code']}"
            self.logger.info(f"Processing {city_name}")
            
            city_dir = self.output_dir / city_row['country_code'] / city_row['name'].replace('/', '_')
            city_dir.mkdir(parents=True, exist_ok=True)

            # Path for the output file
            geojson_path = city_dir / 'street_ends.geojson'
            
            if self.find_street_ends:
                self.logger.info(f"Finding street ends for {city_name}")
                start = time.time()
                finder = StreetEndFinder(city_name, threshold_distance=10, geojson_file=geojson_path)
                water_features, near_water = finder.process()
                self.logger.info(f"Found {len(near_water) if near_water else 0} street ends in {time.time() - start:.2f} seconds")
                
                if not near_water or len(near_water) == 0:
                    self.logger.warning(f"No water-adjacent street ends found for {city_name}")
                    return None

            self.logger.info(f"Stage 1 complete for {city_name}")
            return {"city_name": city_name, "stage1_geojson": str(geojson_path)}
        
        except Exception as e:
            self.logger.error(f"Error in Stage 1 for {city_name}: {str(e)}", exc_info=True)
            return None

    @log_timing
    async def analyze_city_stage2_async(self, city_row):
        """Async version of analyze_city_stage2"""
        try:
            city_name = f"{city_row['name']}, {city_row['country_code']}"
            self.logger.info(f"Starting enrichment for {city_name}")
            
            city_dir = self.output_dir / city_row['country_code'] / city_row['name'].replace('/', '_')
            geojson_path = city_dir / 'street_ends.geojson'
            
            if not geojson_path.exists():
                self.logger.warning(f"Cannot enrich {city_name}, missing {geojson_path}")
                return None

            if self.enrich_data:
                self.logger.info(f"Enriching data for {city_name}")
                start = time.time()
                config = LocationConfig(
                    radius_km=1.0,
                    output_dir=city_dir,
                    max_locations=-1,
                    save_images=True,
                    save_detailed_json=True
                )
                enricher = LocationEnricher(config)
                results = await enricher.process_locations(str(geojson_path))
                await enricher.cleanup()
                self.logger.info(f"Enrichment completed in {time.time() - start:.2f} seconds")
            else:
                results = []

            # Render map
            self.logger.info(f"Rendering map for {city_name}")
            start = time.time()
            renderer = StreetEndRenderer(
                str(geojson_path),
                city_name,
                summary_file=str(city_dir / 'summary.json')
            )
            map_html = city_dir / 'map.html'
            renderer.render(str(map_html))
            self.logger.info(f"Map rendering completed in {time.time() - start:.2f} seconds")

            # Save summary as GeoJSON FeatureCollection with city metadata as properties
            summary = {
                'type': 'FeatureCollection',
                'name': 'street_ends',
                'properties': {
                    'city_name': city_name,
                    'population': city_row['population'],
                    'latitude': city_row['latitude'],
                    'longitude': city_row['longitude'],
                    'total_street_ends': len(results['features']) if isinstance(results, dict) else 0,
                    'analysis_date': datetime.now().isoformat()
                },
                'crs': {
                    'type': 'name',
                    'properties': {
                        'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'
                    }
                },
                'features': results['features'] if isinstance(results, dict) else []
            }
            
            with open(city_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            self.analyzed_cities.append(city_name)
            self.logger.info(f"Stage 2 complete for {city_name}")
            return summary

        except Exception as e:
            self.logger.error(f"Error in Stage 2 for {city_name}: {str(e)}", exc_info=True)
            return None

    @log_timing
    def run_global_analysis_stage1(self, num_cities=10, target_cities=None):
        """
        Runs only Stage 1 for multiple cities:
        1) Get top cities or user-specified target cities
        2) For each city, generate street_ends.geojson
        """
        self.start_time = time.time()
        self.logger.info(f"Starting Global Analysis Stage 1 with {num_cities} cities")
        self.logger.info(f"Target cities: {target_cities if target_cities else 'Using top cities by population'}")
        
        if target_cities is None:
            cities = self.get_top_cities()
        else:
            cities = self.get_target_cities(target_cities)
        
        if cities is None:
            self.logger.error("Failed to get city list for Stage 1")
            return

        results = []
        for _, city in tqdm(cities.head(num_cities).iterrows(), total=len(cities.head(num_cities)), desc="Stage 1"):
            res = self.analyze_city_stage1(city)
            if res:
                results.append(res)

        # Save partial results if you want
        with open(self.output_dir / 'stage1_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    @log_timing
    async def run_global_analysis_stage2_async(self, num_cities=10, target_cities=None):
        """Async version of run_global_analysis_stage2"""
        self.start_time = time.time()
        self.logger.info(f"Starting Global Analysis Stage 2 with {num_cities} cities")
        self.logger.info(f"Target cities: {target_cities if target_cities else 'Using top cities by population'}")
        
        if target_cities is None:
            cities = self.get_top_cities()
        else:
            cities = self.get_target_cities(target_cities)
        
        if cities is None:
            self.logger.error("Failed to get city list for Stage 2")
            return

        tasks = []
        for _, city in cities.head(num_cities).iterrows():
            tasks.append(self.analyze_city_stage2_async(city))

        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]

        # Save final results
        with open(self.output_dir / 'stage2_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def run_global_analysis_stage2(self, num_cities=10, target_cities=None):
        """Synchronous wrapper for run_global_analysis_stage2_async"""
        return asyncio.run(self.run_global_analysis_stage2_async(num_cities, target_cities))

    def run_global_analysis(self, num_cities=100, target_cities=None):
        """Analyze all top cities sequentially"""
        self.logger.info(f"Starting global analysis with {num_cities} cities")
        
        if target_cities is None:
            cities = self.get_top_cities()
            self.logger.info("Using top cities by population")
        else:
            cities = self.get_target_cities(target_cities)
            self.logger.info(f"Using target cities: {target_cities}")

        if cities is None:
            self.logger.error("Failed to get city list")
            return
        
        results = []
        
        for _, city in tqdm(cities.iterrows(), total=len(cities), desc="Analyzing cities"):
            try:
                result = self.analyze_city(city)
                
                if result:
                    results.append(result)
                    self.logger.info(f"Saved results for {city['name']}")
                    # Save progress after each city
                    with open(self.output_dir / 'global_results.json', 'w') as f:
                        json.dump(results, f, indent=2)
                        
            except Exception as e:
                self.logger.error(f"Analysis failed for {city['name']}: {str(e)}", 
                              exc_info=True)
                continue
        
        self.logger.info(f"Analysis complete. Processed {len(results)} cities successfully")
        self.generate_report(results)
        
    def run_us_analysis(self, num_cities=20, target_cities=None):
        """Analyze top 20 US cities"""

        if target_cities is None:
            cities = self.get_top_us_cities(num_cities)
        else:
            cities = self.get_target_cities(target_cities)

        if cities is None:
            logging.error("Failed to get US city list")
            return
        
        results = []
        for _, city in tqdm(cities.iterrows(), total=len(cities), desc="Analyzing US cities"):
            result = self.analyze_city(city)
            if result:
                results.append(result)
            
            # Save progress after each city
            with open(self.output_dir / 'us_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        # Generate US-specific report
        self.generate_report(results, filename_prefix='us')

    def generate_report(self, results, filename_prefix=''):
        """Generate summary report of global analysis"""
        try:
            df = pd.DataFrame(results)
            
            # Convert int64 to regular int for JSON serialization
            stats = {
                'total_cities_analyzed': int(len(results)),
                'total_street_ends_found': int(df['total_street_ends'].sum()),
                'average_street_ends_per_city': float(df['total_street_ends'].mean()),
                'cities_by_street_ends': df[['city_name', 'total_street_ends']].sort_values('total_street_ends', ascending=False).to_dict('records'),
                'analysis_date': time.strftime('%Y-%m-%d')
            }
            
            # Convert any remaining int64 values in the results
            def convert_int64(obj):
                if isinstance(obj, np.int64):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_int64(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_int64(item) for item in obj]
                return obj
            
            stats = convert_int64(stats)
            
            # Save statistics
            with open(self.output_dir / f'{filename_prefix}_statistics.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate map link
            map_link = self.maps_exporter.export_cities(self.analyzed_cities)
            
            # Generate HTML report
            html_report = f"""
            <html>
                <head>
                    <title>Global Street Ends Analysis</title>
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                </head>
                <body>
                    <div class="container mt-5">
                        <h1>Global Street Ends Analysis</h1>
                        <p>Analysis Date: {stats['analysis_date']}</p>
                        <p><a href="{map_link}" target="_blank">View All Street Ends on Google Maps</a></p>
                        <h2>Summary Statistics</h2>
                        <ul>
                            <li>Total Cities Analyzed: {stats['total_cities_analyzed']}</li>
                            <li>Total Street Ends Found: {stats['total_street_ends_found']}</li>
                            <li>Average Street Ends per City: {stats['average_street_ends_per_city']:.2f}</li>
                        </ul>
                        <h2>Top 10 Cities by Street Ends</h2>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>City</th>
                                    <th>Street Ends</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {self._generate_table_rows(stats['cities_by_street_ends'])}
                            </tbody>
                        </table>
                    </div>
                </body>
            </html>
            """
            
            with open(self.output_dir / f'{filename_prefix}_report.html', 'w') as f:
                f.write(html_report)
                
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")

    def _generate_table_rows(self, cities_data):
        """Generate HTML table rows from city data"""
        rows = ""
        for city in cities_data:
            city_name = city['city_name'].split(',')[0].strip()
            map_path = f"{city['city_name'].split(',')[1].strip()}/{city_name}/map.html"
            
            # Check if map exists
            if (self.output_dir / map_path).exists():
                rows += f"""
                    <tr>
                        <td>{city['city_name']}</td>
                        <td>{city['total_street_ends']}</td>
                        <td><a href="{map_path}" target="_blank" class="btn btn-sm btn-primary">View Map</a></td>
                    </tr>
                """
            else:
                rows += f"""
                    <tr>
                        <td>{city['city_name']}</td>
                        <td>{city['total_street_ends']}</td>
                        <td>No map available</td>
                    </tr>
                """
        return rows

    def get_target_cities(self, target_cities):
        """Get data for specific target cities"""
        try:
            url = "http://download.geonames.org/export/dump/cities15000.zip"
            df = pd.read_csv(url, compression='zip', sep='\t', header=None,
                           names=['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 
                                'longitude', 'feature_class', 'feature_code', 'country_code', 
                                'cc2', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code',
                                'population', 'elevation', 'dem', 'timezone', 'modification_date'])
            
            # Filter for specified cities
            target_cities_df = df[df['name'].isin(target_cities) & (df['feature_class'] == 'P')]
            
            if len(target_cities_df) == 0:
                logging.warning(f"No matching cities found for: {target_cities}")
                return None
            
            return target_cities_df
            
        except Exception as e:
            logging.error(f"Error getting target cities: {str(e)}")
            return None

if __name__ == "__main__":
    cities_to_run = ["Skokie", "Lagrange", "Wilmington"]

    # Stage 1
    # analyzer_stage1 = GlobalCityAnalyzer(find_street_ends=True, enrich_data=False)
    # analyzer_stage1.run_global_analysis_stage1(
    #     num_cities=len(cities_to_run),
    #     target_cities=cities_to_run
    # )

    # Stage 2
    analyzer_stage2 = GlobalCityAnalyzer(find_street_ends=False, enrich_data=True)
    results = analyzer_stage2.run_global_analysis_stage2(
        num_cities=len(cities_to_run),
        target_cities=cities_to_run
    )
    analyzer_stage2.generate_report(results)