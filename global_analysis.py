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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor,as_completed
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
        self.max_workers = kwargs.get('max_workers', 4)
        self.update_existing = kwargs.get('update_existing', False)
        self.use_multithreading = kwargs.get('use_multithreading', False)
    def get_top_cities(self, num_cities=1000):
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
            return cities.head(num_cities)
            
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
            
            # Skip if file exists and we're not updating
            if geojson_path.exists() and not self.update_existing:
                self.logger.info(f"Skipping {city_name} - file exists and update_existing=False")
                return None
            if self.find_street_ends:
                self.logger.info(f"Finding street ends for {city_name}")
                start = time.time()
                finder = StreetEndFinder(city_name, threshold_distance=30, geojson_file=geojson_path, city_dir=city_dir)
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
    def analyze_city_stage2(self, city_row):
        """Synchronous version of analyze_city_stage2_async"""
        try:
            city_name = f"{city_row['name']}, {city_row['country_code']}"
            self.logger.info(f"Starting enrichment for {city_name}")
            
            city_dir = self.output_dir / city_row['country_code'] / city_row['name'].replace('/', '_')
            geojson_path = city_dir / 'street_ends.geojson'
            summary_path = city_dir / 'summary.json'
            skip_enrichment = False

            if not geojson_path.exists():
                self.logger.warning(f"Cannot enrich {city_name}, missing {geojson_path}")
                return None
            if summary_path.exists() and not self.update_existing:
                skip_enrichment = True
                self.logger.info(f"Skipping {city_name} - file exists and update_existing=False")
            if self.enrich_data and not skip_enrichment:
                self.logger.info(f"Enriching data for {city_name}")
                start = time.time()
                config = LocationConfig(
                    radius_km=1.0,
                    output_dir=city_dir,
                    max_locations=-1,
                    save_images=False,
                    save_detailed_json=True,
                    batch_size=10,
                    cache_osm_data=True,
                    concurrency_limit=1,
                )
                enricher = LocationEnricher(config)
                # Use asyncio.run to run the async method in a sync context
                results = asyncio.run(enricher.process_locations(str(geojson_path)))
                asyncio.run(enricher.cleanup())
                self.logger.info(f"Enrichment completed in {time.time() - start:.2f} seconds")
            else:
                results = []

            # Render map
            self.logger.info(f"Rendering map for {city_name}")
            start = time.time()
            renderer = StreetEndRenderer(
                str(geojson_path),
                city_name,
                summary_file=str(summary_path)
            )
            map_html = city_dir / 'map.html'
            renderer.render(str(map_html))
            self.logger.info(f"Map rendering completed in {time.time() - start:.2f} seconds")

            # Create summary
            summary = {
                'type': 'FeatureCollection',
                'name': 'street_ends',
                'properties': {
                    'city_name': city_name,
                    'country_code': city_row['country_code'],
                    'name': city_row['name'],
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
    def run_global_analysis_stage1(self, num_cities=10, target_cities=None, max_workers=4, use_multithreading=None):
        """
        Runs only Stage 1 for multiple cities:
        1) Get top cities or user-specified target cities
        2) For each city, generate street_ends.geojson
        
        Args:
            num_cities: Number of cities to process
            target_cities: List of specific cities to target
            max_workers: Number of parallel workers (only used if use_multithreading=True)
            use_multithreading: Whether to use parallel processing (defaults to self.use_multithreading)
        """
        self.start_time = time.time()
        
        # Determine whether to use multithreading
        if use_multithreading is None:
            use_multithreading = self.use_multithreading
            
        workers_msg = f"using {max_workers} workers" if use_multithreading else "sequentially"
        self.logger.warning(f"Starting Global Analysis Stage 1 with {num_cities} cities {workers_msg}")
        self.logger.warning(f"Target cities: {target_cities if target_cities else 'Using top cities by population'}")
        
        if target_cities is None:
            cities = self.get_top_cities(num_cities)
        else:
            cities = self.get_target_cities(target_cities)
        
        if cities is None:
            self.logger.error("Failed to get city list for Stage 1")
            return

        results = []
        
        # Use multithreading if requested
        if use_multithreading:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Create a dictionary of future to city name for better error reporting
                future_to_city = {
                    executor.submit(self.analyze_city_stage1, city): city['name'] 
                    for _, city in cities.head(num_cities).iterrows()
                }

                # Process completed futures as they come in
                for future in tqdm(as_completed(future_to_city), 
                                 total=len(future_to_city), 
                                 desc="Processing cities"):
                    city_name = future_to_city[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            self.logger.warning(f"Successfully processed {city_name}")
                    except Exception as e:
                        self.logger.error(f"City processing failed for {city_name}: {str(e)}", 
                                        exc_info=True)
        else:
            # Process cities sequentially
            for _, city in tqdm(cities.head(num_cities).iterrows(), 
                              total=min(num_cities, len(cities)), 
                              desc="Processing cities"):
                try:
                    result = self.analyze_city_stage1(city)
                    if result:
                        results.append(result)
                        self.logger.warning(f"Successfully processed {city['name']}")
                except Exception as e:
                    self.logger.error(f"City processing failed for {city['name']}: {str(e)}", 
                                    exc_info=True)

        # Save partial results
        with open(self.output_dir / 'stage1_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Stage 1 complete. Successfully processed {len(results)} out of {num_cities} cities")
        return results

    @log_timing
    def run_global_analysis_stage2(self, num_cities=10, target_cities=None, max_workers=4, use_multithreading=None):
        """
        Process multiple cities for Stage 2 (enrichment and map generation)
        
        Args:
            num_cities: Number of cities to process
            target_cities: List of specific cities to target
            max_workers: Number of parallel workers (only used if use_multithreading=True)
            use_multithreading: Whether to use parallel processing (defaults to self.use_multithreading)
        """
        self.start_time = time.time()
        
        # Determine whether to use multithreading
        if use_multithreading is None:
            use_multithreading = self.use_multithreading
            
        workers_msg = f"using {max_workers} workers" if use_multithreading else "sequentially"
        self.logger.info(f"Starting Global Analysis Stage 2 with {num_cities} cities {workers_msg}")
        self.logger.info(f"Target cities: {target_cities if target_cities else 'Using top cities by population'}")
        
        if target_cities is None:
            cities = self.get_top_cities()
        else:
            cities = self.get_target_cities(target_cities)
        
        if cities is None:
            self.logger.error("Failed to get city list for Stage 2")
            return

        results = []
        
        # Use multithreading if requested
        if use_multithreading:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Create a dictionary of future to city name for better error reporting
                future_to_city = {
                    executor.submit(self.analyze_city_stage2, city): city['name'] 
                    for _, city in cities.head(num_cities).iterrows()
                }

                # Process completed futures as they come in
                for future in tqdm(as_completed(future_to_city), 
                                 total=len(future_to_city), 
                                 desc="Processing cities"):
                    city_name = future_to_city[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            self.logger.warning(f"Successfully processed {city_name}")
                    except Exception as e:
                        self.logger.error(f"City processing failed for {city_name}: {str(e)}", 
                                        exc_info=True)
        else:
            # Process cities sequentially
            for _, city in tqdm(cities.head(num_cities).iterrows(), 
                              total=min(num_cities, len(cities)), 
                              desc="Processing cities"):
                try:
                    result = self.analyze_city_stage2(city)
                    if result:
                        results.append(result)
                        self.logger.warning(f"Successfully processed {city['name']}")
                except Exception as e:
                    self.logger.error(f"City processing failed for {city['name']}: {str(e)}", 
                                    exc_info=True)

        # Save final results
        with open(self.output_dir / 'stage2_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

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
                        
                        <h2>Top Cities by Street Ends</h2>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>City</th>
                                    <th>Country</th>
                                    <th>Population</th>
                                    <th>Street Ends</th>
                                    <th>Map</th>
                                    <th>Data</th>
                                </tr>
                            </thead>
                            <tbody>
                                {self._generate_table_rows(df)}
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

    def _generate_table_rows(self, df: pd.DataFrame) -> str:
        """Generate HTML table rows from DataFrame"""
        rows = ""
        for _, row in df.iterrows():
            try:
                # Get city name with error handling
                city_name = row.get('city_name', 'Unknown')
                
                # Handle potential errors in city_name format
                try:
                    if ',' in city_name:
                        country_code = city_name.split(',')[1].strip()
                        city = city_name.split(',')[0].strip()
                    else:
                        country_code = row.get('country_code', 'Unknown')
                        city = city_name
                except Exception:
                    country_code = row.get('country_code', 'Unknown')
                    city = city_name
                
                # Sanitize values for path construction
                city = city.replace('/', '_').replace('\\', '_')
                country_code = country_code.replace('/', '_').replace('\\', '_')
                
                # Create paths
                map_path = f"{country_code}/{city}/map.html"
                summary_path = f"{country_code}/{city}/summary.json"
                
                # Get other values with defaults
                population = row.get('population', 0)
                total_street_ends = row.get('total_street_ends', 0)
                
                # Check if map exists
                if (self.output_dir / map_path).exists():
                    rows += f"""
                        <tr>
                            <td>{city_name}</td>
                            <td>{country_code}</td>
                            <td>{population:,}</td>
                            <td>{total_street_ends}</td>
                            <td>
                                <a href="{map_path}" target="_blank" class="btn btn-sm btn-primary">View Map</a>
                            </td>
                            <td>
                                <a href="{summary_path}" target="_blank" class="btn btn-sm btn-primary">View raw data</a>
                            </td>
                        </tr>
                    """
                else:
                    rows += f"""
                        <tr>
                            <td>{city_name}</td>
                            <td>{country_code}</td>
                            <td>{population:,}</td>
                            <td>{total_street_ends}</td>
                            <td colspan="2">No map available</td>
                        </tr>
                    """
            except Exception as e:
                self.logger.error(f"Error generating table row: {str(e)}", exc_info=True)
                # Add a fallback row with error information
                rows += f"""
                    <tr>
                        <td colspan="6" class="text-danger">Error processing city data: {str(e)}</td>
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

    @log_timing
    def analyze_city(self, city_row):
        """
        Analyze a single city by running both stage 1 and stage 2 sequentially.
        This is used by run_global_analysis and run_us_analysis methods.
        """
        try:
            city_name = f"{city_row['name']}, {city_row['country_code']}"
            self.logger.info(f"Processing {city_name}")
            
            # Run stage 1 - find street ends
            stage1_result = self.analyze_city_stage1(city_row)
            if not stage1_result:
                self.logger.warning(f"Stage 1 failed for {city_name}")
                return None
                
            # Run stage 2 - enrich data and create map
            stage2_result = self.analyze_city_stage2(city_row)
            if not stage2_result:
                self.logger.warning(f"Stage 2 failed for {city_name}")
                return None
                
            return stage2_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {city_row['name']}: {str(e)}", exc_info=True)
            return None

    @log_timing
    def run_global_analysis_stage3(self, results_dir=None):
        """
        Stage 3: Generate comprehensive reports from completed analyses
        Combines all city summaries and generates global statistics and visualizations
        """
        if results_dir is None:
            results_dir = self.output_dir
        
        self.logger.info(f"Starting Global Analysis Stage 3 - Report Generation")
        
        try:
            # Collect all summary files
            summaries = []
            for country_dir in Path(results_dir).glob('*'):
                if not country_dir.is_dir():
                    continue
                    
                for city_dir in country_dir.glob('*'):
                    summary_file = city_dir / 'summary.json'
                    if summary_file.exists():
                        try:
                            with open(summary_file) as f:
                                summary = json.load(f)
                                # Validate summary structure with better error handling
                                if not isinstance(summary, dict):
                                    self.logger.warning(f"Invalid summary format in {summary_file} - not a dictionary")
                                    continue
                                    
                                # Check and fix required keys
                                if 'type' not in summary:
                                    self.logger.warning(f"Missing 'type' in {summary_file}, adding default")
                                    summary['type'] = 'FeatureCollection'
                                    
                                if 'features' not in summary:
                                    self.logger.warning(f"Missing 'features' in {summary_file}, adding empty list")
                                    summary['features'] = []
                                    
                                if 'properties' not in summary:
                                    self.logger.warning(f"Missing 'properties' in {summary_file}, adding default")
                                    summary['properties'] = {
                                        'city_name': city_dir.name,
                                        'country_code': country_dir.name,
                                        'name': city_dir.name,
                                        'population': 0,
                                        'total_street_ends': 0,
                                        'latitude': 0,
                                        'longitude': 0,
                                        'analysis_date': datetime.now().isoformat()
                                    }
                                
                                # Ensure all required properties exist
                                required_props = ['city_name', 'country_code', 'name', 'population', 
                                                 'total_street_ends', 'latitude', 'longitude', 'analysis_date']
                                for prop in required_props:
                                    if prop not in summary['properties']:
                                        self.logger.warning(f"Missing property '{prop}' in {summary_file}, adding default")
                                        if prop in ['population', 'total_street_ends', 'latitude', 'longitude']:
                                            summary['properties'][prop] = 0
                                        elif prop == 'analysis_date':
                                            summary['properties'][prop] = datetime.now().isoformat()
                                        else:
                                            summary['properties'][prop] = city_dir.name if prop == 'name' else \
                                                                         f"{city_dir.name}, {country_dir.name}" if prop == 'city_name' else \
                                                                         country_dir.name
                                
                                # Now that we've fixed any issues, add to summaries
                                summaries.append(summary)
                                self.analyzed_cities.append(summary['properties']['city_name'])
                                
                                # Optionally save the fixed summary back to file
                                with open(summary_file, 'w') as f_out:
                                    json.dump(summary, f_out, indent=2)
                                    
                        except Exception as e:
                            self.logger.error(f"Error reading {summary_file}: {str(e)}", exc_info=True)
            
            if not summaries:
                self.logger.error("No summary files found to generate report")
                return
            
            # Convert summaries to DataFrame for analysis with error handling
            df_data = []
            for s in summaries:
                try:
                    props = s['properties']
                    df_data.append({
                        'city_name': props.get('city_name', 'Unknown'),
                        'country_code': props.get('country_code', 'Unknown'),
                        'name': props.get('name', 'Unknown'),
                        'population': props.get('population', 0),
                        'total_street_ends': props.get('total_street_ends', 0),
                        'latitude': props.get('latitude', 0),
                        'longitude': props.get('longitude', 0),
                        'analysis_date': props.get('analysis_date', datetime.now().isoformat()),
                        'features': len(s.get('features', []))
                    })
                except Exception as e:
                    self.logger.error(f"Error processing summary: {str(e)}", exc_info=True)
            
            df = pd.DataFrame(df_data)
            
            # Generate statistics with error handling
            try:
                stats = {
                    'total_cities_analyzed': int(len(summaries)),
                    'total_street_ends_found': int(df['total_street_ends'].sum()),
                    'average_street_ends_per_city': float(df['total_street_ends'].mean()),
                    'cities_by_street_ends': df[['city_name', 'total_street_ends']].sort_values('total_street_ends', ascending=False).to_dict('records'),
                    'analysis_date': datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"Error calculating statistics: {str(e)}", exc_info=True)
                stats = {
                    'total_cities_analyzed': len(summaries),
                    'total_street_ends_found': 0,
                    'average_street_ends_per_city': 0,
                    'cities_by_street_ends': [],
                    'analysis_date': datetime.now().isoformat()
                }
            
            # Save statistics
            with open(results_dir / 'global_statistics.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate map link with error handling
            try:
                map_link = self.maps_exporter.export_cities(self.analyzed_cities)
            except Exception as e:
                self.logger.error(f"Error generating map link: {str(e)}", exc_info=True)
                map_link = "#"
            
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
                        
                        <h2>Top Cities by Street Ends</h2>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>City</th>
                                    <th>Country</th>
                                    <th>Population</th>
                                    <th>Street Ends</th>
                                    <th>Map</th>
                                    <th>Data</th>
                                </tr>
                            </thead>
                            <tbody>
                                {self._generate_table_rows(df)}
                            </tbody>
                        </table>
                    </div>
                </body>
            </html>
            """
            
            with open(results_dir / 'global_report.html', 'w') as f:
                f.write(html_report)
                
            self.logger.info(f"Stage 3 complete - Report generated at {results_dir / 'global_report.html'}")
            
        except Exception as e:
            self.logger.error(f"Error in Stage 3: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # cities_to_run = ["Skokie", "Chicago", "Toronto"]
    cities_to_run = ['Skokie', "Chicago"]

    # Example 1: Run Stage 1 sequentially (default)
    analyzer_stage1 = GlobalCityAnalyzer(find_street_ends=True, enrich_data=False, update_existing=False)
    # analyzer_stage1.run_global_analysis_stage1(
    #     target_cities=cities_to_run
    # )

    # Example 2: Run Stage 1 with multithreading (explicitly enabled)
    # analyzer_stage1_mt = GlobalCityAnalyzer(
    #     find_street_ends=True, 
    #     enrich_data=False, 
    #     update_existing=False,
    #     use_multithreading=True,  # Enable multithreading
    #     max_workers=4
    # )
    # analyzer_stage1_mt.run_global_analysis_stage1(
    #     target_cities=cities_to_run
    # )

    # Example 3: Run Stage 2 sequentially (default)
    analyzer_stage2 = GlobalCityAnalyzer(find_street_ends=False, enrich_data=False, update_existing=False)
    # results = analyzer_stage2.run_global_analysis_stage2(
    #     target_cities=['Chicago']
    # )

    # Example 4: Run Stage 2 with multithreading (explicitly enabled)
    # analyzer_stage2_mt = GlobalCityAnalyzer(
    #     find_street_ends=False, 
    #     enrich_data=True, 
    #     update_existing=False,
    #     use_multithreading=True,  # Enable multithreading
    #     max_workers=4
    # )
    # results_mt = analyzer_stage2_mt.run_global_analysis_stage2(
    #     target_cities=['Chicago']
    # )

    # Stage 3: Generate report (always runs sequentially)
    analyzer_stage3 = GlobalCityAnalyzer()
    analyzer_stage3.run_global_analysis_stage3()