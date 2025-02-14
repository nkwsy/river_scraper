import pandas as pd
import time
from tqdm import tqdm
import logging
from pathlib import Path
from main import StreetEndFinder
from render_map import StreetEndRenderer
from enrich_data import LocationEnricher
import json
from google_maps_export import GoogleMapsExporter
import geopandas as gpd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils.logging_config import setup_logger

class GlobalCityAnalyzer:
    def __init__(self):
        self.output_dir = Path('global_analysis')
        self.output_dir.mkdir(exist_ok=True)
        self.logger = setup_logger('global_analyzer')
        self.maps_exporter = GoogleMapsExporter()
        self.analyzed_cities = []  # Add this to track cities
        
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

    def analyze_city(self, city_row):
        """Analyze a single city"""
        try:
            city_name = f"{city_row['name']}, {city_row['country_code']}"
            city_dir = self.output_dir / city_row['country_code'] / city_row['name'].replace('/', '_')
            city_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Analyzing {city_name}")
            
            # Find street ends
            finder = StreetEndFinder(city_name, threshold_distance=10)
            water_features, near_water = finder.process()  # This saves to street_ends_near_river.geojson
            
            # Skip if no water-adjacent street ends found
            if not near_water or len(near_water) == 0:
                self.logger.info(f"Skipping {city_name}: No water-adjacent street ends found")
                return None
            
            # Copy the GeoJSON file to city directory
            source_geojson = 'street_ends_near_river.geojson'
            geojson_path = city_dir / 'street_ends.geojson'
            
            # Read and save to new location
            gdf = gpd.read_file(source_geojson)
            gdf.to_file(str(geojson_path), driver='GeoJSON')
            
            # Continue with rest of analysis
            
            enricher = LocationEnricher()
            results = enricher.process_locations(str(geojson_path))

            # Continue with rest of analysis
            renderer = StreetEndRenderer(str(geojson_path), city_name)
            renderer.render(str(city_dir / 'map.html'))
            
            
            summary = {
                'city_name': city_name,
                'population': city_row['population'],
                'latitude': city_row['latitude'],
                'longitude': city_row['longitude'],
                'total_street_ends': len(near_water),
                'analysis_date': time.strftime('%Y-%m-%d'),
                'enrichment_results': results
            }
            
            with open(city_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Add city to analyzed list
            self.analyzed_cities.append(city_name)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing {city_name}: {str(e)}")
            return None

    def run_global_analysis(self, num_cities=100, target_cities=None):
        """Analyze all top cities using multiple threads"""
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
        results_lock = threading.Lock()
        
        def analyze_and_save(city):
            thread_logger = setup_logger(f'city_analyzer_{threading.get_ident()}')
            try:
                thread_logger.info(f"Starting analysis of {city['name']}")
                result = self.analyze_city(city)
                
                if result:
                    with results_lock:
                        results.append(result)
                        thread_logger.info(f"Saved results for {city['name']}")
                        # Save progress after each city
                        with open(self.output_dir / 'global_results.json', 'w') as f:
                            json.dump(results, f, indent=2)
                return result
                
            except Exception as e:
                thread_logger.error(f"Analysis failed for {city['name']}: {str(e)}", 
                                  exc_info=True)
                return None
        
        self.logger.info(f"Initializing thread pool with {min(8, len(cities))} workers")
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_city = {executor.submit(analyze_and_save, city): city 
                            for _, city in cities.iterrows()}
            
            for future in tqdm(as_completed(future_to_city), 
                             total=len(future_to_city),
                             desc="Analyzing cities"):
                try:
                    future.result()
                except Exception as e:
                    city = future_to_city[future]
                    self.logger.error(f"City analysis failed for {city['name']}: {str(e)}", 
                                    exc_info=True)
        
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
            map_path = f"{city['country_code']}/{city_name}/map.html"
            
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
    analyzer = GlobalCityAnalyzer()
    # Choose which analysis to run
    # analyzer.run_us_analysis(num_cities=20)  # For US cities only
    # analyzer.run_global_analysis(num_cities=10)  # For global analysis 
    analyzer.run_us_analysis(target_cities=["Skokie", "Vancouver"])