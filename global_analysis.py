import pandas as pd
import time
from tqdm import tqdm
import logging
from pathlib import Path
from main import StreetEndFinder
from render_map import StreetEndRenderer
from enrich_data import LocationEnricher
import json

class GlobalCityAnalyzer:
    def __init__(self):
        self.output_dir = Path('global_analysis')
        self.output_dir.mkdir(exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            filename=self.output_dir / 'global_analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

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

    def get_top_us_cities(self):
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
            
            logging.info(f"Analyzing {city_name}")
            
            # Find street ends
            finder = StreetEndFinder(city_name, threshold_distance=10)
            water_features, near_water = finder.process()
            
            # Save GeoJSON
            geojson_path = city_dir / 'street_ends.geojson'
            
            # Render map
            renderer = StreetEndRenderer('street_ends_near_river.geojson', city_name)
            renderer.render(str(city_dir / 'map.html'))
            
            # Enrich data
            enricher = LocationEnricher()
            results = enricher.process_locations(str(geojson_path))
            
            # Save city summary
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
                
            return summary
            
        except Exception as e:
            logging.error(f"Error analyzing {city_name}: {str(e)}")
            return None

    def run_global_analysis(self):
        """Analyze all top cities"""
        cities = self.get_top_cities()
        if cities is None:
            logging.error("Failed to get city list")
            return
        
        results = []
        for _, city in tqdm(cities.iterrows(), total=len(cities), desc="Analyzing cities"):
            result = self.analyze_city(city)
            if result:
                results.append(result)
            
            # Save progress after each city
            with open(self.output_dir / 'global_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        # Generate final report
        self.generate_report(results)
        
    def run_us_analysis(self):
        """Analyze top 20 US cities"""
        cities = self.get_top_us_cities()
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
            
            # Basic statistics
            stats = {
                'total_cities_analyzed': len(results),
                'total_street_ends_found': df['total_street_ends'].sum(),
                'average_street_ends_per_city': df['total_street_ends'].mean(),
                'cities_by_street_ends': df.nlargest(10, 'total_street_ends')[['city_name', 'total_street_ends']].to_dict(),
                'analysis_date': time.strftime('%Y-%m-%d')
            }
            
            # Save statistics
            with open(self.output_dir / f'{filename_prefix}_statistics.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
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

    def _generate_table_rows(self, cities_dict):
        rows = ""
        for city in cities_dict['city_name'].values():
            street_ends = cities_dict['total_street_ends'][list(cities_dict['city_name'].keys())[0]]
            rows += f"<tr><td>{city}</td><td>{street_ends}</td></tr>"
        return rows

if __name__ == "__main__":
    analyzer = GlobalCityAnalyzer()
    # Choose which analysis to run
    analyzer.run_us_analysis()  # For US cities only
    # analyzer.run_global_analysis()  # For global analysis 