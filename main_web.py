
# main.py
from config import Config
from street_end_finder import StreetEndFinder
from enrich_data import LocationEnricher
from render_map import StreetEndRenderer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Find street ends near water bodies.')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--locations', type=str, nargs='+', help='Locations to process')
    parser.add_argument('--threshold', type=int, help='Threshold distance to water (meters)')
    parser.add_argument('--buffer', type=int, help='Buffer distance around water (meters)')
    parser.add_argument('--skip-enrich', action='store_true', help='Skip data enrichment')
    parser.add_argument('--skip-render', action='store_true', help='Skip map rendering')
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update config with command line arguments"""
    if args.locations:
        config.settings['locations'] = args.locations
    
    if args.threshold:
        config.settings['analysis']['threshold_distance'] = args.threshold
        
    if args.buffer:
        config.settings['analysis']['buffer_distance'] = args.buffer
    
    return config

def main():
    args = parse_args()
    
    # Load configuration
    config = Config(config_file=args.config if args.config else None)
    
    # Update config with command line arguments
    config = update_config_from_args(config, args)
    
    # Process each location
    locations = config.get_locations()
    output_settings = config.get_output_settings()
    
    for location in locations:
        # Find street ends
        finder = StreetEndFinder(config)
        geojson_file = finder.process_location(location)
        
        # Enrich data (if not skipped)
        # if not args.skip_enrich:
        #     enricher = LocationEnricher()
        #     enricher.process_locations(geojson_file)
        
        # Render map (if not skipped)
        if not args.skip_render:
            location_slug = location.split(',')[0].lower().replace(' ', '_')
            map_template = output_settings['map_filename_template']
            map_file = map_template.replace('{location}', location_slug)
            
            renderer = StreetEndRenderer(geojson_file, location)
            renderer.render(map_file)

if __name__ == "__main__":
    main()


# Example config.yaml
"""
locations:
  - Skokie, USA
  - Chicago, USA
  - Evanston, USA

water_features:
  water:
    - river
    - stream
    - canal
  natural:
    - riverbank

analysis:
  threshold_distance: 10
  buffer_distance: 100
  deduplication_distance: 25

output:
  geojson_filename: street_ends_{location}.geojson
  map_filename_template: street_ends_{location}.html
"""