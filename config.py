# config.py
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_file=None):
        self.config_file = config_file or Path('config.yaml')
        self.settings = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_file.exists():
            self._create_default_config()
        
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'locations': ['Skokie, USA'],
            'water_features': {
                'water': ['river', 'stream', 'canal']
            },
            'analysis': {
                'threshold_distance': 10,  # meters
                'buffer_distance': 100,    # meters
                'deduplication_distance': 25  # meters
            },
            'output': {
                'geojson_filename': 'street_ends_near_water.geojson',
                'map_filename_template': 'street_ends_{location}.html'
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def get_locations(self):
        """Get list of locations to process"""
        return self.settings['locations']
    
    def get_water_tags(self):
        """Get water feature tags for OSMnx"""
        return self.settings['water_features']
    
    def get_analysis_params(self):
        """Get analysis parameters"""
        return self.settings['analysis']
    
    def get_output_settings(self):
        """Get output file settings"""
        return self.settings['output']
    
    def update_config(self, new_settings):
        """Update configuration with new settings"""
        self.settings.update(new_settings)
        with open(self.config_file, 'w') as f:
            yaml.dump(self.settings, f, default_flow_style=False)



