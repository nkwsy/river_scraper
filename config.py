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
            return self.settings
        
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Ensure all required sections exist with default values
        default_config = {
            'locations': ['Skokie, USA'],
            'water_features': {
                'water': ['river', 'stream', 'canal']
            },
            'analysis': {
                'threshold_distance': 10,
                'buffer_distance': 100,
                'deduplication_distance': 25
            },
            'output': {
                'geojson_filename': 'street_ends_{location}.geojson',
                'map_filename_template': 'street_ends_{location}.html'
            }
        }
        
        # Merge with defaults, keeping existing values
        for section, defaults in default_config.items():
            if section not in config:
                config[section] = defaults
            elif isinstance(defaults, dict):
                for key, value in defaults.items():
                    if key not in config[section]:
                        config[section][key] = value
        
        return config
    
    def _create_default_config(self):
        """Create default configuration file"""
        self.settings = {
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
                'geojson_filename': 'street_ends_{location}.geojson',
                'map_filename_template': 'street_ends_{location}.html'
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.settings, f, default_flow_style=False)
    
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



