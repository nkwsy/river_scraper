import os
from dotenv import load_dotenv
import osmnx as ox

# Load environment variables
load_dotenv()

# Get cache directory from environment variables
cache_dir = os.getenv('CACHE_DIR', 'cache')
overpass_url = os.getenv('OVERPASS_URL', 'http://overpass-api.de/api/')
overpass_rate_limit = os.getenv('OVERPASS_RATE_LIMIT', True)
overpass_verbose = os.getenv('OVERPASS_VERBOSE', False)
# Create cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)


def get_ox():
    # Configure OSMnx
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir
    ox.settings.log_console = overpass_verbose
    ox.settings.overpass_url = overpass_url
    ox.settings.overpass_rate_limit = overpass_rate_limit
    return ox