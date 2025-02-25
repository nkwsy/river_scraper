import folium
import geopandas as gpd
import json
from utils.logging_config import setup_logger
from utils.osmnx_load import get_ox
import numpy as np
ox = get_ox()
import os

class StreetEndRenderer:
    """Creates interactive maps showing street ends near water features.
    
    Displays points on both street and satellite maps with popups containing
    location data, images, and external map links.
    """
    
    def __init__(self, summary_file=None, location=None, **kwargs):
        """
        Args:
            points_file (str): Path to GeoJSON with street end points (not used if summary_file provided)
            location (str): Location name (e.g., "Chicago, USA")
            summary_file (str): Path to summary.json with enriched data
            **kwargs:
                threshold_distance (float): Distance in meters (default: 1000)
        """
        self.logger = setup_logger(__name__)
        self.location = location
        self.water_features = None
        self.map = None
        self.threshold_distance = kwargs.get('threshold_distance', 1000)
        
        # Load summary data
        if summary_file and os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    self.logger.info(f"Loaded summary data from {summary_file}")
                
                # Use the summary data directly as it's already a FeatureCollection
                self.points = gpd.GeoDataFrame.from_features(summary_data)
                # self.points = summary_data
                self.output_dir = os.path.dirname(os.path.abspath(summary_file))
                
            except Exception as e:
                self.logger.error(f"Error loading summary data: {str(e)}", exc_info=True)
                raise
        else:
            self.logger.error("No valid summary file provided")
            raise ValueError("Summary file is required")

    def get_water_features(self):
        """Get water features for the location"""
        self.water_features = ox.features_from_place(self.location, tags={
            "natural": ["water", "stream", "riverbank"],
            "water": ["river", "stream", "canal"]
        })
        
    def create_map(self):
        """Initialize the map with both default and satellite layers"""
        center_lat = self.points.geometry.y.mean()
        center_lon = self.points.geometry.x.mean()
        
        # Create map with default OpenStreetMap layer
        self.map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            control_scale=True
        )
        
        # Add satellite layer
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Satellite View',
            overlay=False
        ).add_to(self.map)
        
        # Rename the default layer
        folium.TileLayer(
            name='Street View',
            control=True
        ).add_to(self.map)
        
    def add_water_features(self):
        """Add water features to the map"""
        folium.GeoJson(
            self.water_features,
            name='Waterways',
            style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0.1}
        ).add_to(self.map)
        
    def add_points(self):
        """Add street end points with popups using available data"""
        for idx, point in self.points.iterrows():
            lat, lon = point.geometry.y, point.geometry.x
            properties = point.get('properties', {})
            
            # if isinstance(properties, str):
                # try:
                #     properties = json.loads(properties)
                # except:
                #     properties = {}
            
            # Determine point color based on desirability score or distance
            color = self._get_point_color(idx, point)
            
            # Create popup content with available properties
            popup_html = self._create_popup_html(idx, point)
            
            # Create the marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=folium.Popup(popup_html, max_width=500),
                color=color,
                fill=True,
                fillOpacity=0.7,
                weight=2
            ).add_to(self.map)
            
            # Add legend on first point
            if idx == 0:
                self._add_legend()

    def _get_point_color(self, idx, point):
        """Determine point color based on available metrics"""
        properties = point
        if 'desirability_score' in properties:
            score = properties['desirability_score']
            if score > 75:
                return 'yellowgreen'
            elif score > 50:
                return 'yellow'
            elif score > 25:
                return 'orange'
            return 'red'
        elif 'distance_to_water' in properties:
            # Fallback to distance-based coloring
            distance = properties['distance_to_water']
            if distance < 5:
                return 'darkgreen'
            elif distance < 10:
                return 'green'
            elif distance < 20:
                return 'orange'
            return 'red'
        return 'gray'

    def _create_popup_html(self, idx, point):
        """Create HTML content for popups with flexible property handling"""
        properties = point
        popup_html = f"""
        <div style="max-width: 400px;">
            <h4>Street End #{idx}</h4>
        """
        
        # Add distance to water if available
        if 'distance_to_water' in properties:
            popup_html += f"<strong>Distance to Water:</strong> {properties['distance_to_water']}m<br>"
        
        # Add enriched data if available
        if 'population' in properties and properties['population'] is not np.nan:
            pop_data = properties['population']
            popup_html += "<div style='margin: 10px 0;'>"
            popup_html += f"<strong>Population Density:</strong> {pop_data.get('population_density', 0):.1f}/km²<br>"
            popup_html += f"<strong>Building Count:</strong> {pop_data.get('building_count', 0)}<br>"
            popup_html += "</div>"
        
        if 'greenspace' in properties and properties['greenspace'] is not np.nan:
            green_data = properties['greenspace']
            popup_html += "<div style='margin: 10px 0;'>"
            popup_html += f"<strong>Green Space:</strong> {green_data.get('green_percentage', 0):.1f}%<br>"
            popup_html += f"<strong>Parks:</strong> {green_data.get('park_count', 0)}<br>"
            popup_html += f"<strong>Distance to Nearest Park:</strong> {green_data.get('nearest_park_distance', -1):.0f}m<br>"
            popup_html += f"<strong>Park Access Score:</strong> {green_data.get('park_distance_score', 0):.1f}/100<br>"
            popup_html += "</div>"
        
        if 'waterway' in properties and properties['waterway'] is not np.nan:
            water_data = properties['waterway']
            popup_html += "<div style='margin: 10px 0;'>"
            popup_html += f"<strong>Waterway:</strong> {water_data.get('name', 'Unknown')}<br>"
            popup_html += f"<strong>Type:</strong> {water_data.get('type', 'Unknown')}<br>"
            popup_html += "</div>"
        
        if 'desirability_score' in properties and properties['desirability_score'] is not None:
            popup_html += f"<strong>Desirability Score:</strong> {properties['desirability_score']:.1f}/100<br>"
        
        # Add images if available
        if 'images' in properties and properties['images'] is not np.nan:
            for img_type, img_path in properties['images'].items():
                if img_path:
                    rel_path = os.path.relpath(img_path, start=self.output_dir)
                    popup_html += f'<img src="{rel_path}" style="width: 100%; margin: 10px 0;">'
        
        # Always add external map link
        popup_html += f"""
            <div style='margin-top: 10px;'>
                <a href="https://www.google.com/maps/@{point.geometry.y},{point.geometry.x},100m/data=!3m1!1e3" 
                   target="_blank" class="btn btn-sm btn-secondary">
                   Open in Google Maps</a>
            </div>
        </div>
        """
        return popup_html

    def _add_legend(self):
        """Add legend to the map based on available metrics"""
        if any('desirability_score' in point.get('properties', {}) 
               for _, point in self.points.iterrows()):
            legend_html = """
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; 
                        border:2px solid grey; z-index:9999; 
                        background-color:white;
                        padding: 10px;
                        font-size: 14px;">
                <p><strong>Desirability Score</strong></p>
                <p>
                    <i class="fa fa-circle" style="color:yellowgreen"></i> Excellent (75-100)<br>
                    <i class="fa fa-circle" style="color:yellow"></i> Good (50-75)<br>
                    <i class="fa fa-circle" style="color:orange"></i> Fair (25-50)<br>
                    <i class="fa fa-circle" style="color:red"></i> Poor (0-25)<br>
                    <i class="fa fa-circle" style="color:gray"></i> No Data
                </p>
            </div>
            """
        else:
            legend_html = """
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; 
                        border:2px solid grey; z-index:9999; 
                        background-color:white;
                        padding: 10px;
                        font-size: 14px;">
                <p><strong>Applicability Score</strong></p>
                <p>
                    <i class="fa fa-circle" style="color:yellowgreen"></i> < 75%<br>
                    <i class="fa fa-circle" style="color:yellow"></i> 75-50%<br>
                    <i class="fa fa-circle" style="color:orange"></i> 50-25%<br>
                    <i class="fa fa-circle" style="color:red"></i> < 25%<br>
                    <i class="fa fa-circle" style="color:gray"></i> Unknown
                </p>
            </div>
            """
        self.map.get_root().html.add_child(folium.Element(legend_html))

    def render(self, output_file='map.html'):
        """Create and save the complete map"""
        self.get_water_features()
        self.create_map()
        self.add_water_features()
        self.add_points()
        folium.LayerControl().add_to(self.map)
        self.map.save(output_file)
        self.logger.info(f"Map saved to {output_file}")

# Example usage:
if __name__ == "__main__":
    # Basic usage
    renderer = StreetEndRenderer('global_analysis/US/Chicago/summary.json', "Chicago, US")
    renderer.render('global_analysis/US/Chicago/map.html')
	
    # Custom usage
    # renderer = StreetEndRenderer(
    #     points_file='my_points.geojson',
    #     location="Seattle, USA",
    #     threshold_distance=500,  # 500 meters
    #     enrich_data=False  # Skip enrichment
    # )
    # renderer.render('seattle_map.html') 