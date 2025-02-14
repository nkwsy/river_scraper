import folium
import geopandas as gpd
import osmnx as ox
from utils.logging_config import setup_logger
from enrich_data import LocationEnricher
import os

class StreetEndRenderer:
    def __init__(self, points_file, location):
        self.logger = setup_logger(__name__)
        self.points = gpd.read_file(points_file)
        self.location = location
        self.water_features = None
        self.map = None
        self.output_dir = os.path.abspath(os.getenv('OUTPUT_DIR', 'location_data'))
        
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
        """Add street end points with enriched data popups"""
        enricher = LocationEnricher()
        
        for idx, point in self.points.iterrows():
            lat, lon = point.geometry.y, point.geometry.x
            
            # Get enriched data for this location
            enriched_data = enricher.enrich_location(lat, lon, idx)
            
            # Determine point color based on desirability score
            if enriched_data and 'desirability_score' in enriched_data:
                score = enriched_data['desirability_score']
                if score > 75:
                    color = 'darkgreen'
                elif score > 50:
                    color = 'green'
                elif score > 25:
                    color = 'orange'
                else:
                    color = 'red'
            else:
                color = 'gray'
            
            # Create popup content
            popup_html = f"""
            <div style="max-width: 400px;">
                <h4>Street End #{idx}</h4>
            """
            
            # Add street view image if available
            if enriched_data and 'images' in enriched_data and 'streetview' in enriched_data['images']:
                # Convert file:// URL to relative path
                image_path = enriched_data['images']['streetview']
                if image_path.startswith('file://'):
                    image_path = os.path.relpath(
                        image_path.replace('file://', ''),
                        start=os.path.dirname(os.path.abspath(output_file))
                    )
                popup_html += f"""
                <img src="{image_path}" 
                     style="width: 100%; margin: 10px 0;">
                """
            
            # Add enriched data details
            if enriched_data:
                popup_html += "<div style='margin: 10px 0;'>"
                if 'desirability_score' in enriched_data:
                    popup_html += f"<strong>Desirability Score:</strong> {enriched_data['desirability_score']:.1f}/100<br>"
                
                if 'population' in enriched_data:
                    pop_data = enriched_data['population']
                    popup_html += f"""
                    <strong>Area Type:</strong> {pop_data.get('density_category', 'Unknown')}<br>
                    <strong>Built-up Area:</strong> {pop_data.get('built_up_percentage', 0):.1f}%<br>
                    <strong>Population Density:</strong> {pop_data.get('population_density', 0):.1f}/km²<br>
                    """
                
                if 'greenspace' in enriched_data:
                    green_data = enriched_data['greenspace']
                    popup_html += f"""
                    <strong>Green Space:</strong> {green_data.get('green_percentage', 0):.1f}%<br>
                    <strong>Nearby Parks:</strong> {green_data.get('num_parks', 0)}<br>
                    """
                
                if 'waterway' in enriched_data:
                    water_data = enriched_data['waterway']
                    popup_html += f"""
                    <strong>Waterway:</strong> {water_data.get('type', 'Unknown')}<br>
                    <strong>Name:</strong> {water_data.get('name', 'Unknown')}<br>
                    """
                popup_html += "</div>"
            
            # Add external links
            popup_html += f"""
                <div style='margin-top: 10px;'>
                    <a href="https://www.google.com/maps/@{lat},{lon},3a,75y,0h,90t/data=!3m6!1e1" 
                       target="_blank" class="btn btn-sm btn-primary">
                       Open in Street View</a>
                    <a href="https://www.google.com/maps/@{lat},{lon},100m/data=!3m1!1e3" 
                       target="_blank" class="btn btn-sm btn-secondary">
                       Open Aerial View</a>
                </div>
            </div>
            """
            
            # Create the marker with the enriched popup
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=folium.Popup(popup_html, max_width=500),
                color=color,
                fill=True,
                fillOpacity=0.7,
                weight=2
            ).add_to(self.map)
            
            # Add a legend
            if idx == 0:  # Only add legend once
                legend_html = """
                <div style="position: fixed; 
                            bottom: 50px; right: 50px; 
                            border:2px solid grey; z-index:9999; 
                            background-color:white;
                            padding: 10px;
                            font-size: 14px;">
                    <p><strong>Desirability Score</strong></p>
                    <p>
                        <i class="fa fa-circle" style="color:darkgreen"></i> Excellent (75-100)<br>
                        <i class="fa fa-circle" style="color:green"></i> Good (50-75)<br>
                        <i class="fa fa-circle" style="color:orange"></i> Fair (25-50)<br>
                        <i class="fa fa-circle" style="color:red"></i> Poor (0-25)<br>
                        <i class="fa fa-circle" style="color:gray"></i> No Data
                    </p>
                </div>
                """
                self.map.get_root().html.add_child(folium.Element(legend_html))
            
    def render(self, output_file='street_ends_map.html'):
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
    renderer = StreetEndRenderer('street_ends_near_river.geojson', "Chicago, USA")
    renderer.render() 