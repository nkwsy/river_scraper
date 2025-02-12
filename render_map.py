import folium
import geopandas as gpd
import osmnx as ox

class StreetEndRenderer:
    def __init__(self, points_file, location):
        self.points = gpd.read_file(points_file)
        self.location = location
        self.water_features = None
        self.map = None
        
    def get_water_features(self):
        """Get water features for the location"""
        self.water_features = ox.features_from_place(self.location, tags={
            "natural": ["water", "stream", "riverbank"],
            "waterway": ["river", "stream", "canal"]
        })
        
    def create_map(self):
        """Initialize the map"""
        center_lat = self.points.geometry.y.mean()
        center_lon = self.points.geometry.x.mean()
        self.map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
    def add_water_features(self):
        """Add water features to the map"""
        folium.GeoJson(
            self.water_features,
            name='Waterways',
            style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0.1}
        ).add_to(self.map)
        
    def add_points(self):
        """Add street end points with popups"""
        for idx, point in self.points.iterrows():
            lat, lon = point.geometry.y, point.geometry.x
            popup_html = f"""
            <div>
                <h4>Street End #{idx}</h4>
                <a href="https://www.google.com/maps/@{lat},{lon},3a,75y,0h,90t/data=!3m6!1e1" 
                   target="_blank">Street View</a><br>
                <a href="https://www.google.com/maps/@{lat},{lon},100m/data=!3m1!1e3" 
                   target="_blank">Aerial View</a>
            </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                popup=popup_html,
                color='red',
                fill=True
            ).add_to(self.map)
            
    def render(self, output_file='street_ends_map.html'):
        """Create and save the complete map"""
        self.get_water_features()
        self.create_map()
        self.add_water_features()
        self.add_points()
        folium.LayerControl().add_to(self.map)
        self.map.save(output_file)
        print(f"Map saved to {output_file}")

# Example usage:
if __name__ == "__main__":
    renderer = StreetEndRenderer('street_ends_near_river.geojson', "Chicago, USA")
    renderer.render() 