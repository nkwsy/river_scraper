# app.py
import streamlit as st
import yaml
import os
from pathlib import Path
import subprocess
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import time

def load_config():
    """Load configuration from YAML file"""
    config_file = Path('config.yaml')
    if not config_file.exists():
        return {}
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    """Save configuration to YAML file"""
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_analysis(location, threshold, buffer, dedup_distance):
    """Run the street end finder analysis"""
    # Update config with current parameters
    config = load_config()
    
    # Update analysis parameters
    if 'analysis' not in config:
        config['analysis'] = {}
    config['analysis']['threshold_distance'] = threshold
    config['analysis']['buffer_distance'] = buffer
    config['analysis']['deduplication_distance'] = dedup_distance
    
    # Update location
    config['locations'] = [location]
    
    # Save updated config
    save_config(config)
    
    # Run the command
    cmd = ["python", "main.py"]
    
    try:
        with st.spinner(f"Analyzing {location}..."):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Error running analysis: {result.stderr}")
                return None
            return result.stdout
    except Exception as e:
        st.error(f"Error running analysis: {str(e)}")
        return None

def display_map(geojson_file):
    """Display the results on a map"""
    if not os.path.exists(geojson_file):
        st.error(f"GeoJSON file not found: {geojson_file}")
        return
    
    # Load the GeoJSON file
    try:
        gdf = gpd.read_file(geojson_file)
    except Exception as e:
        st.error(f"Error loading GeoJSON: {str(e)}")
        return
    
    # Display statistics
    st.subheader("Street End Statistics")
    st.write(f"Found {len(gdf)} street ends near water")
    
    if len(gdf) > 0:
        # Show statistics
        distance_stats = gdf['distance_to_water'].describe()
        st.dataframe(pd.DataFrame(distance_stats).T)
        
        # Create map
        # Get the center of the data
        center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=13)
        
        # Add a marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each street end
        for idx, row in gdf.iterrows():
            popup_text = f"""
            <b>Distance to Water:</b> {row['distance_to_water']} meters<br>
            """
            if 'address' in row:
                popup_text += f"<b>Address:</b> {row['address']}<br>"
            if 'neighborhood' in row:
                popup_text += f"<b>Neighborhood:</b> {row['neighborhood']}<br>"
                
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)
        
        # Display the map
        folium_static(m)
        
        # Display data table
        with st.expander("View Data Table"):
            st.dataframe(gdf)
            
        # Download option
        st.download_button(
            label="Download GeoJSON",
            data=open(geojson_file, 'rb').read(),
            file_name=os.path.basename(geojson_file),
            mime="application/json"
        )

def main():
    st.title("Street End Finder")
    st.write("Find and visualize street ends near water bodies")
    
    # Load existing config
    config = load_config()
    
    # Set up tabs
    tab1, tab2, tab3 = st.tabs(["Run Analysis", "View Results", "Settings"])
    
    with tab1:
        st.header("Run Analysis")
        
        # Location input
        location = st.text_input(
            "Location (City, Country)", 
            value=config.get('locations', ['Skokie, USA'])[0]
        )
        
        # Analysis parameters
        analysis_params = config.get('analysis', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            threshold = st.number_input(
                "Threshold Distance (m)",
                min_value=1,
                max_value=100,
                value=analysis_params.get('threshold_distance', 10)
            )
        
        with col2:
            buffer = st.number_input(
                "Buffer Distance (m)",
                min_value=50,
                max_value=500,
                value=analysis_params.get('buffer_distance', 100)
            )
            
        with col3:
            dedup_distance = st.number_input(
                "Deduplication Distance (m)",
                min_value=5,
                max_value=100,
                value=analysis_params.get('deduplication_distance', 25)
            )
        
        # Run button
        if st.button("Run Analysis"):
            output = run_analysis(location, threshold, buffer, dedup_distance)
            if output:
                st.success("Analysis completed successfully!")
                # Generate filename based on location
                location_slug = location.split(',')[0].lower().replace(' ', '_')
                output_settings = config.get('output', {})
                geojson_template = output_settings.get('geojson_filename', 'street_ends_{location}.geojson')
                geojson_file = geojson_template.replace('{location}', location_slug)
                
                # Store for other tabs
                st.session_state['current_geojson'] = geojson_file
                
                # Display the map
                display_map(geojson_file)
    
    with tab2:
        st.header("View Results")
        
        # List all available GeoJSON files
        geojson_files = list(Path('.').glob('street_ends_*.geojson'))
        if not geojson_files:
            st.info("No analysis results found. Run an analysis first.")
        else:
            # Allow user to select a file
            file_options = [f.name for f in geojson_files]
            selected_file = st.selectbox(
                "Select Results to View",
                options=file_options,
                index=0
            )
            
            # Display the selected file
            if selected_file:
                display_map(selected_file)
    
    with tab3:
        st.header("Settings")
        
        # Water features configuration
        st.subheader("Water Feature Types")
        
        water_features = config.get('water_features', {})
        
        # Water types
        water_types = st.multiselect(
            "Water Types",
            options=["river", "stream", "canal", "lake", "pond", "reservoir"],
            default=water_features.get('water', ["river", "stream", "canal"])
        )
        
        # Natural features
        natural_features = st.multiselect(
            "Natural Features",
            options=["riverbank", "water", "wetland", "marsh"],
            default=water_features.get('natural', [])
        )
        
        # Output filename templates
        st.subheader("Output Files")
        output_settings = config.get('output', {})
        
        geojson_template = st.text_input(
            "GeoJSON Filename Template",
            value=output_settings.get('geojson_filename', 'street_ends_{location}.geojson')
        )
        
        map_template = st.text_input(
            "Map Filename Template",
            value=output_settings.get('map_filename_template', 'street_ends_{location}.html')
        )
        
        # Save settings
        if st.button("Save Settings"):
            # Update config
            if 'water_features' not in config:
                config['water_features'] = {}
            
            config['water_features']['water'] = water_types
            if natural_features:
                config['water_features']['natural'] = natural_features
            else:
                if 'natural' in config['water_features']:
                    del config['water_features']['natural']
            
            if 'output' not in config:
                config['output'] = {}
            
            config['output']['geojson_filename'] = geojson_template
            config['output']['map_filename_template'] = map_template
            
            # Save updated config
            save_config(config)
            st.success("Settings saved successfully!")

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