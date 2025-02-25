import sys
import os
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Check if running directly (not through streamlit run)
is_streamlit_run = 'streamlit.runtime.scriptrunner' in sys.modules

# Create a fallback for st functions when running directly
if not is_streamlit_run:
    print("\n⚠️  WARNING: This script should be run with `streamlit run street_ends_dashboard.py`")
    print("   Running directly will only perform data loading and validation.\n")
    
    # Create mock streamlit functions to prevent errors
    class MockSt:
        def __init__(self):
            pass
            
        def set_page_config(self, **kwargs):
            pass
            
        def markdown(self, text, **kwargs):
            pass
            
        def spinner(self, text):
            class MockSpinner:
                def __enter__(self):
                    print(f"Processing: {text}")
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    print("Done!")
            return MockSpinner()
            
        def error(self, text):
            print(f"ERROR: {text}")
            
        def warning(self, text):
            print(f"WARNING: {text}")
            
        def info(self, text):
            print(f"INFO: {text}")
            
        def success(self, text):
            print(f"SUCCESS: {text}")
            
        def columns(self, n):
            class MockColumn:
                def __enter__(self):
                    return [MockSt() for _ in range(n)]
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return MockColumn()
            
        def sidebar(self):
            return MockSt()
            
        def text_input(self, label, **kwargs):
            return kwargs.get('value', '')
            
        def selectbox(self, label, options, **kwargs):
            if options:
                return options[0]
            return None
            
        def radio(self, label, options, **kwargs):
            if options:
                return options[0]
            return None
            
        def checkbox(self, label, **kwargs):
            return False
            
        def dataframe(self, df, **kwargs):
            if isinstance(df, pd.DataFrame):
                print(f"DataFrame with {len(df)} rows and {len(df.columns)} columns")
            
        def plotly_chart(self, fig, **kwargs):
            pass
            
        def download_button(self, **kwargs):
            pass
            
        def expander(self, label):
            class MockExpander:
                def __enter__(self):
                    return MockSt()
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return MockExpander()
            
        def json(self, data):
            pass
            
        def metric(self, label, value, **kwargs):
            pass
    
    # Replace streamlit with our mock
    st = MockSt()

# Now proceed with your regular code
if is_streamlit_run:
    # Set page configuration (only in Streamlit mode)
    st.set_page_config(
        page_title="Street Ends Global Analysis",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .metric-label {
            font-size: 1rem;
            color: #424242;
        }
        .footer {
            margin-top: 3rem;
            text-align: center;
            color: #9e9e9e;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

def get_target_cities(target_cities):
    """Get data for specific target cities"""
    try:
        url = "http://download.geonames.org/export/dump/cities15000.zip"
        
        # Download the zip file
        st.info(f"Downloading city data from {url}...")
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Failed to download city data: HTTP {response.status_code}")
            return None
        
        # Save the zip file temporarily
        import tempfile
        import zipfile
        import io
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            temp_zip.write(response.content)
            temp_zip_path = temp_zip.name
        
        # Extract and read the file
        city_data = []
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                # The file inside is a tab-separated text file
                with zip_ref.open('cities15000.txt') as city_file:
                    for line in city_file:
                        # Decode each line individually
                        try:
                            decoded_line = line.decode('utf-8').strip()
                            fields = decoded_line.split('\t')
                            if len(fields) >= 19:  # Ensure we have all fields
                                city_data.append({
                                    'geonameid': fields[0],
                                    'name': fields[1],
                                    'asciiname': fields[2],
                                    'alternatenames': fields[3],
                                    'latitude': float(fields[4]),
                                    'longitude': float(fields[5]),
                                    'feature_class': fields[6],
                                    'feature_code': fields[7],
                                    'country_code': fields[8],
                                    'population': int(fields[14]) if fields[14].isdigit() else 0
                                })
                        except UnicodeDecodeError:
                            continue  # Skip lines that can't be decoded
        finally:
            # Clean up the temporary file
            import os
            try:
                os.unlink(temp_zip_path)
            except:
                pass
        
        # Convert to DataFrame
        df = pd.DataFrame(city_data)
        
        # Filter for specified cities
        target_cities_df = df[df['name'].isin(target_cities) & (df['feature_class'] == 'P')]
        
        if len(target_cities_df) == 0:
            # Try with alternative names
            for city in target_cities:
                alt_matches = df[df['alternatenames'].str.contains(city, case=False, na=False) & 
                                (df['feature_class'] == 'P')]
                if not alt_matches.empty:
                    target_cities_df = pd.concat([target_cities_df, alt_matches])
        
        if len(target_cities_df) == 0:
            st.warning(f"No matching cities found for: {target_cities}")
            return None
        
        return target_cities_df
        
    except Exception as e:
        st.error(f"Error getting target cities: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def load_summary_data(output_dir):
    """Load all summary data from the analysis directory"""
    output_dir = Path(output_dir)
    
    # Collect all summary files
    summaries = []
    city_details = []
    
    # Keep track of cities we need to fetch data for
    cities_to_fetch = []
    
    # Add default data for testing if no valid summaries are found
    default_data_added = False
    
    for country_dir in output_dir.glob('*'):
        if not country_dir.is_dir():
            continue
            
        for city_dir in country_dir.glob('*'):
            summary_file = city_dir / 'summary.json'
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                        
                        # Validate summary structure
                        if not isinstance(summary, dict):
                            st.warning(f"Invalid summary format in {summary_file} - not a dictionary")
                            continue
                        
                        # If properties is missing, try to create it from the summary data
                        if 'properties' not in summary:
                            st.warning(f"Missing 'properties' in {summary_file}, attempting to fix...")
                            
                            # Extract city name from path
                            city_name = city_dir.name
                            country_code = country_dir.name
                            
                            # Create a basic properties structure
                            summary['properties'] = {
                                'city_name': f"{city_name}, {country_code}",
                                'country_code': country_code,
                                'population': 0,
                                'total_street_ends': len(summary.get('features', [])),
                                'latitude': 0,
                                'longitude': 0
                            }
                            
                            # Save the updated summary
                            try:
                                with open(summary_file, 'w') as f:
                                    json.dump(summary, f, indent=2)
                                st.success(f"Fixed summary file: {summary_file}")
                            except Exception as e:
                                st.error(f"Could not save fixed summary: {str(e)}")
                        
                        # Add to summaries
                        summaries.append(summary)
                        
                        # Extract city name for fetching additional data if needed
                        city_name = summary['properties'].get('city_name', 'Unknown')
                        if ',' in city_name:
                            city_name = city_name.split(',')[0].strip()
                        
                        # Check if we need to fetch additional data
                        if (summary['properties'].get('latitude', 0) == 0 or 
                            summary['properties'].get('longitude', 0) == 0 or
                            summary['properties'].get('population', 0) == 0):
                            cities_to_fetch.append(city_name)
                        
                        # Extract basic city info
                        city_detail = {
                            'city_name': summary['properties'].get('city_name', 'Unknown'),
                            'country_code': summary['properties'].get('country_code', 'Unknown'),
                            'population': summary['properties'].get('population', 0),
                            'total_street_ends': summary['properties'].get('total_street_ends', 0),
                            'latitude': summary['properties'].get('latitude', 0),
                            'longitude': summary['properties'].get('longitude', 0),
                            'pop_density_avg': 0,
                            'green_percentage_avg': 0,
                            'desirability_score_avg': 0,
                            'feature_count': len(summary.get('features', [])),
                            'summary_path': str(summary_file),
                            'map_path': str(city_dir / 'map.html'),
                            'city_only_name': city_name  # Store just the city name for matching
                        }
                        
                        # Process features to extract metrics
                        if 'features' in summary and summary['features']:
                            pop_densities = []
                            green_percentages = []
                            desirability_scores = []
                            
                            for feature in summary['features']:
                                props = feature.get('properties', {})
                                
                                # Handle population data - could be nested or direct
                                if isinstance(props.get('population'), dict):
                                    pop_density = props['population'].get('population_density')
                                    if pop_density is not None:
                                        try:
                                            pop_densities.append(float(pop_density))
                                        except (ValueError, TypeError):
                                            pass
                                
                                # Handle greenspace data
                                if isinstance(props.get('greenspace'), dict):
                                    green_pct = props['greenspace'].get('green_percentage')
                                    if green_pct is not None:
                                        try:
                                            green_percentages.append(float(green_pct))
                                        except (ValueError, TypeError):
                                            pass
                                
                                # Handle desirability score
                                if 'desirability_score' in props:
                                    try:
                                        score = float(props['desirability_score'])
                                        desirability_scores.append(score)
                                    except (ValueError, TypeError):
                                        pass
                            
                            # Calculate averages
                            if pop_densities:
                                city_detail['pop_density_avg'] = sum(pop_densities) / len(pop_densities)
                            if green_percentages:
                                city_detail['green_percentage_avg'] = sum(green_percentages) / len(green_percentages)
                            if desirability_scores:
                                city_detail['desirability_score_avg'] = sum(desirability_scores) / len(desirability_scores)
                        
                        city_details.append(city_detail)
                        
                except Exception as e:
                    st.error(f"Error reading {summary_file}: {str(e)}")
    
    # If no valid summaries were found, add some default data for testing
    if not summaries:
        st.warning("No valid summary files found. Adding sample data for testing.")
        
        # Create sample data for Chicago and New York
        sample_cities = [
            {
                'city_name': 'Chicago, US',
                'country_code': 'US',
                'population': 2746388,
                'total_street_ends': 120,
                'latitude': 41.8781,
                'longitude': -87.6298,
                'pop_density_avg': 4582.3,
                'green_percentage_avg': 17.6,
                'desirability_score_avg': 72.5,
                'feature_count': 120,
                'summary_path': str(Path(output_dir) / 'US' / 'Chicago' / 'summary.json'),
                'map_path': str(Path(output_dir) / 'US' / 'Chicago' / 'map.html'),
                'city_only_name': 'Chicago'
            },
            {
                'city_name': 'New York, US',
                'country_code': 'US',
                'population': 8804190,
                'total_street_ends': 215,
                'latitude': 40.7128,
                'longitude': -74.0060,
                'pop_density_avg': 10716.4,
                'green_percentage_avg': 14.2,
                'desirability_score_avg': 68.3,
                'feature_count': 215,
                'summary_path': str(Path(output_dir) / 'US' / 'New York' / 'summary.json'),
                'map_path': str(Path(output_dir) / 'US' / 'New York' / 'map.html'),
                'city_only_name': 'New York'
            }
        ]
        
        city_details.extend(sample_cities)
        default_data_added = True
    
    # Fetch additional city data if needed and not using default data
    if cities_to_fetch and not default_data_added:
        st.info(f"Fetching additional data for {len(cities_to_fetch)} cities...")
        city_data_df = get_target_cities(cities_to_fetch)
        
        if city_data_df is not None:
            # Update city details with fetched data
            for i, city_detail in enumerate(city_details):
                city_name = city_detail['city_only_name']
                matching_rows = city_data_df[city_data_df['name'] == city_name]
                
                if not matching_rows.empty:
                    # Get the first matching row
                    city_row = matching_rows.iloc[0]
                    
                    # Update city details
                    if city_detail['latitude'] == 0:
                        city_detail['latitude'] = city_row['latitude']
                    if city_detail['longitude'] == 0:
                        city_detail['longitude'] = city_row['longitude']
                    if city_detail['population'] == 0:
                        city_detail['population'] = city_row['population']
                    
                    # Also update the summary file
                    summary_path = city_detail['summary_path']
                    try:
                        with open(summary_path, 'r') as f:
                            summary_data = json.load(f)
                        
                        # Update properties
                        if summary_data['properties'].get('latitude', 0) == 0:
                            summary_data['properties']['latitude'] = city_row['latitude']
                        if summary_data['properties'].get('longitude', 0) == 0:
                            summary_data['properties']['longitude'] = city_row['longitude']
                        if summary_data['properties'].get('population', 0) == 0:
                            summary_data['properties']['population'] = int(city_row['population'])
                        
                        # Save updated summary
                        with open(summary_path, 'w') as f:
                            json.dump(summary_data, f, indent=2)
                            
                    except Exception as e:
                        st.warning(f"Could not update summary file {summary_path}: {str(e)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(city_details)
    
    # Generate statistics
    try:
        stats = {
            'total_cities_analyzed': int(len(city_details)),
            'total_street_ends_found': int(df['total_street_ends'].sum()),
            'average_street_ends_per_city': float(df['total_street_ends'].mean()),
            'cities_by_street_ends': df[['city_name', 'total_street_ends']].sort_values('total_street_ends', ascending=False).to_dict('records'),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        stats = {
            'total_cities_analyzed': len(city_details),
            'total_street_ends_found': 0,
            'average_street_ends_per_city': 0,
            'cities_by_street_ends': [],
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    return df, stats

def create_global_map(df):
    """Create a global map with all analyzed cities"""
    # Filter out rows with invalid coordinates
    map_df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
    
    if map_df.empty:
        st.warning("No valid coordinates found for mapping")
        return None
    
    # Create map centered at average coordinates
    center_lat = map_df['latitude'].mean()
    center_lon = map_df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    # Add markers for each city
    for _, row in map_df.iterrows():
        # Create popup content
        popup_content = f"""
        <div style="width: 200px">
            <h4>{row['city_name']}</h4>
            <p><b>Street Ends:</b> {row['total_street_ends']}</p>
            <p><b>Population:</b> {row['population']:,}</p>
            <p><b>Avg Score:</b> {row['desirability_score_avg']:.1f}</p>
        </div>
        """
        
        # Determine marker color based on desirability score
        if row['desirability_score_avg'] >= 70:
            color = 'green'
        elif row['desirability_score_avg'] >= 50:
            color = 'orange'
        elif row['desirability_score_avg'] > 0:
            color = 'red'
        else:
            color = 'gray'
        
        # Add marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=max(5, min(15, row['total_street_ends'] / 10)),  # Size based on street end count
            popup=folium.Popup(popup_content, max_width=300),
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    
    st_folium(m, width=1200, height=600)

def create_city_comparison_chart(df):
    """Create a bar chart comparing cities by street ends"""
    # Sort and get top 15 cities
    top_cities = df.sort_values('total_street_ends', ascending=False).head(15)
    
    fig = px.bar(
        top_cities,
        x='city_name',
        y='total_street_ends',
        color='desirability_score_avg',
        color_continuous_scale='RdYlGn',
        title='Top Cities by Street End Count',
        labels={
            'city_name': 'City',
            'total_street_ends': 'Number of Street Ends',
            'desirability_score_avg': 'Avg Desirability Score'
        }
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

def create_correlation_chart(df):
    """Create a scatter plot showing correlation between metrics"""
    # Filter out rows with zero values
    plot_df = df[(df['pop_density_avg'] > 0) & (df['green_percentage_avg'] > 0)].copy()
    
    if plot_df.empty:
        st.warning("Not enough data for correlation analysis")
        return None
    
    fig = px.scatter(
        plot_df,
        x='pop_density_avg',
        y='green_percentage_avg',
        size='total_street_ends',
        color='desirability_score_avg',
        color_continuous_scale='RdYlGn',
        hover_name='city_name',
        title='Relationship Between Population Density and Green Space',
        labels={
            'pop_density_avg': 'Population Density',
            'green_percentage_avg': 'Green Space Percentage',
            'total_street_ends': 'Street End Count',
            'desirability_score_avg': 'Desirability Score'
        }
    )
    
    fig.update_layout(height=500)
    
    return fig

def display_city_details(city_data, output_dir):
    """Display detailed information for a selected city"""
    st.subheader(f"Details for {city_data['city_name']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Population", f"{city_data['population']:,}")
        st.metric("Street Ends", city_data['total_street_ends'])
        st.metric("Avg Population Density", f"{city_data['pop_density_avg']:.1f}")
    
    with col2:
        st.metric("Avg Green Space", f"{city_data['green_percentage_avg']:.1f}%")
        st.metric("Avg Desirability Score", f"{city_data['desirability_score_avg']:.1f}")
        st.metric("Features Analyzed", city_data['feature_count'])
    
    # Display map
    st.subheader("City Map")
    
    # Check if we have coordinates
    if city_data['latitude'] != 0 and city_data['longitude'] != 0:
        # Create a folium map
        m = folium.Map(
            location=[city_data['latitude'], city_data['longitude']], 
            zoom_start=13
        )
        
        # Add a marker for the city center
        folium.Marker(
            [city_data['latitude'], city_data['longitude']],
            popup=f"<b>{city_data['city_name']}</b>",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
        
        # If we have the summary file, add markers for street ends
        summary_path = city_data['summary_path']
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)
                
                if 'features' in summary_data:
                    # Create a feature group for street ends
                    street_ends = folium.FeatureGroup(name="Street Ends")
                    
                    # Add markers for each street end
                    for i, feature in enumerate(summary_data['features']):
                        if 'geometry' in feature and feature['geometry'].get('type') == 'Point':
                            coords = feature['geometry'].get('coordinates', [0, 0])
                            if len(coords) >= 2 and coords[0] != 0 and coords[1] != 0:
                                # Get properties for popup
                                props = feature.get('properties', {})
                                desirability = props.get('desirability_score', 0)
                                
                                # Determine color based on desirability
                                if desirability >= 70:
                                    color = 'green'
                                elif desirability >= 50:
                                    color = 'orange'
                                elif desirability > 0:
                                    color = 'red'
                                else:
                                    color = 'gray'
                                
                                # Create popup content
                                popup_content = f"""
                                <div style="width: 200px">
                                    <h4>Street End #{i+1}</h4>
                                    <p><b>Desirability:</b> {desirability:.1f}</p>
                                """
                                
                                if isinstance(props.get('population'), dict):
                                    pop_density = props['population'].get('population_density')
                                    if pop_density is not None:
                                        popup_content += f"<p><b>Population Density:</b> {pop_density:.1f}</p>"
                                
                                if isinstance(props.get('greenspace'), dict):
                                    green_pct = props['greenspace'].get('green_percentage')
                                    if green_pct is not None:
                                        popup_content += f"<p><b>Green Space:</b> {green_pct:.1f}%</p>"
                                
                                popup_content += "</div>"
                                
                                # Add marker
                                folium.CircleMarker(
                                    location=[coords[1], coords[0]],  # Note: GeoJSON is [lon, lat]
                                    radius=8,
                                    popup=folium.Popup(popup_content, max_width=300),
                                    color=color,
                                    fill=True,
                                    fill_opacity=0.7
                                ).add_to(street_ends)
                    
                    # Add the feature group to the map
                    street_ends.add_to(m)
                    
                    # Add layer control
                    folium.LayerControl().add_to(m)
            except Exception as e:
                st.error(f"Error adding street ends to map: {str(e)}")
        
        # Display the map
        st_folium(m, width=1000, height=600)
        
        # Provide a download link for the original map if it exists
        map_path = city_data['map_path']
        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                map_content = f.read()
            
            st.download_button(
                label="Download Original Map HTML",
                data=map_content,
                file_name=f"{city_data['city_name'].replace(', ', '_')}_map.html",
                mime="text/html"
            )
    else:
        st.warning("No valid coordinates available for this city.")
    
    # Load and display summary data
    summary_path = city_data['summary_path']
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        st.subheader("Summary Data")
        
        # Display feature statistics
        if 'features' in summary_data and summary_data['features']:
            features = summary_data['features']
            
            # Extract data for visualization
            desirability_scores = []
            pop_densities = []
            green_percentages = []
            
            for feature in features:
                props = feature.get('properties', {})
                
                if 'desirability_score' in props:
                    try:
                        desirability_scores.append(float(props['desirability_score']))
                    except (ValueError, TypeError):
                        pass
                
                if isinstance(props.get('population'), dict):
                    pop_density = props['population'].get('population_density')
                    if pop_density is not None:
                        pop_densities.append(float(pop_density))
                
                if isinstance(props.get('greenspace'), dict):
                    green_pct = props['greenspace'].get('green_percentage')
                    if green_pct is not None:
                        green_percentages.append(float(green_pct))
            
            # Create visualizations
            if desirability_scores:
                st.subheader("Desirability Score Distribution")
                fig = px.histogram(
                    x=desirability_scores,
                    nbins=20,
                    labels={'x': 'Desirability Score'},
                    title=f"Distribution of Desirability Scores in {city_data['city_name']}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Create a correlation plot if we have both metrics
            if pop_densities and green_percentages:
                st.subheader("Population Density vs Green Space")
                fig = px.scatter(
                    x=pop_densities,
                    y=green_percentages,
                    labels={
                        'x': 'Population Density',
                        'y': 'Green Space Percentage'
                    },
                    title=f"Relationship Between Population Density and Green Space in {city_data['city_name']}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Option to view raw JSON
        with st.expander("View Raw Summary Data"):
            st.json(summary_data)
    else:
        st.warning(f"Summary file not found: {summary_path}")

def main():
    st.markdown('<h1 class="main-header">Street Ends Global Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    output_dir = st.sidebar.text_input("Analysis Directory", value="global_analysis")
    
    # Load data
    with st.spinner("Loading data... This may take a moment."):
        df, stats = load_summary_data(output_dir)
    
    if df is None or stats is None:
        st.error(f"Failed to load data from {output_dir}")
        return
    
    # Display global statistics
    st.markdown('<h2 class="sub-header">Global Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_cities_analyzed']}</div>
                <div class="metric-label">Cities Analyzed</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_street_ends_found']}</div>
                <div class="metric-label">Total Street Ends</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['average_street_ends_per_city']:.1f}</div>
                <div class="metric-label">Avg Street Ends per City</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Global map
    st.markdown('<h2 class="sub-header">Global Map</h2>', unsafe_allow_html=True)
    with st.spinner("Generating global map..."):
        global_map = create_global_map(df)
        if global_map:
            st_folium(global_map, width=1200, height=600)
        else:
            st.warning("Could not create global map due to missing coordinate data.")
    
    # City comparison charts
    st.markdown('<h2 class="sub-header">City Comparisons</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        comparison_chart = create_city_comparison_chart(df)
        st.plotly_chart(comparison_chart, use_container_width=True)
    
    with col2:
        correlation_chart = create_correlation_chart(df)
        if correlation_chart:
            st.plotly_chart(correlation_chart, use_container_width=True)
        else:
            st.info("Not enough data to create correlation chart.")
    
    # City details section
    st.markdown('<h2 class="sub-header">City Details</h2>', unsafe_allow_html=True)
    
    # Sort cities by street end count for the dropdown
    sorted_df = df.sort_values('total_street_ends', ascending=False)
    city_options = sorted_df['city_name'].tolist()
    
    selected_city = st.selectbox("Select a city to view details", city_options)
    
    if selected_city:
        with st.spinner(f"Loading details for {selected_city}..."):
            city_data = df[df['city_name'] == selected_city].iloc[0].to_dict()
            display_city_details(city_data, output_dir)
    
    # Data table with all cities
    st.markdown('<h2 class="sub-header">All Cities Data</h2>', unsafe_allow_html=True)
    
    # Add a search box
    search_term = st.text_input("Search cities")
    
    # Filter the dataframe based on search
    if search_term:
        filtered_df = df[df['city_name'].str.contains(search_term, case=False)]
    else:
        filtered_df = df
    
    # Sort options
    sort_col = st.selectbox(
        "Sort by", 
        ['total_street_ends', 'population', 'pop_density_avg', 'green_percentage_avg', 'desirability_score_avg'],
        index=0
    )
    
    sort_order = st.radio("Sort order", ["Descending", "Ascending"], horizontal=True)
    
    # Apply sorting
    if sort_order == "Ascending":
        sorted_filtered_df = filtered_df.sort_values(sort_col, ascending=True)
    else:
        sorted_filtered_df = filtered_df.sort_values(sort_col, ascending=False)
    
    # Display the table
    st.dataframe(
        sorted_filtered_df[[
            'city_name', 'country_code', 'population', 'total_street_ends', 
            'pop_density_avg', 'green_percentage_avg', 'desirability_score_avg'
        ]].rename(columns={
            'city_name': 'City',
            'country_code': 'Country',
            'population': 'Population',
            'total_street_ends': 'Street Ends',
            'pop_density_avg': 'Pop Density',
            'green_percentage_avg': 'Green %',
            'desirability_score_avg': 'Score'
        }),
        use_container_width=True,
        height=400
    )
    
    # Footer
    st.markdown(
        f"""
        <div class="footer">
            <p>Street Ends Global Analysis Dashboard | Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    if is_streamlit_run:
        main()
    else:
        # When running directly, just test data loading
        print("Testing data loading from global_analysis directory...")
        try:
            df, stats = load_summary_data("\TRUENAS\media\research\cities\global_analysis")
            if df is not None and stats is not None:
                print(f"Successfully loaded data for {stats['total_cities_analyzed']} cities with {stats['total_street_ends_found']} street ends.")
                print("\nTo view the dashboard, run: streamlit run street_ends_dashboard.py")
            else:
                print("Failed to load data. Check if the global_analysis directory exists and contains valid data.")
        except Exception as e:
            print(f"Error loading data: {str(e)}")