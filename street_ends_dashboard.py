import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Set page configuration
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

def load_summary_data(output_dir):
    """Load all summary data from the analysis directory"""
    output_dir = Path(output_dir)
    
    # Collect all summary files
    summaries = []
    city_details = []
    
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
                            
                        # Check required keys
                        if 'properties' not in summary:
                            st.warning(f"Missing 'properties' in {summary_file}, skipping")
                            continue
                        
                        # Add to summaries
                        summaries.append(summary)
                        
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
                            'map_path': str(city_dir / 'map.html')
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
                                        pop_densities.append(float(pop_density))
                                
                                # Handle greenspace data
                                if isinstance(props.get('greenspace'), dict):
                                    green_pct = props['greenspace'].get('green_percentage')
                                    if green_pct is not None:
                                        green_percentages.append(float(green_pct))
                                
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
    
    if not summaries:
        st.error("No summary files found to generate report")
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(city_details)
    
    # Generate statistics
    try:
        stats = {
            'total_cities_analyzed': int(len(summaries)),
            'total_street_ends_found': int(df['total_street_ends'].sum()),
            'average_street_ends_per_city': float(df['total_street_ends'].mean()),
            'cities_by_street_ends': df[['city_name', 'total_street_ends']].sort_values('total_street_ends', ascending=False).to_dict('records'),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        stats = {
            'total_cities_analyzed': len(summaries),
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
    
    return m

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
    
    # Check if map file exists
    map_path = city_data['map_path']
    if os.path.exists(map_path):
        st.subheader("City Map")
        
        # Create an iframe to display the HTML map
        map_html = f'<iframe src="file://{os.path.abspath(map_path)}" width="100%" height="500"></iframe>'
        st.markdown(map_html, unsafe_allow_html=True)
        
        # Also provide a download link
        with open(map_path, 'r') as f:
            map_content = f.read()
        
        st.download_button(
            label="Download Map HTML",
            data=map_content,
            file_name=f"{city_data['city_name'].replace(', ', '_')}_map.html",
            mime="text/html"
        )
    else:
        st.warning(f"Map file not found: {map_path}")
    
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
    global_map = create_global_map(df)
    if global_map:
        folium_static(global_map, width=1200, height=600)
    
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
    
    # City details section
    st.markdown('<h2 class="sub-header">City Details</h2>', unsafe_allow_html=True)
    
    # Sort cities by street end count for the dropdown
    sorted_df = df.sort_values('total_street_ends', ascending=False)
    city_options = sorted_df['city_name'].tolist()
    
    selected_city = st.selectbox("Select a city to view details", city_options)
    
    if selected_city:
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
    main()