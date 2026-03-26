import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import pathlib
import json
import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily
import numpy as np

# Constants
DATA_FILE = pathlib.Path('data/processed/clusters.parquet')
POINTS_FILE = pathlib.Path('data/processed/cluster_points.parquet')
DEFAULT_CITY = "Frankfurt am Main"

st.set_page_config(layout="wide", page_title="Urban Mobility Risk App")

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error(f"Data file not found at {DATA_FILE}. Please run the processor first.")
        return pd.DataFrame()
    return pd.read_parquet(DATA_FILE)

@st.cache_data
def load_points_data():
    if not POINTS_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(POINTS_FILE)

@st.dialog("Cluster Zoom View", width="large")
def show_cluster_dialog(selected_cluster, points_df, selected_mode):
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown(f"**Detailed Map:** {selected_cluster['LocationName']}")
        cluster_points = pd.DataFrame()
        noise_points = pd.DataFrame()
        
        if not points_df.empty:
            cluster_points = points_df[
                (points_df['City'] == selected_cluster['City']) &
                (points_df['Mode'] == selected_cluster['Mode']) &
                (points_df['Cluster_ID'] == selected_cluster['Cluster_ID'])
            ]
            noise_points = points_df[
                (points_df['City'] == selected_cluster['City']) &
                (points_df['Mode'] == selected_cluster['Mode']) &
                (points_df['Cluster_ID'] == -1)
            ]
        
        if not cluster_points.empty:
            COORDINATE_SYSTEM_GPS = "EPSG:25832"
            COORDINATE_SYSTEM_WEB = "EPSG:3857"
            MAP_PADDING_FACTOR = 0.1
            
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            if not noise_points.empty:
                gdf_noise = gpd.GeoDataFrame(
                    noise_points, 
                    geometry=gpd.points_from_xy(noise_points.X_Meters, noise_points.Y_Meters),
                    crs=COORDINATE_SYSTEM_GPS
                ).to_crs(COORDINATE_SYSTEM_WEB)
                gdf_noise.plot(ax=ax, color='lightgrey', markersize=5, alpha=0.3)
                
            # Increased Jitter to spread perfectly stacked coordinates across the intersection
            jitter_x = cluster_points.X_Meters + np.random.uniform(-15, 15, len(cluster_points))
            jitter_y = cluster_points.Y_Meters + np.random.uniform(-15, 15, len(cluster_points))
            gdf_clusters = gpd.GeoDataFrame(
                cluster_points, 
                geometry=gpd.points_from_xy(jitter_x, jitter_y), 
                crs=COORDINATE_SYSTEM_GPS
            ).to_crs(COORDINATE_SYSTEM_WEB)
            
            color = "red" 
            if selected_mode == 'Bicycle': color = 'blue'
            elif selected_mode == 'Pedestrian': color = 'green'
            
            # Using alpha transparency and white edgecolors to distinctly expose overlapping dots!
            gdf_clusters.plot(ax=ax, color=color, markersize=50, alpha=0.6, edgecolors='white', linewidth=0.5, legend=False)
            
            minx, miny, maxx, maxy = gdf_clusters.total_bounds
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            d = max(maxx - minx, maxy - miny) * (1 + 2 * MAP_PADDING_FACTOR)
            if d == 0: d = 1000
            ax.set_xlim(cx - d/2, cx + d/2)
            ax.set_ylim(cy - d/2, cy + d/2)
            contextily.add_basemap(ax, source=contextily.providers.CartoDB.Positron)
            ax.set_axis_off()
            st.pyplot(fig)
        else:
            st.warning("Point data for the selected cluster not found. Background offline processor may still be running for this city.")

    with col2:
        st.markdown("**Accident Types:**")
        st.markdown(f"*{selected_cluster['AccidentCount']} Total Accidents*")
        if 'AccidentTypeStats' in selected_cluster and selected_cluster['AccidentTypeStats']:
            try:
                stats = json.loads(selected_cluster['AccidentTypeStats'])
                accident_map = {'1': "Driving", '2': "Turn off", '3': "Crossing/turning", '4': "Pedestrian crossing", '5': "Stationary vehicle", '6': "Moving vehicles", '7': "Other"}
                mapped_stats = {accident_map.get(k, f"Code {k}"): v for k, v in stats.items()}
                chart_df = pd.DataFrame(list(mapped_stats.items()), columns=['Type', 'Count'])
                
                c = alt.Chart(chart_df).mark_bar().encode(
                    x='Count', 
                    y=alt.Y('Type', sort='-x'), 
                    tooltip=['Type', 'Count']
                ).properties(height=300)
                st.altair_chart(c, use_container_width=True)
                
                st.markdown(f"[📍 Open Intersection in Google Maps]({selected_cluster.get('GoogleMapsLink', '#')})")
            except Exception as e:
                st.caption("No valid type breakdown available.")

def main():
    st.title("🚦 Urban Mobility Risk Dashboard")
    st.markdown("Identify high-leverage intervention points for traffic safety.")

    df = load_data()
    points_df = load_points_data()
    if df.empty:
        return

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    
    # City Selector
    available_cities = sorted(df['City'].unique())
    default_index = available_cities.index(DEFAULT_CITY) if DEFAULT_CITY in available_cities else 0
    selected_city = st.sidebar.selectbox("Select City", available_cities, index=default_index)

    # Accident Type Filter
    # Get available modes for this city (though typically fixed set)
    # We want "All" to show everything, or select specific?
    # Processor saves 'Mode' column: 'Bicycle', 'Car', 'Pedestrian', 'All' (if we kept it? We might have skipped 'All' in loop if dict was just keys)
    # Let's check processor logic: "for mode_name, col_name in modes.items():" -> if I ran it, it would have 'All' too.
    # The user said "I want to look at vehicle, pedestrian and bicycle accidents as filter options"
    # So drop-down.
    
    available_modes = sorted(df['Mode'].unique())
    # Ensure 'All' is first if present, otherwise All means no filter? 
    # Actually the processor generates CLUSTERS for specific modes. 
    # So if I select "Bicycle", I show clusters generated from Bicycle data.
    # If I select "All", I show clusters generated from ALL data.
    # This is distinct data rows.
    
    selected_mode = st.sidebar.selectbox("Accident Type Involved", available_modes)

    # --- Filtering ---
    # Filter by City AND Mode
    filtered_data = df[
        (df['City'] == selected_city) & 
        (df['Mode'] == selected_mode)
    ]

    # --- Main Content ---
    
    # KPIs
    st.markdown("### Key Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Clusters", len(filtered_data))
    col2.metric("Total Accidents", filtered_data['AccidentCount'].sum() if not filtered_data.empty else 0)

    col_map, col_details = st.columns([2, 1])

    with col_map:
        st.subheader(f"Risk Map ({selected_city})")
        
        if not filtered_data.empty:
            # Centroid
            center_lat = filtered_data['Lat'].mean()
            center_lon = filtered_data['Lon'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")
            
            for _, row in filtered_data.iterrows():
                # Color logic
                color = "red"
                if selected_mode == 'Bicycle': color = 'blue'
                elif selected_mode == 'Pedestrian': color = 'green'
                
                # Check for Google Maps Link
                gmaps_link = row.get('GoogleMapsLink', '#')

                # Popup Content
                popup_html = f"""
                <div style="font-family: sans-serif;">
                    <b>{row['LocationName']}</b><br>
                    Accidents: {row['AccidentCount']}<br>
                    <a href="{gmaps_link}" target="_blank">Open in Google Maps</a>
                </div>
                """
                
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']],
                    radius=6, # Fixed size as requested ("...away from a scaling bubble")
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{row['LocationName']} ({row['AccidentCount']} accidents)",
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    fill_color=color
                ).add_to(m)
            
            map_output = st_folium(m, use_container_width=True, height=600, key="main_map")
            
            clicked_lat_lon = None
            if map_output and map_output.get("last_object_clicked"):
                clicked_lat_lon = (map_output["last_object_clicked"]["lat"], map_output["last_object_clicked"]["lng"])
                
                # Check session state so it doesn't infinitely loop
                if st.session_state.get('last_opened_cluster') != clicked_lat_lon:
                    st.session_state['last_opened_cluster'] = clicked_lat_lon
                    
                    dist = (filtered_data['Lat'] - clicked_lat_lon[0])**2 + (filtered_data['Lon'] - clicked_lat_lon[1])**2
                    if not dist.empty and dist.min() < 1e-4:
                        selected_cluster = filtered_data.loc[dist.idxmin()]
                        show_cluster_dialog(selected_cluster, points_df, selected_mode)
                        st.rerun() # Refresh state clean
        else:
            st.info("No clusters found matching the criteria.")

    with col_details:
        st.subheader("High-Risk Locations")
        if not filtered_data.empty:
            st.info("Click a marker on the map or select below to view details.")
            top_clusters = filtered_data.sort_values(by='AccidentCount', ascending=False)
            
            for i, row in top_clusters.head(8).iterrows():
                with st.container(border=True):
                    st.markdown(f"**#{i+1}: {row['LocationName']}**")
                    st.markdown(f"{row['AccidentCount']} Accidents")
                    if st.button("🔍 View Details", key=f"btn_{row['City']}_{row['Cluster_ID']}"):
                        show_cluster_dialog(row, points_df, selected_mode)
        else:
            st.write("Select a city and filters to view details.")

    # --- Footer ---
    st.caption(f"Data Source: German Traffic Accident Data (2016-2024). Processing version 2.0 (Type-Specific Clusters).")

if __name__ == "__main__":
    main()
