import streamlit as st
import numpy as np
import pandas as pd
import json
import math
import pydeck as pdk

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="19.3 Maps and Geospatial Data",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.html(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    </style>
    """
)

st.markdown('<div class="main-title">19.3 Maps and Geospatial Data in Streamlit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Complete architectural guide to st.map(), WGS84 coordinate systems, PyDeck deck.gl integration, multi-layer 3D visualizations, GeoJSON polygons, and real-time geographic filtering.</div>',
    unsafe_allow_html=True
)

# Helper function to generate reproducible geographic telemetry data
def get_sample_geo_data(center_lat=37.7749, center_lon=-122.4194, n_points=60, spread=0.04):
    np.random.seed(42)
    lats = center_lat + np.random.randn(n_points) * spread
    lons = center_lon + np.random.randn(n_points) * spread
    severities = np.random.choice(["High", "Medium", "Low"], size=n_points, p=[0.25, 0.45, 0.30])
    requests_per_sec = np.random.randint(150, 4500, size=n_points)
    
    # Assign color codes
    color_map = {
        "High": [239, 68, 68, 200],     # Red
        "Medium": [245, 158, 11, 180],  # Amber
        "Low": [16, 185, 129, 160]      # Emerald
    }
    hex_map = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
    
    df = pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "node_id": [f"EDGE-NODE-{i:03d}" for i in range(1, n_points + 1)],
        "severity": severities,
        "rps": requests_per_sec,
        "color_rgba": [color_map[s] for s in severities],
        "color_hex": [hex_map[s] for s in severities],
        "point_size": requests_per_sec / 40.0
    })
    return df

# -----------------------------------------------------------------------------
# TOPIC 1: ST.MAP() FUNDAMENTALS & PARAMETER SIGNATURE
# -----------------------------------------------------------------------------
st.header("1. st.map() Fundamentals & Parameter Signature", divider="blue")

st.markdown(
    r"""
`st.map()` is Streamlit's native point map visualizer powered internally by Mapbox and Deck.gl.
It plots points on a high-performance vector tile map with automatic zoom and bounds centering.

```python
st.map(
    data=None,
    *,
    latitude=None,
    longitude=None,
    color=None,
    size=None,
    zoom=None,
    use_container_width=None, # Deprecated in Streamlit 1.60+ in favor of width='stretch'
    width="stretch"
)
```
"""
)

st.markdown(
    r"""
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame`, `list[dict]` | `None` | Tabular data containing geographic coordinates and optional style columns. |
| `latitude` | `str` | `None` | Column name for latitude. Defaults to auto-detecting `"latitude"`, `"lat"`, `"LAT"`. |
| `longitude` | `str` | `None` | Column name for longitude. Defaults to auto-detecting `"longitude"`, `"lon"`, `"lng"`. |
| `color` | `str` | `None` | Hex string (e.g. `"#FF4B4B"`) or column name with hex strings for categorical coloring. |
| `size` | `str` or `int` | `None` | Radius of points in meters (int), or column name containing per-point sizes. |
| `zoom` | `int` | `None` | Initial zoom level between `0` (global) and `20` (street level). |
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
import pandas as pd

# 1. Geographic DataFrame
df_servers = pd.DataFrame({
    "lat": [37.7749, 37.7833, 37.7650],
    "lon": [-122.4194, -122.4167, -122.4300],
    "node_color": ["#EF4444", "#10B981", "#3B82F6"],
    "traffic_size": [120, 250, 80]
})

# 2. Render native vector map
st.map(
    df_servers,
    latitude="lat",
    longitude="lon",
    color="node_color",
    size="traffic_size",
    zoom=12
)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Native st.map() Render Output")
        
        sample_df = get_sample_geo_data(n_points=35)
        st.map(
            sample_df,
            latitude="latitude",
            longitude="longitude",
            color="color_hex",
            size="point_size",
            zoom=11
        )
        st.caption("Auto-centered over San Francisco Bay Area cluster with categorical colors.")

# -----------------------------------------------------------------------------
# TOPIC 2: LATITUDE AND LONGITUDE COORDINATE STANDARDS
# -----------------------------------------------------------------------------
st.header("2. Latitude and Longitude Coordinate Standards (WGS84)", divider="blue")

st.markdown(
    r"""
Streamlit and Deck.gl adhere strictly to the **WGS84 (EPSG:4326)** World Geodetic System:
- **Latitude ($\phi$)**: Measured North-South relative to the Equator. Valid range: $[-90.0, +90.0]$.
- **Longitude ($\lambda$)**: Measured East-West relative to the Prime Meridian. Valid range: $[-180.0, +180.0]$.
- **Auto-Detection**: Streamlit automatically scans DataFrame columns for `latitude`, `lat`, `y`, `longitude`, `lon`, `lng`, `x` (case-insensitive).
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
import pandas as pd

def validate_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str):
    valid_lat = df[lat_col].between(-90.0, 90.0).all()
    valid_lon = df[lon_col].between(-180.0, 180.0).all()
    if not (valid_lat and valid_lon):
        st.error("Invalid WGS84 coordinates detected! Out of bounds.")
        return False
    return True
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("WGS84 Coordinate Validator")
        
        c_lat, c_lon = st.columns(2)
        with c_lat:
            in_lat = st.number_input("Test Latitude (°)", value=40.7128, min_value=-90.0, max_value=90.0, step=0.01)
        with c_lon:
            in_lon = st.number_input("Test Longitude (°)", value=-74.0060, min_value=-180.0, max_value=180.0, step=0.01)
            
        st.success(f"✅ Coordinates Valid: NYC Coordinates (`{in_lat:.4f} N, {abs(in_lon):.4f} W`)")
        st.dataframe(
            pd.DataFrame([{"Location": "New York City", "Latitude": in_lat, "Longitude": in_lon, "EPSG": "4326"}]),
            width="stretch"
        )

# -----------------------------------------------------------------------------
# TOPIC 3: DISPLAYING GEOGRAPHIC DATASETS
# -----------------------------------------------------------------------------
st.header("3. Displaying Geographic Datasets", divider="blue")

st.markdown(
    r"""
Real-world geospatial telemetry contains rich auxiliary attributes (request throughput, latency, temperatures, 
incident severities). Pair `st.map()` with linked data tables or summary KPI cards to provide contextual telemetry.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st

df = load_telemetry_nodes()

# Summary Telemetry Header
col1, col2, col3 = st.columns(3)
col1.metric("Active Nodes", len(df))
col2.metric("Mean Latency", f"{df['latency_ms'].mean():.1f} ms")
col3.metric("Critical Nodes", len(df[df['severity'] == 'High']))

# Render geographic distribution
st.map(df, color="color_hex", size="point_size")

# Tabular inspector
with st.expander("Inspect Raw Sensor Data"):
    st.dataframe(df.drop(columns=["color_rgba"]))
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Linked Geospatial Telemetry Dashboard")
        
        geo_df = get_sample_geo_data(n_points=40)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Monitored Nodes", f"{len(geo_df)} Sites")
        m2.metric("Avg Throughput", f"{geo_df['rps'].mean():.0f} req/s")
        m3.metric("Critical Alerts", f"{len(geo_df[geo_df['severity'] == 'High'])}", delta="Requires Review")
        
        st.map(geo_df, color="color_hex", size="point_size", zoom=11)
        
        with st.expander("🔍 Tabular Geographic Records"):
            st.dataframe(geo_df[["node_id", "severity", "rps", "latitude", "longitude"]].head(6), width="stretch")

# -----------------------------------------------------------------------------
# TOPIC 4: PYDECK INTEGRATION
# -----------------------------------------------------------------------------
st.header("4. PyDeck Integration (`st.pydeck_chart`)", divider="blue")

st.markdown(
    r"""
While `st.map()` provides simple 2D dot plots, **PyDeck** (`pydeck.Deck`) unlocks high-performance WebGL 3D 
visualizations via Uber's `deck.gl` library.
It supports pitch, bearing, 3D extruded columns, dynamic lighting, custom camera perspectives, and interactive tooltips.

```python
st.pydeck_chart(
    pydeck_obj=None,
    use_container_width=True,
    on_select="ignore",        # Streamlit 1.35+ enables clickable selection return
    selection_mode="single-object"
)
```
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
import pydeck as pdk

# 1. Define 3D Viewport State
view_state = pdk.ViewState(
    latitude=37.7749,
    longitude=-122.4194,
    zoom=11.5,
    pitch=45,       # 3D Tilt Angle
    bearing=20      # Compass Rotation
)

# 2. Configure Deck Object with Basemap Style
deck = pdk.Deck(
    layers=[...],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v10",
    tooltip={"text": "Node: {node_id}\nTraffic: {rps} req/s"}
)

st.pydeck_chart(deck)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("PyDeck 3D Viewport Controller")
        
        c_pitch, c_bear, c_style = st.columns(3)
        with c_pitch:
            cam_pitch = st.slider("Camera Pitch (Tilt)", 0, 60, 45, key="pdk_pitch")
        with c_bear:
            cam_bearing = st.slider("Camera Bearing (Rotation)", 0, 360, 25, key="pdk_bear")
        with c_style:
            pdk_style = st.selectbox("Basemap Theme", ["dark", "light", "road"], key="pdk_theme")
            
        style_urls = {
            "dark": "mapbox://styles/mapbox/dark-v10",
            "light": "mapbox://styles/mapbox/light-v10",
            "road": "mapbox://styles/mapbox/streets-v11"
        }
        
        view_state = pdk.ViewState(
            latitude=37.7749,
            longitude=-122.4194,
            zoom=11.2,
            pitch=cam_pitch,
            bearing=cam_bearing
        )
        
        deck = pdk.Deck(
            layers=[],
            initial_view_state=view_state,
            map_style=style_urls[pdk_style]
        )
        st.pydeck_chart(deck)

# -----------------------------------------------------------------------------
# TOPIC 5: SCATTERPLOT MAPS & 3D HEXAGON DENSITY
# -----------------------------------------------------------------------------
st.header("5. Scatterplot Maps & 3D Hexagon Density", divider="blue")

st.markdown(
    r"""
PyDeck offers specialized analytical layer types:
- **`ScatterplotLayer`**: 2D/3D circular glyphs with per-point radii, borders, and RGBA fills.
- **`HexagonLayer`**: Aggregates spatial points into hexagonal bins with 3D extruded heights proportional to point density or metric sums.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import pydeck as pdk
import streamlit as st

# HexagonLayer for Spatial Density Aggregation
layer_hex = pdk.Layer(
    "HexagonLayer",
    data=df,
    get_position=["longitude", "latitude"],
    radius=250,              # Hexagon radius in meters
    elevation_scale=50,      # Vertical extrusion multiplier
    elevation_range=[0, 1000],
    extruded=True,           # Enables 3D heights
    pickable=True
)

st.pydeck_chart(pdk.Deck(
    layers=[layer_hex],
    initial_view_state=pdk.ViewState(
        latitude=37.7749, longitude=-122.4194, zoom=11, pitch=50
    ),
    tooltip={"text": "Density Count: {elevationValue}"}
))
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Layer Mode Switcher")
        
        layer_mode = st.radio(
            "Select PyDeck Visualization Layer",
            ["ScatterplotLayer (Nodes)", "HexagonLayer (3D Density Grid)"],
            horizontal=True,
            key="pdk_layer_mode_radio"
        )
        
        df_pdk = get_sample_geo_data(n_points=80)
        
        if layer_mode == "ScatterplotLayer (Nodes)":
            active_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_pdk,
                get_position=["longitude", "latitude"],
                get_color="color_rgba",
                get_radius="point_size * 10",
                radius_min_pixels=4,
                radius_max_pixels=25,
                pickable=True
            )
            ttip = {"text": "Node: {node_id}\nSeverity: {severity}\nRPS: {rps}"}
        else:
            active_layer = pdk.Layer(
                "HexagonLayer",
                data=df_pdk,
                get_position=["longitude", "latitude"],
                radius=300,
                elevation_scale=30,
                elevation_range=[0, 800],
                extruded=True,
                pickable=True
            )
            ttip = {"text": "Hexagon Cluster Count: {elevationValue}"}
            
        render_deck = pdk.Deck(
            layers=[active_layer],
            initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=11.2, pitch=45, bearing=15),
            tooltip=ttip
        )
        st.pydeck_chart(render_deck)

# -----------------------------------------------------------------------------
# TOPIC 6: GEOJSON CONCEPTS & BOUNDARY POLYGONS
# -----------------------------------------------------------------------------
st.header("6. GeoJSON Concepts & Boundary Polygons", divider="blue")

st.markdown(
    r"""
**GeoJSON** is the standard open format for representing geographic features with JSON.
It encodes `Point`, `LineString`, `Polygon`, and `FeatureCollection` structures.
PyDeck's `GeoJsonLayer` renders complex multi-polygon boundaries (e.g. delivery zones, county lines, delivery geofences).
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import pydeck as pdk
import streamlit as st

# Define synthetic GeoJSON boundary polygon
delivery_zone_geojson = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"zone_name": "Downtown Express Zone", "sla_minutes": 15},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-122.435, 37.785],
                [-122.395, 37.785],
                [-122.395, 37.765],
                [-122.435, 37.765],
                [-122.435, 37.785]
            ]]
        }
    }]
}

layer_geojson = pdk.Layer(
    "GeoJsonLayer",
    data=delivery_zone_geojson,
    opacity=0.4,
    stroked=True,
    filled=True,
    get_fill_color=[59, 130, 246, 120],  # Semi-transparent blue
    get_line_color=[30, 58, 138, 255],   # Navy border
    get_line_width=40,
    pickable=True
)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("GeoJSON Boundary Overlay Pipeline")
        
        synth_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": "High-Priority SLA Geofence", "nodes": 18},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.440, 37.795],
                        [-122.390, 37.795],
                        [-122.390, 37.760],
                        [-122.440, 37.760],
                        [-122.440, 37.795]
                    ]]
                }
            }]
        }
        
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            data=synth_geojson,
            opacity=0.35,
            stroked=True,
            filled=True,
            get_fill_color=[59, 130, 246, 140],
            get_line_color=[30, 58, 138, 255],
            get_line_width=30,
            pickable=True
        )
        
        points_layer = pdk.Layer(
            "ScatterplotLayer",
            data=get_sample_geo_data(n_points=30),
            get_position=["longitude", "latitude"],
            get_color="color_rgba",
            get_radius=120,
            pickable=True
        )
        
        st.pydeck_chart(pdk.Deck(
            layers=[geojson_layer, points_layer],
            initial_view_state=pdk.ViewState(latitude=37.778, longitude=-122.415, zoom=12, pitch=30),
            tooltip={"text": "Zone: {name}\nActive Nodes: {nodes}"}
        ))
        st.info("🗺️ Demonstrating composite layout: `GeoJsonLayer` polygon + `ScatterplotLayer` point entities.")

# -----------------------------------------------------------------------------
# TOPIC 7: INTERACTIVE MAP LAYERS (ARCLAYER & FLOW PATHS)
# -----------------------------------------------------------------------------
st.header("7. Interactive Map Layers (Multi-Layer Composition)", divider="blue")

st.markdown(
    r"""
Composite maps combine multiple visual layers simultaneously:
- **`ArcLayer`**: 3D parabolas connecting source and target coordinates (e.g. network routing, flight trajectories, logistics pipelines).
- Layer ordering in `pdk.Deck(layers=[L1, L2, L3])` renders from bottom to top.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import pydeck as pdk
import streamlit as st

# ArcLayer for Point-to-Point Routing Flows
df_arcs = pd.DataFrame({
    "source_lon": [-122.4194, -122.4194],
    "source_lat": [37.7749, 37.7749],
    "target_lon": [-122.4450, -122.3900],
    "target_lat": [37.7950, 37.7600],
    "flow_volume": [840, 1420]
})

arc_layer = pdk.Layer(
    "ArcLayer",
    data=df_arcs,
    get_source_position=["source_lon", "source_lat"],
    get_target_position=["target_lon", "target_lat"],
    get_source_color=[59, 130, 246, 200],  # Blue
    get_target_color=[239, 68, 68, 200],  # Red
    get_width=4
)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("3D Arc Routing Composite Layer")
        
        hub_lon, hub_lat = -122.4194, 37.7749
        np.random.seed(99)
        targets = get_sample_geo_data(center_lat=hub_lat, center_lon=hub_lon, n_points=12, spread=0.03)
        
        df_arcs = pd.DataFrame({
            "source_lon": [hub_lon] * len(targets),
            "source_lat": [hub_lat] * len(targets),
            "target_lon": targets["longitude"],
            "target_lat": targets["latitude"],
            "volume": targets["rps"]
        })
        
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=df_arcs,
            get_source_position=["source_lon", "source_lat"],
            get_target_position=["target_lon", "target_lat"],
            get_source_color=[16, 185, 129, 220],
            get_target_color=[239, 68, 68, 220],
            get_width="volume / 300",
            pickable=True
        )
        
        hub_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lon": hub_lon, "lat": hub_lat}]),
            get_position=["lon", "lat"],
            get_color=[255, 255, 255, 255],
            get_radius=220,
            pickable=True
        )
        
        st.pydeck_chart(pdk.Deck(
            layers=[arc_layer, hub_layer],
            initial_view_state=pdk.ViewState(latitude=hub_lat, longitude=hub_lon, zoom=12, pitch=50, bearing=30),
            tooltip={"text": "Flow Volume: {volume} pkts/s"}
        ))

# -----------------------------------------------------------------------------
# TOPIC 8: GEOGRAPHIC FILTERING (HAVERSINE RADIUS)
# -----------------------------------------------------------------------------
st.header("8. Geographic Filtering & Radial Distance Queries", divider="blue")

st.markdown(
    r"""
Interactive dashboards allow users to dynamically filter telemetry points by distance from a target coordinate.
The **Haversine formula** computes great-circle distance between two points on a sphere:

$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
"""
)

# Implementation of Haversine formula in pure Python
def haversine_distance_km(lat1, lon1, lat2, lon2):
    r = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
import numpy as np

# Dynamic Radius Filtering
max_radius_km = st.slider("Filter Radius (km)", 1.0, 10.0, 4.0)

center_lat, center_lon = 37.7749, -122.4194

# Compute distance for every point
df["dist_km"] = df.apply(
    lambda row: haversine_distance_km(
        center_lat, center_lon, row["latitude"], row["longitude"]
    ),
    axis=1
)

# Apply geographic filter
filtered_df = df[df["dist_km"] <= max_radius_km]

st.metric("Sites in Range", len(filtered_df), delta=f"{len(df) - len(filtered_df)} outside")
st.map(filtered_df)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Live Haversine Proximity Filter Engine")
        
        filter_radius = st.slider(
            "Target Search Radius ($R$ in km)",
            min_value=1.0,
            max_value=8.0,
            value=3.5,
            step=0.5,
            key="haversine_rad_slider"
        )
        
        all_points = get_sample_geo_data(n_points=60)
        c_lat, c_lon = 37.7749, -122.4194
        
        all_points["distance_km"] = all_points.apply(
            lambda r: haversine_distance_km(c_lat, c_lon, r["latitude"], r["longitude"]),
            axis=1
        )
        
        subset = all_points[all_points["distance_km"] <= filter_radius]
        
        c_tot, c_in, c_mean = st.columns(3)
        c_tot.metric("Total Monitored", f"{len(all_points)}")
        c_in.metric("Inside Radius", f"{len(subset)} Nodes")
        c_mean.metric("Avg Distance", f"{subset['distance_km'].mean():.2f} km" if not subset.empty else "0.0 km")
        
        if not subset.empty:
            st.map(subset, color="color_hex", size="point_size", zoom=12)
        else:
            st.warning("No telemetry points exist within the selected radial distance.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL SUMMARY
# -----------------------------------------------------------------------------
st.header("9. Geospatial Framework Comparison Matrix", divider="blue")

st.markdown(
    r"""
| Capability / Dimension | Native `st.map()` | PyDeck `st.pydeck_chart()` | Folium (`streamlit-folium`) |
| :--- | :--- | :--- | :--- |
| **Rendering Engine** | Mapbox + Deck.gl (Simplified) | WebGL Deck.gl Hardware Accelerated | Leaflet.js HTML/DOM |
| **3D Support** | ❌ 2D Only | ✅ Full 3D (Pitch, Bearing, Extrusions) | ❌ 2D Only |
| **Layer Diversity** | Single point scatter | Multi-layer (Scatter, Arc, Hex, GeoJSON, Heatmap) | Markers, GeoJSON, Tile layers |
| **Max Recommended Points** | ~50,000 points | >500,000 points (GPU Accelerated) | ~5,000 points (DOM limited) |
| **Complexity** | 1 line of Python | Declarative Layer & ViewState API | Object-oriented Python wrapper |
| **Selection Events** | ❌ None | ✅ `on_select="rerun"` (Streamlit 1.35+) | ✅ Click events via component state |
"""
)
