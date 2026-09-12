
import folium
from folium.plugins import MarkerCluster, HeatMap

def create_map(center=[28.6139, 77.2090], zoom_start=11):
    """
    Creates base Folium map centered on target coordinates (PDF Module 10).
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    return m

def add_heatmap(m, df):
    """
    Adds spatial heatmap layer for trajectory GPS points (PDF Module 10).
    """
    if "Latitude" in df.columns and "Longitude" in df.columns:
        heat_data = df[["Latitude", "Longitude"]].dropna().values.tolist()
        if len(heat_data) > 0:
            HeatMap(heat_data, radius=12, blur=15, min_opacity=0.4).add_to(m)
    return m

def add_points(m, df, color="blue", popup_col=None):
    """
    Adds marker cluster for trajectory points.
    """
    group = MarkerCluster(name="GPS Trajectory Pings").add_to(m)
    for _, row in df.iterrows():
        popup = str(row[popup_col]) if popup_col and popup_col in row else None
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup
        ).add_to(group)
    return m

def add_search_radii(m, lat, lon, case_id="Case"):
    """
    Adds tactical 1km, 5km, and 10km search priority radius circles around Last Known Location (PDF Module 10).
    """
    # 1km Circle - Critical Immediate Radius
    folium.Circle(
        location=[lat, lon],
        radius=1000,
        color="red",
        weight=2,
        fill=True,
        fill_opacity=0.15,
        popup=f"Immediate Search Zone (1 km) - {case_id}"
    ).add_to(m)
    
    # 5km Circle - High Priority Zone
    folium.Circle(
        location=[lat, lon],
        radius=5000,
        color="orange",
        weight=1.5,
        fill=True,
        fill_opacity=0.08,
        popup=f"High Priority Transit Buffer (5 km) - {case_id}"
    ).add_to(m)
    
    # 10km Circle - Regional Perimeter
    folium.Circle(
        location=[lat, lon],
        radius=10000,
        color="blue",
        weight=1,
        fill=False,
        popup=f"Outer Patrol Perimeter (10 km) - {case_id}"
    ).add_to(m)
    
    # Last Known Location Marker
    folium.Marker(
        location=[lat, lon],
        popup=f"LAST KNOWN SIGNAL LOCATION: {case_id}",
        tooltip="Last Seen Location",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    return m

def add_route_polyline(m, route_coords, label="Probable Route"):
    """
    Adds movement route polyline connecting sequential coordinates (PDF Module 7 & 10).
    """
    if len(route_coords) > 1:
        folium.PolyLine(
            locations=route_coords,
            color="darkpurple",
            weight=4,
            opacity=0.8,
            tooltip=label
        ).add_to(m)
    return m

