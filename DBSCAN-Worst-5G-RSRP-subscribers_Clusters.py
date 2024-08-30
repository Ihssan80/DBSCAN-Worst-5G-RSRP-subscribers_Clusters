
#Load map
# http://127.0.0.1:8050/

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Polygon, Point
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import simplekml

# Function to convert distance in meters to degrees
def meters_to_degrees(meters):
    return meters / 111320

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load the CSV file
csv_file_path = os.path.join(script_dir, 'Clustering_IMSIs_worst_RSRP_5G.csv')
df = pd.read_csv(csv_file_path)

# Load the Excel file with site coordinates
site_file_path = os.path.join(script_dir, '5G_Sites_Coord.xlsx')
site_df = pd.read_excel(site_file_path)

# Drop missing values from the relevant columns
df_cleaned = df.dropna(subset=['Longitude', 'Latitude', '5G RSRP(dBm)'])

# Filter samples with RSRP value less than -105
filtered_data = df_cleaned[df_cleaned['5G RSRP(dBm)'] < -105]

# Extract relevant columns
data_filtered = filtered_data[['Longitude', 'Latitude']]

# The rest of your code remains the same...


# Standardize the data
scaler = StandardScaler()
scaled_data_filtered = scaler.fit_transform(data_filtered)

# Initial clustering parameters
initial_eps = meters_to_degrees(250)
initial_min_samples = 20

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=initial_eps, min_samples=initial_min_samples, metric='euclidean')
filtered_data['Cluster'] = dbscan.fit_predict(scaled_data_filtered)

# Function to create convex hull polygons
def create_polygons(clusters, min_samples):
    polygons = []
    valid_clusters = {}
    for cluster_id, points in clusters.items():
        multi_point = MultiPoint(points)
        convex_hull = multi_point.convex_hull
        if isinstance(convex_hull, Polygon):  # Ensure the convex hull is a polygon
            hull_points = [p for p in points if convex_hull.contains(Point(p))]
            if len(hull_points) >= min_samples:
                valid_clusters[cluster_id] = hull_points
                polygons.append(convex_hull)
    return valid_clusters, polygons

# Gather cluster points
clusters = {}
for idx, row in filtered_data.iterrows():
    if row['Cluster'] != -1:
        if row['Cluster'] not in clusters:
            clusters[row['Cluster']] = []
        clusters[row['Cluster']].append((row['Longitude'], row['Latitude']))

# Create polygons
valid_clusters, polygons = create_polygons(clusters, initial_min_samples)

# Debug: Print polygons
print("Polygons:", polygons)

# Prepare Plotly figure
fig = go.Figure()

# Add points to the map
fig.add_trace(go.Scattermapbox(
    lon=filtered_data['Longitude'],
    lat=filtered_data['Latitude'],
    mode='markers',
    marker=dict(size=5, color='blue'),  # Set points color to blue
    text=filtered_data.apply(lambda row: f"RSRP: {row['5G RSRP(dBm)']}", axis=1),  # Add RSRP level
    name='Points',
    showlegend=False  # Do not show in legend
))

# Define a color mapping for the bands
color_mapping = {
    'N41': 'red',
    'N77': 'blue',
    'N41&N77': 'green'
}

# Apply color mapping to the site dataframe
site_df['Color'] = site_df['band'].map(color_mapping)

# Add site points to the map with different colors based on the band
for band in color_mapping.keys():
    band_sites = site_df[site_df['band'] == band]
    fig.add_trace(go.Scattermapbox(
        lon=band_sites['Longitude'],
        lat=band_sites['Latitude'],
        mode='markers+text',
        marker=dict(size=10, color=color_mapping[band]),
        text=band_sites['SiteName'],  # Show site name
        textposition="top right",  # Position the text label
        name=f'Sites ({band})',
        showlegend=True  # Show in legend
    ))

# Add polygons to the map and ensure only one entry per cluster is shown in the legend
cluster_legend_shown = set()
for cluster_id, points in valid_clusters.items():
    lons, lats = zip(*MultiPoint(points).convex_hull.exterior.coords)
    show_legend = cluster_id not in cluster_legend_shown
    fig.add_trace(go.Scattermapbox(
        lon=lons,
        lat=lats,
        mode='lines',
        line=dict(color='green'),
        name=f'Cluster {cluster_id}',
        showlegend=show_legend  # Show in legend only once per cluster
    ))
    cluster_legend_shown.add(cluster_id)

# Update layout
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=10,
    mapbox_center={"lat": filtered_data['Latitude'].mean(), "lon": filtered_data['Longitude'].mean()},
    title="Clustered Map with Polygons",
    mapbox=dict(
        accesstoken="pk.eyJ1IjoiaWhzc2FuODAiLCJhIjoiY2xvaG80dG5wMWg0ZTJpbWU0OXpjOWZ1ZyJ9.P8rKiZGf7V4c_uHunqYWqQ"  # Replace with your Mapbox token
    ),
    height=650,
    showlegend=True
)

# Dash app for interactive sliders
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='map', figure=fig),
    html.Label('Min Samples'),
    dcc.Slider(id='min_samples_slider', min=1, max=50, step=1, value=initial_min_samples),
    html.Label('Cell Radius (meters)'),
    dcc.Slider(id='cell_radius_slider', min=100, max=2000, step=50, value=250),
    html.Button("Export to KML", id="export_button"),
    dcc.Download(id="download_kml"),
    html.Div(id='metrics_div')
])

@app.callback(
    [Output('map', 'figure'), Output('metrics_div', 'children')],
    [Input('min_samples_slider', 'value'), Input('cell_radius_slider', 'value')]
)
def update_map(min_samples, cell_radius):
    eps = meters_to_degrees(cell_radius)
    
    # Re-cluster data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    filtered_data['Cluster'] = dbscan.fit_predict(scaler.transform(data_filtered))
    
    # Gather cluster points
    clusters = {}
    for idx, row in filtered_data.iterrows():
        if row['Cluster'] != -1:
            if row['Cluster'] not in clusters:
                clusters[row['Cluster']] = []
            clusters[row['Cluster']].append((row['Longitude'], row['Latitude']))
    
    # Create polygons
    valid_clusters, polygons = create_polygons(clusters, min_samples)
    
    # Update figure
    fig = go.Figure()
    
    # Add valid cluster points to the map without showing them in the legend
    for cluster_id, points in valid_clusters.items():
        cluster_df = pd.DataFrame(points, columns=['Longitude', 'Latitude'])
        cluster_df = pd.merge(cluster_df, filtered_data[['Longitude', 'Latitude', '5G RSRP(dBm)']],
                              on=['Longitude', 'Latitude'], how='left')
        fig.add_trace(go.Scattermapbox(
            lon=cluster_df['Longitude'],
            lat=cluster_df['Latitude'],
            mode='markers',
            marker=dict(size=6, color='black'),  # Set points color to blue
            text=cluster_df.apply(lambda row: f"RSRP: {row['5G RSRP(dBm)']}", axis=1),  # Add RSRP level
            name=f'Cluster {cluster_id}',
            showlegend=False  # Do not show in legend
        ))
    
    # Add site points to the map with different colors based on the band
    for band in color_mapping.keys():
        band_sites = site_df[site_df['band'] == band]
        fig.add_trace(go.Scattermapbox(
            lon=band_sites['Longitude'],
            lat=band_sites['Latitude'],
            mode='markers+text',
            marker=dict(size=10, color=color_mapping[band]),
            text=band_sites['SiteName'],  # Show site name
            textposition="top right",  # Position the text label
            name=f'Sites ({band})',
            showlegend=True  # Show in legend
        ))

    # Add polygons to the map and ensure only one entry per cluster is shown in the legend
    cluster_legend_shown = set()
    for cluster_id, points in valid_clusters.items():
        lons, lats = zip(*MultiPoint(points).convex_hull.exterior.coords)
        show_legend = cluster_id not in cluster_legend_shown
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(color='green'),
            name=f'Cluster {cluster_id}',
            showlegend=show_legend  # Show in legend only once per cluster
        ))
        cluster_legend_shown.add(cluster_id)
    
    # Update layout
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10,
        mapbox_center={"lat": filtered_data['Latitude'].mean(), "lon": filtered_data['Longitude'].mean()},
        title="Clustered Map with Polygons",
        mapbox=dict(
            accesstoken="pk.eyJ1IjoiaWhzc2FuODAiLCJhIjoiY2xvaG80dG5wMWg0ZTJpbWU0OXpjOWZ1ZyJ9.P8rKiZGf7V4c_uHunqYWqQ"  # Replace with your Mapbox token
        ),
        height=650,
        showlegend=True
    )
    
    # Update metrics
    number_of_clusters = len(valid_clusters)
    number_of_noise_points = list(filtered_data['Cluster']).count(-1)
    metrics_text = f"""
        <b>Evaluation Metrics:</b><br>
        Number of clusters: {number_of_clusters}<br>
        Number of noise points: {number_of_noise_points}<br>
    """
    
    return fig, metrics_text

# Function to create KML structure with minimal icons and no labels
def create_kml(points, sites, polygons):
    kml = simplekml.Kml()
    
    points_folder = kml.newfolder(name="Points")
    sites_folder = kml.newfolder(name="Sites")
    
    point_style = simplekml.Style()
    point_style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    point_style.iconstyle.scale = 0.6  # Smaller scale for lighter icons
    point_style.iconstyle.color = simplekml.Color.blue  # Set color to blue
    
    site_style = simplekml.Style()
    site_style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png'
    site_style.iconstyle.scale = 0.7  # Smaller scale for lighter icons
    site_style.iconstyle.color = simplekml.Color.red  # Set color to red
    
    for point in points:
        pnt = points_folder.newpoint(coords=[(point['Longitude'], point['Latitude'])], description=f"RSRP: {point['5G RSRP(dBm)']}")
        pnt.style = point_style
    
    for idx, site in sites.iterrows():
        pnt = sites_folder.newpoint(coords=[(site['Longitude'], site['Latitude'])])
        pnt.style = site_style
    
    for polygon in polygons:
        pol = kml.newpolygon(name="Cluster", outerboundaryis=list(polygon.exterior.coords))
        pol.style.polystyle.color = simplekml.Color.changealphaint(0, simplekml.Color.green)  # Transparent fill
        pol.style.linestyle.color = simplekml.Color.white  # White outline
        pol.style.linestyle.width = 2  # Set the width of the outline
    
    return kml.kml()

@app.callback(
    Output('download_kml', 'data'),
    Input('export_button', 'n_clicks'),
    [State('map', 'figure')]
)
def export_to_kml(n_clicks, figure):
    if n_clicks is None:
        return None

    points = filtered_data[filtered_data['Cluster'] != -1]
    points_dict = points[['Longitude', 'Latitude', '5G RSRP(dBm)']].to_dict('records')

    kml_str = create_kml(points_dict, site_df, polygons)
    
    return dcc.send_string(kml_str, "data.kml")

if __name__ == '__main__':
    app.run_server(debug=True)
