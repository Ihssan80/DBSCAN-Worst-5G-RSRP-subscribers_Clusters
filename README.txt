# DBSCAN-5G-Worst-RSRP-Clusters

## Overview

This project uses the DBSCAN algorithm to cluster 5G RSRP (Reference Signal Received Power) data for worst-case subscribers and visualize the results on an interactive map. The map includes clusters outlined with convex hull polygons and shows site coordinates with different color-coded bands. The project is built using Python and Dash, offering an interactive interface for adjusting clustering parameters and visualizing results.

## Features

- **DBSCAN Clustering:** Automatically identifies clusters of worst 5G RSRP readings.
- **Interactive Visualization:** Utilizes Plotly and Dash to provide an interactive map showing clustered points, site locations, and convex hull polygons.
- **Real-time Clustering Adjustment:** Use sliders to dynamically adjust clustering parameters and see the changes reflected on the map.
- **KML Export:** Allows exporting the clustered data, site locations, and polygons to a KML file for further analysis in GIS tools like Google Earth.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Required Python Packages

Install the required packages using the `requirements.txt` file:

1. Open the Command Prompt.
2. Navigate to the project directory.
3. Run the following command:

   ```cmd
   pip install -r requirements.txt



File Structure
Ensure the following files are in the same directory as the script:

Clustering_IMSIs_worst_RSRP_5G.csv: CSV file containing the 5G RSRP data.
5G_Sites_Coord.xlsx: Excel file containing site coordinates.
Your project directory should look like this:



DBSCAN-Worst-5G-RSRP-subscribers_Clusters/
├── Clustering_IMSIs_worst_RSRP_5G.csv
├── 5G_Sites_Coord.xlsx
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

Usage
 Running the Application
1- Open the Command Prompt.

2- Navigate to the project directory:

cd path\to\DBSCAN-5G-Worst-RSRP-Clusters


3- Run the Dash application:

python app.py

4- Open your web browser and go to http://127.0.0.1:8050/ to access the interactive dashboard.

Dashboard Controls
Min Samples Slider: Adjust the minimum number of points required to form a cluster.
Cell Radius Slider: Change the cell radius (in meters) to influence how clusters are formed.
Export to KML: Click this button to export the current clusters, site locations, and polygons to a KML file.
Example Workflow
Set the Min Samples slider to 20 and the Cell Radius slider to 250 meters.
Observe how the clusters are formed on the interactive map.
Click the "Export to KML" button to download the clustered data in KML format.
Exporting to KML
The "Export to KML" button allows you to download the clustered points, site locations, and polygons as a KML file. This file can be imported into Google Earth or other GIS tools for further analysis.

Notes
Replace the Mapbox token in the fig.update_layout section with your own Mapbox access token.
The app assumes that the CSV and Excel files are in the same directory as the script.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or support, feel free to contact [ihssan.alfaqeah@gmail.com].
