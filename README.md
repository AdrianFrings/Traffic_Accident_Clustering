# Traffic Accident Risk Analysis

**Identifying High-Leverage Intervention Points for Traffic Safety using Machine Learning and Geospatial Analytics.**

## The Problem: Invisible Patterns in Static Data
Every year, Germany sees approximately 250,000 traffic accidents involving personal injury. While the German Unfallatlas (Accident Atlas) is a vital resource for mapping these incidents, it suffers from two major limitations for urban planners:
* Temporal Fragmentation: It only displays one year at a time, making it difficult to spot systemic, multi-year danger zones.
* Lack of Prioritization: It visualizes points but doesn't identify "high-leverage" clusters where infrastructure changes would yield the highest safety ROI.

## The Solution: Targeted Geospatial Intelligence
The Accident Risk Dashboard distills nearly a decade of data (2016–2024) into actionable hotspots. The tool identifies the top 3–10 highest-density "danger zones" per municipality across three categories: Car, Bicycle, and Pedestrian.

> **💡 Actionable Insight Case Study: Frankfurt am Main**
>
> *When running the tool for Frankfurt on pedestrian accidents, a cluster is identified on a street that on first glance looks unassuming. Inspecting the street closely, one can identify a bus stop right in the middle of the cluster. This suggests that the bus stop is not well designed for pedestrian safety, we see that the busstop in this street is in the middle of a street with a an akward sidewalk in the miffle of both lanes. We find that 12 accidents with injury at this exact spot were caused by crossing the road. City planners could use this information to move the busstop to a safer place. This is an example of an accident cluster that would probably stay hidden without the tool.
 When analyzing pedestrian accidents in Frankfurt, the tool identified a high-density cluster on an otherwise unassuming street. Closer inspection revealed a bus stop situated on a narrow median between two lanes.

The Finding: 12 accidents involving injury occurred at this exact coordinate, specifically caused by pedestrians crossing the road to reach the stop.
The Fix: This data provides city planners with a clear mandate to relocate the bus stop or install a protected crossing—a systemic issue that remains hidden in standard annual reports.*
![My Dashboard](/Frankfurt_example.png)

## Technical Architecture
The application is a full-stack, Dockerized data pipeline built to handle large-scale geospatial processing. It utilizes the DBSCAN clustering algorithm to distill millions of scattered data points into geographic hotspots, segregated by transport mode (Bicycle, Car, and Pedestrian).

### Why DBSCAN?
Unlike algorithms like K-Means that require a pre-defined number of clusters, DBSCAN (Density-Based Spatial Clustering of Applications with Noise) discovers them organically based on the actual geographic grouping of the data.
* How it works: The algorithm draws a search radius ($eps$) around a point. If it finds a minimum number of neighbors, it forms a "core" cluster and expands outward.
* Noise Filtering: Crucially, any isolated accidents that do not meet the density threshold are categorized as "noise" and ignored. This makes it ideal for this use case, as it naturally filters out random, one-off crashes and isolates only the true, systemic danger zones.
![My Dashboard](/dbscan_explanation.png)

### Dynamic Density Scaling 
Applying a single density threshold across an entire country is ineffective, as a setting tuned for rural towns would merge an entire city into one massive cluster. To solve this, I implemented a dynamic scaling loop:
1. The system iterates through all 2,000+ municipalities individually.
2. For each city, the algorithm runs a continuous loop, dynamically scaling the minimum required accident density (min_samples).
3. It stops once it actively suppresses background noise and isolates the top 3-to-8 most accident-dense clusters.

### DBSCAN vs. HDBSCAN
While HDBSCAN is a popular modern alternative that handles varying densities natively, I deliberately chose the classic DBSCAN to strictly enforce the $eps$ (search radius) parameter. In HDBSCAN, the spatial radius is flexible, meaning a cluster could theoretically stretch for several hundred meters. While mathematically valid, this is practically useless for a city planner. By hardcoding $eps$ to 50 meters, I forced the algorithm to obey physical reality: the core of a cluster must fit within the footprint of a single intersection or street segment. This ensures every resulting hotspot is tightly localized and highly actionable.


## Project Context & Collaboration
This dashboard is an evolution of a research project conducted with Michael Fryer and Gaziza Janabayeva, where we initially analyzed German accident data.

While this version introduces the full-stack architecture, Dockerization, and the dynamic scaling logic, it utilizes and builds upon the core clustering methodology developed during that collaboration, where I implemented the inital clustering approach.
Original Research Repo: https://github.com/HumbleHominid/urban-mobility-risk-analysis

## Technologies/Packages Used
* **ML & Data:** Scikit-Learn, Pandas, GeoPandas, Pyproj
* **Visualization:** Streamlit, Folium, Altair, Contextily
* **Infrastructure:** Docker, VPS Deployment

## Data Source
* Destatis, Unfallatlas, https://unfallatlas.statistikportal.de/, accessed on 24th March 2026; License: dl-de/by-2-0 (https://www.govdata.de/dl-de/by-2-0)
* Urban Mobility Risk Analysis, https://github.com/HumbleHominid/urban-mobility-risk-analysis, accessed on 24th March 2026