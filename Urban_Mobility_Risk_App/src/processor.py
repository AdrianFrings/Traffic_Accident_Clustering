import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import reverse_geocoder as rg
import os
from pathlib import Path
from pyproj import Transformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_DIR = Path('data')
PROCESSED_DATA_DIR = Path('data/processed')
EPS_METERS = 50
MIN_SAMPLES_DEFAULT = 10  # Starting point for dynamic target hunting
OUTPUT_FILE = PROCESSED_DATA_DIR / 'clusters.parquet'
FEATURE_COLS = ['UKATEGORIE', 'UTYP1', 'ULICHTVERH', 'IstRad', 'IstPKW', 'IstFuss', 'LINREFX', 'LINREFY', 'UGEMEINDE', 'UKREIS', 'UREGBEZ', 'ULAND']

def load_data(years: list[int]) -> pd.DataFrame:
    """Loads and concatenates accident data for specified years."""
    dfs = []
    
    # Check if we have the city info first, needed for filtering later if we merge early
    # But for raw data loading, we just need the year CSVs
    
    for year in years:
        file_path = DATA_DIR / f"{year}.csv"
        if not file_path.exists():
            logging.warning(f"Data for year {year} not found at {file_path}. Skipping.")
            continue
            
        logging.info(f"Loading data for {year}...")
        try:
             # Using the same separator and decimal as fetch_data.py
            df = pd.read_csv(
                file_path, 
                sep=";", 
                decimal=",", 
                usecols=lambda c: c in FEATURE_COLS + ['OID_', 'UIDENTSTLAE', 'UIDENTSTLA', 'Community_key'] # Load potentially relevant ID cols
                # Note: fetch_data.py renames columns but we are reading raw here? 
                # Wait, fetch_data.py seems to be a utility to fetch AND cleaned/rename? 
                # The user said "Use cleaned data with X_Meters...". 
                # src/data/2024.csv headers looked like: OID_;...;ULAND;...;LINREFX;LINREFY...
                # So the raw CSVs are already "semi-clean" in terms of having content, but maybe not renamed?
                # Let's rely on the headers I saw: LINREFX, LINREFY, UKATEGORIE, ULICHTVERH etc.
            )
            # Just to be safe, I'll load all columns and filter later or map names if needed
            # Re-reading the fetch_data.py: it does renaming. I should probably replicate that logic or import it?
            # User said "The Offline Processor... Generate a robust script...". It implies self-contained or reusing logic.
            # I will reuse the logic but implement it here to be explicit and control the process.
        except ValueError:
             # Fallback if usecols fails
             df = pd.read_csv(file_path, sep=";", decimal=",")

        
        # Basic Renaming to match Task description if needed
        # Task says: "Load Data: Use cleaned data with X_Meters, Y_Meters..."
        # I will rename LINREFX -> X_Meters, LINREFY -> Y_Meters for clarity
        rename_map = {
            "LINREFX": "X_Meters",
            "LINREFY": "Y_Meters",
            "LICHT": "ULICHTVERH", # In case it differs
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure we have the necessary columns
        required_cols = ['X_Meters', 'Y_Meters', 'UKATEGORIE']
        if not all(col in df.columns for col in required_cols):
             logging.warning(f"Year {year} missing required columns. Available: {df.columns}. Skipping.")
             continue
             
        # Add Year column if not present
        if 'UJAHR' not in df.columns:
            df['UJAHR'] = year
            
        dfs.append(df)
        
    if not dfs:
        raise ValueError("No data loaded!")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total records loaded: {len(combined_df)}")
    return combined_df

def calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates RiskScore based on accident severity."""
    # Logic: Fatal (1) = 10, Severe (2) = 5, Minor (3) = 1
    
    conditions = [
        (df['UKATEGORIE'] == 1),
        (df['UKATEGORIE'] == 2),
        (df['UKATEGORIE'] == 3)
    ]
    scores = [10, 5, 1]
    
    df['RiskScore'] = np.select(conditions, scores, default=1)
    return df

def get_city_list() -> pd.DataFrame:
    """Loads the list of cities and their regional keys."""
    path = DATA_DIR / "city_info.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
        
    df = pd.read_csv(
        path,
        sep=";",
        converters={"regional key": lambda x: str(x)[:5] + str(x)[9:]} # Transform 12-digit AGS to 8-digit key
    )
    # The 'regional key' usually matches the start of AGS (Allgemeiner Gemeindeschlüssel)
    # in fetch_data.py: df.loc[df["ULAND"] == s, "Community_key"] = f"{s}000000"
    # and "Community_key" = df["ULAND"] + df["UREGBEZ"] + df["UKREIS"] + df["UGEMEINDE"]
    
    return df

def construct_community_key(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs the Community_key to match regional keys from city_info."""
    # Ensure columns are strings and padded
    # ULAND: 2 chars
    # UREGBEZ: 1 char
    # UKREIS: 2 chars
    # UGEMEINDE: 3 chars
    
    # Note: The CSVs might have them as ints.
    for col in ['ULAND', 'UREGBEZ', 'UKREIS', 'UGEMEINDE']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.zfill(2 if col == 'UKREIS' or col == 'ULAND' else 1 if col == 'UREGBEZ' else 3)
            # Actually ULAND is 2, UREGBEZ 1, UKREIS 2, UGEMEINDE 3 based on standard AGS
            # But let's check fetch_data.py logic:
            # df["Community_key"] = df["ULAND"] + df["UREGBEZ"] + df["UKREIS"] + df["UGEMEINDE"]
            # And handling state cities (Berlin, Hamburg etc):
            # df.loc[df["ULAND"] == s, "Community_key"] = f"{s}000000"
            pass

    # Pad correctly
    df['ULAND'] = df['ULAND'].astype(str).str.zfill(2)
    df['UREGBEZ'] = df['UREGBEZ'].astype(str) # Usually 1 digit, but let's be safe. 
    # Actually UREGBEZ can be '0' for some states.
    df['UKREIS'] = df['UKREIS'].astype(str).str.zfill(2)
    df['UGEMEINDE'] = df['UGEMEINDE'].astype(str).str.zfill(3)

    df['Community_key'] = df['ULAND'] + df['UREGBEZ'] + df['UKREIS'] + df['UGEMEINDE']
    
    # Handle City States (Berlin 11, Hamburg 02, Bremen 04 - wait Bremen is split)
    # fetch_data.py says: states = ["11", "02"] -> "000000" suffix
    # Berlin (11), Hamburg (02)
    city_states = ["11", "02"] 
    for s in city_states:
        mask = df['ULAND'] == s
        df.loc[mask, 'Community_key'] = s + "000000"
        
    return df

def cluster_city_data(city_name: str, city_df: pd.DataFrame) -> tuple:
    """Runs DBSCAN on a specific city's data, for different transport modes."""
    
    # Define transport modes to analyze
    # IstRad = Bicycle
    # IstPKW = Car
    # IstFuss = Pedestrian
    modes = {
        'Bicycle': 'IstRad',
        'Car': 'IstPKW',
        'Pedestrian': 'IstFuss',
        'All': None # Special case for all data
    }
    
    all_city_clusters = []
    all_city_points = []

    for mode_name, col_name in modes.items():
        if col_name:
            # Filter for this mode (1 means involved)
            mode_df = city_df[city_df[col_name] == 1].copy()
        else:
            mode_df = city_df.copy()
            
        if len(mode_df) < MIN_SAMPLES_DEFAULT:
            continue

        coords = mode_df[['X_Meters', 'Y_Meters']].values
        
        # Target-Hunting Loop for DBSCAN min_samples
        # We aim for ~3 to 8 clusters per mode per city to isolate the worst intersections.
        target_max_clusters = 8
        min_s = MIN_SAMPLES_DEFAULT
        labels = None
        
        while True:
            db = DBSCAN(eps=EPS_METERS, min_samples=min_s, metric='euclidean', n_jobs=-1)
            current_labels = db.fit_predict(coords)
            
            n_clusters = len(set(current_labels)) - (1 if -1 in current_labels else 0)
            
            if n_clusters <= target_max_clusters or min_s > 200:
                # If we hit 0 clusters because we incremented too far, and we had previous clusters, revert.
                # But since it's an offline script, if it drops to 0 immediately, just take 0.
                if n_clusters == 0 and labels is not None:
                    # We keep the previous labels (which had > target_max_clusters) rather than nothing.
                    pass
                else:
                    labels = current_labels
                break
                
            labels = current_labels
            min_s += 5
            
        logging.info(f"   -> Mode: {mode_name}, final min_samples: {min_s}, clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
        
        mode_df['Cluster_ID'] = labels
        mode_df['City'] = city_name
        mode_df['Mode'] = mode_name
        
        all_city_points.append(mode_df[['City', 'Mode', 'Cluster_ID', 'X_Meters', 'Y_Meters']])
        
        # Filter noise (-1)
        clusters = mode_df[mode_df['Cluster_ID'] != -1]
        
        if clusters.empty:
            continue

        # Aggregate
        # MostFreqLight logic: take mode
        
        # UTYP1 Statistics: We want a summary string or dict
        # We can store it as a JSON string for simple parquet storage
        def get_utyp1_stats(series):
            return series.value_counts().to_json()

        agg_funcs = {
            'RiskScore': 'sum',
            'City': 'count', # Count of accidents (guaranteed not null)
            'X_Meters': 'mean',
            'Y_Meters': 'mean',
            'ULICHTVERH': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            'UTYP1': get_utyp1_stats
        }
        
        cluster_stats = clusters.groupby('Cluster_ID').agg(agg_funcs).reset_index()
        cluster_stats.rename(columns={
            'City': 'AccidentCount', 
            'X_Meters': 'Centroid_X', 
            'Y_Meters': 'Centroid_Y', 
            'ULICHTVERH': 'MostFreqLight',
            'UTYP1': 'AccidentTypeStats'
        }, inplace=True)
        
        cluster_stats['City'] = city_name
        cluster_stats['Mode'] = mode_name
        
        all_city_clusters.append(cluster_stats)
    
    if not all_city_clusters:
        return pd.DataFrame(), pd.DataFrame()
        
    return pd.concat(all_city_clusters, ignore_index=True), pd.concat(all_city_points, ignore_index=True)

def transform_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Converts coordinates and adds address info."""
    if df.empty:
        return df
        
    # Project EPSG:25832 (UTM 32N) to EPSG:4326 (WGS84 Lat/Lon)
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=False) # Lat, Lon order
    
    # Transform
    # Transformer.transform takes x, y. arrays.
    # Note: always_xy=False usually returns (lat, lon). PROJ defaults can be tricky.
    # EPSG:4326 is Lat, Lon.
    lats, lons = transformer.transform(df['Centroid_X'].values, df['Centroid_Y'].values)
    
    df['Lat'] = lats
    df['Lon'] = lons
    
    # Reverse Geocode
    # rg.search takes (lat, lon) tuples
    coordinates = list(zip(df['Lat'], df['Lon']))
    results = rg.search(coordinates)
    
    # Extract meaningful address info
    # rg returns list of dicts: {'lat':..., 'lon':..., 'name': 'Street/Neighborhood', 'admin1':..., 'admin2':...}
    # 'name' is often the neighborhood or street.
    df['LocationName'] = [res['name'] for res in results]
    
    # Google Maps Link
    # https://www.google.com/maps/search/?api=1&query=50.131887668760044,8.683750031440045
    df['GoogleMapsLink'] = df.apply(lambda row: f"https://www.google.com/maps/search/?api=1&query={row['Lat']},{row['Lon']}", axis=1)
    
    return df

def main():
    if not PROCESSED_DATA_DIR.exists():
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    logging.info("Starting Offline Processor...")
    
    # 1. Load Data
    raw_data = load_data(list(range(2016, 2025)))
    
    # 2. Risk Scoring
    raw_data = calculate_risk_score(raw_data)
    
    # 3. Construct Keys for mapping
    # Check if necessary columns exist
    if 'Community_key' not in raw_data.columns:
        if all(c in raw_data.columns for c in ['ULAND', 'UREGBEZ', 'UKREIS', 'UGEMEINDE']):
            raw_data = construct_community_key(raw_data)
        else:
            logging.warning("Cannot construct community key, missing columns. Clustering might fail.")
            
    # 4. Process per City
    city_info = get_city_list()
    
    all_clusters = []
    all_points = []
    
    for _, row in city_info.iterrows():
        city_name = row['city']
        reg_key = row['regional key']
        
        # Filter data for this city
        # Note: reg_key in city_info usually matches Community_key
        # Check if reg_key needs simplified matching (e.g. starts with)
        # In fetch_data.py, they did exact match.
        
        city_mask = raw_data['Community_key'] == reg_key
        city_accidents = raw_data[city_mask]
        
        if city_accidents.empty:
            continue
            
        logging.info(f"Processing {city_name} ({len(city_accidents)} accidents)...")
        
        clusters_df, points_df = cluster_city_data(city_name, city_accidents)
        if not clusters_df.empty:
            all_clusters.append(clusters_df)
            all_points.append(points_df)
            
    if all_clusters:
        final_df = pd.concat(all_clusters, ignore_index=True)
        final_points = pd.concat(all_points, ignore_index=True)
        
        # 5. Transform and Enrich
        logging.info("Transforming coordinates and geocoding...")
        final_df = transform_and_enrich(final_df)
        
        # 6. Save
        logging.info(f"Saving {len(final_df)} clusters to {OUTPUT_FILE}...")
        final_df.to_parquet(OUTPUT_FILE)
        
        POINTS_FILE = PROCESSED_DATA_DIR / 'cluster_points.parquet'
        logging.info(f"Saving {len(final_points)} detailed points to {POINTS_FILE}...")
        final_points.to_parquet(POINTS_FILE)
        
        logging.info("Done.")
    else:
        logging.warning("No clusters generated.")

if __name__ == "__main__":
    main()
