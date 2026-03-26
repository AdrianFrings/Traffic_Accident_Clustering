import pandas as pd
clusters = pd.read_parquet('data/processed/clusters.parquet')
points = pd.read_parquet('data/processed/cluster_points.parquet')
frankfurt_c = clusters[(clusters['City'] == 'Frankfurt am Main') & (clusters['Mode'] == 'Bicycle')]
frankfurt_p = points[(points['City'] == 'Frankfurt am Main') & (points['Mode'] == 'Bicycle')]

mismatch = False
for _, row in frankfurt_c.iterrows():
    cid = row['Cluster_ID']
    p_count = len(frankfurt_p[frankfurt_p['Cluster_ID'] == cid])
    if row['AccidentCount'] != p_count:
        print(f"Mismatch for Cluster {cid}! Count in clusters: {row['AccidentCount']}, Points: {p_count}")
        mismatch = True

if not mismatch:
    print("All cluster counts match exactly!")
