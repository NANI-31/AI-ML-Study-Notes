import json
import pandas as pd
import numpy as np

# Direct relative string paths (no os, no pathlib)
dataset_dir = "app/dataset"
states_csv = "app/dataset/india_states.csv"
districts_csv = "app/dataset/india_districts.csv"
subdistricts_csv = "app/dataset/india_subdistricts.csv"
localities_csv = "app/dataset/india_localities.csv"
houses_csv = "app/dataset/india_house_prices_synthetic.csv"

out_dir = "app/data/locations"

# 1. Load authoritative reference CSVs
states_df = pd.read_csv(states_csv)
districts_df = pd.read_csv(districts_csv)
subdistricts_df = pd.read_csv(subdistricts_csv)
localities_df = pd.read_csv(localities_csv)

# Build mappings
state_name_map = dict(zip(states_df['state_id'], states_df['state_name']))
dist_name_map = dict(zip(districts_df['district_id'], districts_df['district_name']))
dist_to_state = dict(zip(districts_df['district_id'], districts_df['state_id']))
sub_name_map = dict(zip(subdistricts_df['subdistrict_id'], subdistricts_df['subdistrict_name']))
sub_to_dist = dict(zip(subdistricts_df['subdistrict_id'], subdistricts_df['district_id']))

# 2. Export states.json
states_data = []
for _, row in states_df.iterrows():
    states_data.append({
        'id': str(row['state_code']),
        'state_id': int(row['state_id']),
        'name': str(row['state_name']),
        'code': str(row['state_code']),
        'capital': str(row['capital']),
        'region': str(row['region'])
    })
with open(f"{out_dir}/states.json", 'w', encoding='utf-8') as f:
    json.dump(states_data, f, indent=2)

# 3. Export districts.json
districts_data = []
for _, row in districts_df.iterrows():
    s_id = int(row['state_id'])
    districts_data.append({
        'id': str(row['district_code']),
        'district_id': int(row['district_id']),
        'state_id': s_id,
        'state_name': state_name_map.get(s_id, ''),
        'name': str(row['district_name']),
        'tier': str(row['tier']),
        'headquarters': str(row['headquarters'])
    })
with open(f"{out_dir}/districts.json", 'w', encoding='utf-8') as f:
    json.dump(districts_data, f, indent=2)

# 4. Export subdistricts.json
subdistricts_data = []
for _, row in subdistricts_df.iterrows():
    d_id = int(row['district_id'])
    s_id = dist_to_state.get(d_id)
    subdistricts_data.append({
        'id': f"SUB_{row['subdistrict_id']}",
        'subdistrict_id': int(row['subdistrict_id']),
        'district_id': d_id,
        'district_name': dist_name_map.get(d_id, ''),
        'state_name': state_name_map.get(s_id, '') if s_id else '',
        'name': str(row['subdistrict_name']),
        'type': str(row['subdistrict_type'])
    })
with open(f"{out_dir}/subdistricts.json", 'w', encoding='utf-8') as f:
    json.dump(subdistricts_data, f, indent=2)

# 5. Export villages.json (localities)
villages_data = []
for _, row in localities_df.iterrows():
    sub_id = int(row['subdistrict_id'])
    d_id = sub_to_dist.get(sub_id)
    s_id = dist_to_state.get(d_id) if d_id else None
    villages_data.append({
        'id': f"LOC_{row['locality_id']}",
        'locality_id': int(row['locality_id']),
        'subdistrict_id': sub_id,
        'district_id': d_id,
        'mandal_name': sub_name_map.get(sub_id, ''),
        'district_name': dist_name_map.get(d_id, '') if d_id else '',
        'state_name': state_name_map.get(s_id, '') if s_id else '',
        'name': str(row['locality_name']),
        'pincode': str(row['pincode']),
        'latitude': float(row['latitude']),
        'longitude': float(row['longitude']),
        'urban_rural': str(row['urban_rural']),
        'base_price_sqft': float(row['base_price_sqft'])
    })
with open(f"{out_dir}/villages.json", 'w', encoding='utf-8') as f:
    json.dump(villages_data, f, indent=2)

# 6. Aggregate real observed dataset statistics from india_house_prices_synthetic.csv
try:
    df = pd.read_csv(houses_csv)
    if 'price_per_sqft' not in df.columns and 'price_inr' in df.columns:
        df['price_per_sqft'] = df['price_inr'] / df['built_up_area_sqft']
    
    loc_stats = {}
    for (state, dist, sub, village), g in df.groupby(['state', 'district', 'mandal_taluk_tehsil', 'village_locality']):
        key = f"{state}|{dist}|{sub}|{village}".lower()
        loc_stats[key] = {
            'state': state,
            'district': dist,
            'subdistrict': sub,
            'village': village,
            'property_count': int(len(g)),
            'median_price_inr': round(float(g['price_inr'].median()), 2),
            'median_price_per_sqft': round(float(g['price_per_sqft'].median()), 2),
            'minimum_price_inr': round(float(g['price_inr'].min()), 2),
            'maximum_price_inr': round(float(g['price_inr'].max()), 2),
            'latitude': round(float(g['latitude'].median()), 5) if 'latitude' in g.columns else None,
            'longitude': round(float(g['longitude'].median()), 5) if 'longitude' in g.columns else None
        }

    dist_stats = {}
    for (state, dist), g in df.groupby(['state', 'district']):
        key = f"{state}|{dist}".lower()
        dist_stats[key] = {
            'state': state,
            'district': dist,
            'property_count': int(len(g)),
            'median_price_inr': round(float(g['price_inr'].median()), 2),
            'median_price_per_sqft': round(float(g['price_per_sqft'].median()), 2),
            'minimum_price_inr': round(float(g['price_inr'].min()), 2),
            'maximum_price_inr': round(float(g['price_inr'].max()), 2)
        }

    with open(f"{out_dir}/location_statistics.json", 'w', encoding='utf-8') as f:
        json.dump({'localities': loc_stats, 'districts': dist_stats}, f, indent=2)
except FileNotFoundError:
    print(f"Houses CSV not found at '{houses_csv}', skipping aggregation.")

print("Export completed successfully.")
