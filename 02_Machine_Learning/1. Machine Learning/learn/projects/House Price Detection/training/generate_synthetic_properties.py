"""
generate_synthetic_properties.py
================================
Generates 505,000 residential property listings across all 36 Indian States/UTs,
covering all 731 Districts, 6,906 Mandals, and 9,259 Localities.

Features & pricing are directly anchored to the authentic 2025-2026 web benchmark
rates from india_districts.csv, india_subdistricts.csv, and india_localities.csv.
"""

import time
import numpy as np
import pandas as pd

def generate_properties(
    districts_path="dataset/india_districts.csv",
    subdistricts_path="dataset/india_subdistricts.csv",
    localities_path="dataset/india_localities.csv",
    states_path="dataset/india_states.csv",
    output_path="dataset/india_house_prices_synthetic.csv",
    total_records=505000,
    random_seed=42
):
    start_time = time.time()
    np.random.seed(random_seed)
    print(f"Starting generation of {total_records:,} property listings...")

    # 1. Load geographic reference tables
    districts = pd.read_csv(districts_path)
    subdistricts = pd.read_csv(subdistricts_path)
    localities = pd.read_csv(localities_path)
    states = pd.read_csv(states_path)

    print(f"Loaded {len(states)} States, {len(districts)} Districts, {len(subdistricts)} Mandals, {len(localities)} Localities.")

    # 2. Merge geographic hierarchy
    # Localities -> Subdistricts -> Districts -> States
    geo = localities.merge(
        subdistricts[['subdistrict_id', 'district_id', 'subdistrict_name']],
        on='subdistrict_id',
        how='left'
    )
    geo = geo.merge(
        districts[['district_id', 'state_id', 'district_name', 'district_code', 'tier']],
        on='district_id',
        how='left'
    )
    geo = geo.merge(
        states[['state_id', 'state_name', 'state_code']],
        on='state_id',
        how='left'
    )

    n_geo = len(geo)
    print(f"Merged reference localities: {n_geo:,}")

    # 3. Compute sampling weights based on City Tier and Urbanization
    tier_weights = {
        'Tier-1': 8.0,
        'Tier-2': 3.5,
        'Tier-3': 1.0
    }
    urban_weights = {
        'Urban': 3.0,
        'Semi-Urban': 1.5,
        'Rural': 0.8
    }

    raw_weights = np.array([
        tier_weights.get(row.tier, 1.0) * urban_weights.get(row.urban_rural, 1.0)
        for _, row in geo[['tier', 'urban_rural']].iterrows()
    ])
    sampling_probs = raw_weights / raw_weights.sum()

    # 4. Sample geographic locations for all total_records
    # Ensure every single district in India (all 731) has guaranteed base listings
    min_per_district = 60
    district_indices_map = geo.groupby('district_id').indices
    guaranteed_indices = []
    for d_id, idxs in district_indices_map.items():
        guaranteed_indices.extend(np.random.choice(idxs, size=min_per_district, replace=True))
    
    guaranteed_indices = np.array(guaranteed_indices)
    remaining_count = total_records - len(guaranteed_indices)
    print(f"Guaranteed base coverage: {len(guaranteed_indices):,} properties across {len(district_indices_map)} districts.")

    # Sample remaining records using tier & urbanization market weights
    market_indices = np.random.choice(n_geo, size=remaining_count, p=sampling_probs)
    sampled_indices = np.concatenate([guaranteed_indices, market_indices])
    np.random.shuffle(sampled_indices)

    sampled_geo = geo.iloc[sampled_indices].reset_index(drop=True)

    print("Geographic location sampling complete. Generating physical property features...")

    # 5. Generate Physical Property Attributes
    # Property Types
    prop_types = ['Apartment', 'Independent House', 'Builder Floor', 'Villa', 'Residential Plot']
    prop_type_probs = [0.55, 0.18, 0.14, 0.07, 0.06]
    property_type = np.random.choice(prop_types, size=total_records, p=prop_type_probs)

    # Listing types
    listing_type = np.random.choice(['Sale', 'Resale', 'New Booking'], size=total_records, p=[0.52, 0.35, 0.13])

    # Status
    property_status = np.random.choice(['Ready to Move', 'Under Construction'], size=total_records, p=[0.76, 0.24])

    # Transaction Type
    transaction_type = np.random.choice(['Freehold', 'Leasehold', 'Power of Attorney'], size=total_records, p=[0.88, 0.09, 0.03])

    # Built up area (sqft) - log-normal distribution centered around 1400 sqft
    built_up_area = np.round(np.random.lognormal(mean=7.20, sigma=0.45, size=total_records), -1)
    built_up_area = np.clip(built_up_area, 480.0, 5200.0)

    # Carpet area (78% to 88% of built up area)
    carpet_ratio = np.random.uniform(0.78, 0.88, size=total_records)
    carpet_area = np.round(built_up_area * carpet_ratio, 1)

    # Bedrooms (BHK) correlated with built_up_area
    bedrooms = np.zeros(total_records, dtype=int)
    for i in range(total_records):
        area = built_up_area[i]
        ptype = property_type[i]
        if ptype == 'Residential Plot':
            bedrooms[i] = 0
        elif area < 800:
            bedrooms[i] = 1
        elif area < 1350:
            bedrooms[i] = 2
        elif area < 2200:
            bedrooms[i] = 3
        elif area < 3200:
            bedrooms[i] = 4
        else:
            bedrooms[i] = 5

    # Bathrooms: typically equal to bedrooms, occasionally +1 or -1
    bathrooms = np.zeros(total_records, dtype=int)
    for i in range(total_records):
        b = bedrooms[i]
        if b == 0:
            bathrooms[i] = 0
        elif b == 1:
            bathrooms[i] = np.random.choice([1, 2], p=[0.75, 0.25])
        else:
            bathrooms[i] = np.random.choice([b - 1, b, b + 1], p=[0.15, 0.70, 0.15])
    bathrooms = np.clip(bathrooms, 0, 6)

    # Balconies (0 to 4)
    balconies = np.random.choice([0.0, 1.0, 2.0, 3.0, 4.0], size=total_records, p=[0.15, 0.40, 0.30, 0.12, 0.03])

    # Floors & Total Floors
    total_floors = np.zeros(total_records, dtype=int)
    floor_number = np.zeros(total_records, dtype=int)
    for i in range(total_records):
        ptype = property_type[i]
        if ptype == 'Residential Plot':
            total_floors[i] = 1
            floor_number[i] = 0
        elif ptype in ['Independent House', 'Villa']:
            tf = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            total_floors[i] = tf
            floor_number[i] = np.random.randint(0, tf)
        elif ptype == 'Builder Floor':
            tf = np.random.choice([3, 4, 5], p=[0.4, 0.4, 0.2])
            total_floors[i] = tf
            floor_number[i] = np.random.randint(1, tf + 1)
        else:  # Apartment
            tier = sampled_geo.iloc[i]['tier']
            if tier == 'Tier-1':
                tf = np.random.randint(8, 35)
            elif tier == 'Tier-2':
                tf = np.random.randint(5, 20)
            else:
                tf = np.random.randint(4, 12)
            total_floors[i] = tf
            floor_number[i] = np.random.randint(1, tf + 1)

    # Property age (years)
    property_age = np.zeros(total_records, dtype=int)
    for i in range(total_records):
        if property_status[i] == 'Under Construction' or listing_type[i] == 'New Booking':
            property_age[i] = 0
        else:
            property_age[i] = np.random.choice(
                [0, 1, 2, 3, 5, 8, 12, 15, 20, 25],
                p=[0.20, 0.15, 0.15, 0.15, 0.12, 0.10, 0.06, 0.04, 0.02, 0.01]
            )

    # Furnishing status
    furnishing_status = np.random.choice(
        ['Unfurnished', 'Semi-Furnished', 'Fully Furnished'],
        size=total_records,
        p=[0.35, 0.45, 0.20]
    )

    # Parking spaces
    parking_spaces = np.random.choice([0.0, 1.0, 2.0, 3.0], size=total_records, p=[0.10, 0.60, 0.25, 0.05])

    # Facing direction
    facing_direction = np.random.choice(
        ['East', 'North', 'North-East', 'West', 'South', 'South-East', 'North-West', 'South-West'],
        size=total_records,
        p=[0.25, 0.22, 0.18, 0.12, 0.08, 0.06, 0.05, 0.04]
    )

    # Road width ft
    road_width_ft = np.random.choice([20.0, 30.0, 40.0, 60.0, 80.0], size=total_records, p=[0.25, 0.40, 0.20, 0.10, 0.05])

    # Gated community
    gated_community = np.where(
        np.isin(property_type, ['Apartment', 'Villa']),
        np.random.choice([1, 0], size=total_records, p=[0.75, 0.25]),
        np.random.choice([1, 0], size=total_records, p=[0.30, 0.70])
    )

    # Construction Quality & Condition
    construction_quality = np.random.choice(
        ['Standard', 'Premium', 'Economy', 'Luxury'],
        size=total_records,
        p=[0.48, 0.25, 0.18, 0.09]
    )
    property_condition = np.where(
        property_age == 0,
        'New',
        np.random.choice(['Excellent', 'Good', 'Needs Renovation'], size=total_records, p=[0.35, 0.55, 0.10])
    )

    # Land Area for plots/independent houses/villas
    land_area = np.full(total_records, np.nan)
    for i in range(total_records):
        if property_type[i] in ['Residential Plot', 'Independent House', 'Villa']:
            land_area[i] = np.round(built_up_area[i] * np.random.uniform(0.95, 1.4), 0)

    # Distances (anchored to locality infrastructure features + realistic spatial variation)
    base_airport = sampled_geo['airport_dist_base'].values
    base_railway = sampled_geo['railway_dist_base'].values
    base_hospital = sampled_geo['hospital_dist_base'].values
    base_school = sampled_geo['school_dist_base'].values
    base_shopping = sampled_geo['shopping_dist_base'].values
    has_metro = sampled_geo['has_metro'].values

    jitter = np.random.uniform(0.85, 1.15, size=total_records)
    dist_airport = np.round(np.clip(base_airport * jitter, 2.0, 160.0), 1)
    dist_railway = np.round(np.clip(base_railway * jitter, 0.8, 80.0), 1)
    dist_hospital = np.round(np.clip(base_hospital * jitter, 0.3, 25.0), 1)
    dist_school = np.round(np.clip(base_school * jitter, 0.2, 15.0), 1)
    dist_shopping = np.round(np.clip(base_shopping * jitter, 0.3, 20.0), 1)

    dist_city_center = np.round(np.clip(base_railway * 1.2 + np.random.uniform(-1.0, 2.0, size=total_records), 0.5, 45.0), 1)
    dist_main_road = np.round(np.random.exponential(scale=1.2, size=total_records) + 0.2, 1)
    dist_main_road = np.clip(dist_main_road, 0.1, 8.0)
    dist_highway = np.round(np.clip(dist_main_road * 2.5 + np.random.uniform(0.5, 3.0, size=total_records), 0.5, 25.0), 1)

    dist_metro = np.zeros(total_records)
    for i in range(total_records):
        if has_metro[i]:
            dist_metro[i] = np.round(np.random.uniform(0.5, 10.0), 1)
        else:
            dist_metro[i] = np.round(np.random.uniform(25.0, 110.0), 1)

    dist_it_hub = np.round(np.clip(dist_city_center * 1.4 + np.random.uniform(1.0, 6.0, size=total_records), 1.0, 50.0), 1)

    nearby_transport = np.random.choice(['High', 'Moderate', 'Low'], size=total_records, p=[0.64, 0.29, 0.07])
    neighborhood_type = np.random.choice(
        ['Residential', 'Commercial-Residential Mixed', 'Gated Township', 'Suburban Layout', 'Rural Settlement'],
        size=total_records,
        p=[0.55, 0.20, 0.15, 0.08, 0.02]
    )

    # Coordinates with subtle locality jitter (~100 to 500 meters)
    lat_jitter = np.random.uniform(-0.005, 0.005, size=total_records)
    lon_jitter = np.random.uniform(-0.005, 0.005, size=total_records)
    latitudes = np.round(sampled_geo['latitude'].values + lat_jitter, 5)
    longitudes = np.round(sampled_geo['longitude'].values + lon_jitter, 5)

    print("Calculating realistic market valuations based on micro-market base rates and feature premiums...")

    # 6. Comprehensive Pricing Engine
    # Base price per sqft directly from the locality reference
    base_sqft = sampled_geo['base_price_sqft'].values

    # Quality delta: Luxury (+18%), Premium (+8%), Standard (0%), Economy (-10%)
    q_delta = {'Luxury': 0.18, 'Premium': 0.08, 'Standard': 0.0, 'Economy': -0.10}
    d_quality = np.array([q_delta[q] for q in construction_quality])

    # Property type delta: Villa (+12%), Independent House (+0.05), Apartment (0%), Builder Floor (-0.04), Residential Plot (-0.15)
    pt_delta = {'Villa': 0.12, 'Independent House': 0.05, 'Apartment': 0.0, 'Builder Floor': -0.04, 'Residential Plot': -0.15}
    d_ptype = np.array([pt_delta[pt] for pt in property_type])

    # Furnishing delta: Fully (+5%), Semi (+2%), Unfurnished (0%)
    f_delta = {'Fully Furnished': 0.05, 'Semi-Furnished': 0.02, 'Unfurnished': 0.0}
    d_furnish = np.array([f_delta[f] for f in furnishing_status])

    # Condition delta: New (+3%), Excellent (+2%), Good (0%), Needs Renovation (-10%)
    c_delta = {'New': 0.03, 'Excellent': 0.02, 'Good': 0.0, 'Needs Renovation': -0.10}
    d_cond = np.array([c_delta[c] for c in property_condition])

    # Age depreciation (capped at -20% max)
    d_age = -np.minimum(0.20, property_age * 0.005)

    # Gated community (+4%)
    d_gated = np.where(gated_community == 1, 0.04, 0.0)

    # Floor rise for apartments (+0.3% per floor above 3rd, max +6%)
    d_floor = np.where(
        property_type == 'Apartment',
        np.minimum(0.06, np.maximum(0, floor_number - 3) * 0.003),
        0.0
    )

    # Road width delta (+1% per 20ft above 20ft, max +3%)
    d_road = np.minimum(0.03, np.maximum(0.0, (road_width_ft - 20.0) * 0.0005))

    # Infrastructure proximity (+4% if metro within 2km)
    d_metro = np.where(dist_metro < 2.0, 0.04, 0.0)

    # Total additive adjustment factor, bounded strictly between 0.75 and 1.30
    total_delta = 1.0 + d_quality + d_ptype + d_furnish + d_cond + d_age + d_gated + d_floor + d_road + d_metro
    total_factor = np.clip(total_delta, 0.75, 1.30)

    # Market fluctuation noise (std ~ 3%)
    noise = np.random.normal(loc=0.0, scale=0.03, size=total_records)
    m_noise = np.exp(noise)

    # Calculate final price per sqft
    effective_price_per_sqft = np.round(base_sqft * total_factor * m_noise, 2)
    price_per_sqft = effective_price_per_sqft

    # Calculate total price in INR (rounded to nearest 10,000 INR)
    price_inr = np.round(built_up_area * price_per_sqft, -4)

    # Land price per sqft
    land_price_per_sqft = np.full(total_records, np.nan)
    for i in range(total_records):
        if not np.isnan(land_area[i]) and land_area[i] > 0:
            land_price_per_sqft[i] = np.round(price_inr[i] / land_area[i], 2)

    # Estimated market value in INR (slight variation +/- 2%)
    emv_variation = 1.0 + np.random.uniform(-0.02, 0.02, size=total_records)
    estimated_market_value_inr = np.round(price_inr * emv_variation, -4)

    # Generate Property IDs
    property_ids = [f"PROP-{str(i+1).zfill(7)}" for i in range(total_records)]

    # 7. Construct Final DataFrame in Exact Column Order
    columns_order = [
        'property_id', 'state', 'state_code', 'district', 'district_code',
        'mandal_taluk_tehsil', 'village_locality', 'latitude', 'longitude', 'pincode',
        'property_type', 'listing_type', 'property_status', 'transaction_type',
        'land_area_sqft', 'built_up_area_sqft', 'carpet_area_sqft',
        'bedrooms', 'bathrooms', 'balconies', 'floor_number', 'total_floors',
        'property_age_years', 'furnishing_status', 'parking_spaces',
        'facing_direction', 'road_width_ft', 'gated_community',
        'construction_quality', 'property_condition',
        'distance_to_city_center_km', 'distance_to_main_road_km',
        'distance_to_highway_km', 'distance_to_metro_km',
        'distance_to_railway_station_km', 'distance_to_airport_km',
        'distance_to_school_km', 'distance_to_hospital_km',
        'distance_to_shopping_center_km', 'distance_to_it_hub_km',
        'nearby_public_transport', 'neighborhood_type', 'urban_rural',
        'price_inr', 'price_per_sqft', 'land_price_per_sqft',
        'estimated_market_value_inr'
    ]

    df = pd.DataFrame({
        'property_id': property_ids,
        'state': sampled_geo['state_name'].values,
        'state_code': sampled_geo['state_code'].values,
        'district': sampled_geo['district_name'].values,
        'district_code': sampled_geo['district_code'].values,
        'mandal_taluk_tehsil': sampled_geo['subdistrict_name'].values,
        'village_locality': sampled_geo['locality_name'].values,
        'latitude': latitudes,
        'longitude': longitudes,
        'pincode': sampled_geo['pincode'].values,
        'property_type': property_type,
        'listing_type': listing_type,
        'property_status': property_status,
        'transaction_type': transaction_type,
        'land_area_sqft': land_area,
        'built_up_area_sqft': built_up_area,
        'carpet_area_sqft': carpet_area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'balconies': balconies,
        'floor_number': floor_number,
        'total_floors': total_floors,
        'property_age_years': property_age,
        'furnishing_status': furnishing_status,
        'parking_spaces': parking_spaces,
        'facing_direction': facing_direction,
        'road_width_ft': road_width_ft,
        'gated_community': gated_community,
        'construction_quality': construction_quality,
        'property_condition': property_condition,
        'distance_to_city_center_km': dist_city_center,
        'distance_to_main_road_km': dist_main_road,
        'distance_to_highway_km': dist_highway,
        'distance_to_metro_km': dist_metro,
        'distance_to_railway_station_km': dist_railway,
        'distance_to_airport_km': dist_airport,
        'distance_to_school_km': dist_school,
        'distance_to_hospital_km': dist_hospital,
        'distance_to_shopping_center_km': dist_shopping,
        'distance_to_it_hub_km': dist_it_hub,
        'nearby_public_transport': nearby_transport,
        'neighborhood_type': neighborhood_type,
        'urban_rural': sampled_geo['urban_rural'].values,
        'price_inr': price_inr,
        'price_per_sqft': price_per_sqft,
        'land_price_per_sqft': land_price_per_sqft,
        'estimated_market_value_inr': estimated_market_value_inr
    })[columns_order]

    print(f"Writing {len(df):,} records to {output_path}...")
    df.to_csv(output_path, index=False)
    elapsed = time.time() - start_time
    print(f"Successfully generated and saved {len(df):,} properties in {elapsed:.2f} seconds.")
    return df

if __name__ == "__main__":
    generate_properties()
