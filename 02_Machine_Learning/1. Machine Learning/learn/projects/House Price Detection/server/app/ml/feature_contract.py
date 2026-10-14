"""
Authoritative feature contract definitions for the Indian House Price Prediction ML Pipeline.
Specifies the exact 36 features expected by the trained Scikit-Learn ColumnTransformer pipeline.
"""
from typing import List, Dict, Any

MODEL_NUMERICAL_FEATURES: List[str] = [
    "built_up_area_sqft",
    "carpet_area_sqft",
    "bedrooms",
    "bathrooms",
    "balconies",
    "floor_number",
    "total_floors",
    "property_age_years",
    "parking_spaces",
    "road_width_ft",
    "latitude",
    "longitude",
    "distance_to_city_center_km",
    "distance_to_main_road_km",
    "distance_to_highway_km",
    "distance_to_metro_km",
    "distance_to_railway_station_km",
    "distance_to_airport_km",
    "distance_to_school_km",
    "distance_to_hospital_km",
    "distance_to_shopping_center_km",
    "distance_to_it_hub_km",
]

MODEL_CATEGORICAL_FEATURES: List[str] = [
    "state",
    "district",
    "property_type",
    "listing_type",
    "property_status",
    "transaction_type",
    "furnishing_status",
    "facing_direction",
    "gated_community",
    "construction_quality",
    "property_condition",
    "nearby_public_transport",
    "neighborhood_type",
    "urban_rural",
]

ALL_EXPECTED_FEATURES: List[str] = MODEL_NUMERICAL_FEATURES + MODEL_CATEGORICAL_FEATURES


def validate_feature_contract(features_dict: Dict[str, Any]) -> List[str]:
    """
    Validates whether the incoming dictionary satisfies the pipeline's feature requirements.
    Returns a list of missing feature keys.
    """
    return [col for col in ALL_EXPECTED_FEATURES if col not in features_dict]
