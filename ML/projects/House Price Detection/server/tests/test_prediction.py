import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """
    TestClient context manager that triggers FastAPI lifespan events (model loading).
    """
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verify root endpoint returns welcome message and version."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_endpoint(client):
    """Verify health endpoint confirms model is loaded."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_valid_apartment(client):
    """Test price prediction for a typical 2 BHK urban apartment."""
    payload = {
        "state": "Telangana",
        "district": "Ranga Reddy",
        "mandal_taluk_tehsil": "Serilingampally",
        "village_locality": "Gachibowli",
        "pincode": "500032",
        "property_type": "Apartment",
        "listing_type": "Sale",
        "property_status": "Ready to Move",
        "transaction_type": "Freehold",
        "built_up_area_sqft": 1250.0,
        "carpet_area_sqft": 1020.0,
        "bedrooms": 2,
        "bathrooms": 2,
        "balconies": 1.0,
        "floor_number": 5,
        "total_floors": 14,
        "property_age_years": 3,
        "furnishing_status": "Semi-Furnished",
        "parking_spaces": 1.0,
        "facing_direction": "East",
        "road_width_ft": 40.0,
        "gated_community": 1,
        "construction_quality": "Premium",
        "property_condition": "New",
        "distance_to_city_center_km": 12.0,
        "distance_to_main_road_km": 0.4,
        "distance_to_highway_km": 2.5,
        "distance_to_metro_km": 1.5,
        "distance_to_railway_station_km": 14.0,
        "distance_to_airport_km": 28.0,
        "distance_to_school_km": 1.0,
        "distance_to_hospital_km": 1.5,
        "distance_to_shopping_center_km": 1.2,
        "distance_to_it_hub_km": 1.0,
        "nearby_public_transport": "High",
        "neighborhood_type": "Residential",
        "urban_rural": "Urban"
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "prediction" in data
    pred = data["prediction"]
    assert "price_inr" in pred
    assert "price_lakh" in pred
    assert "price_crore" in pred

    # Mathematical consistency checks
    assert pred["price_inr"] > 0
    assert abs(pred["price_lakh"] - (pred["price_inr"] / 100000.0)) < 0.01
    assert abs(pred["price_crore"] - (pred["price_inr"] / 10000000.0)) < 0.01

    assert "model" in data
    assert "name" in data["model"]


def test_prediction_valid_villa(client):
    """Test price prediction for an independent villa in Bengaluru."""
    payload = {
        "state": "Karnataka",
        "district": "Bengaluru Urban",
        "property_type": "Villa",
        "listing_type": "New Booking",
        "property_status": "Under Construction",
        "transaction_type": "Freehold",
        "built_up_area_sqft": 3000.0,
        "carpet_area_sqft": 2500.0,
        "bedrooms": 4,
        "bathrooms": 4,
        "balconies": 2.0,
        "floor_number": 0,
        "total_floors": 2,
        "property_age_years": 0,
        "furnishing_status": "Unfurnished",
        "parking_spaces": 2.0,
        "facing_direction": "North-East",
        "road_width_ft": 60.0,
        "gated_community": 1,
        "construction_quality": "Luxury",
        "property_condition": "New",
        "distance_to_city_center_km": 18.0,
        "distance_to_main_road_km": 0.2,
        "distance_to_highway_km": 1.5,
        "distance_to_metro_km": 2.0,
        "distance_to_railway_station_km": 18.0,
        "distance_to_airport_km": 40.0,
        "distance_to_school_km": 1.0,
        "distance_to_hospital_km": 2.0,
        "distance_to_shopping_center_km": 2.5,
        "distance_to_it_hub_km": 1.5,
        "nearby_public_transport": "High",
        "neighborhood_type": "Gated Township",
        "urban_rural": "Urban"
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"]["price_inr"] > 0


def test_prediction_invalid_negative_area(client):
    """Verify 422 Unprocessable Entity when built_up_area_sqft <= 0."""
    payload = {
        "state": "Maharashtra",
        "district": "Mumbai Suburban",
        "property_type": "Apartment",
        "built_up_area_sqft": -500.0,  # Invalid
        "bedrooms": 2,
        "bathrooms": 2,
        "distance_to_city_center_km": 5.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_prediction_invalid_carpet_greater_than_builtup(client):
    """Verify validation error when carpet_area_sqft > built_up_area_sqft."""
    payload = {
        "state": "Maharashtra",
        "district": "Mumbai Suburban",
        "property_type": "Apartment",
        "built_up_area_sqft": 1000.0,
        "carpet_area_sqft": 1500.0,  # Invalid
        "bedrooms": 2,
        "bathrooms": 2,
        "distance_to_city_center_km": 5.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_prediction_invalid_floor_greater_than_total(client):
    """Verify validation error when floor_number > total_floors."""
    payload = {
        "state": "Delhi",
        "district": "South Delhi",
        "property_type": "Builder Floor",
        "built_up_area_sqft": 1200.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "floor_number": 10,
        "total_floors": 4,  # Invalid: floor 10 in 4 floor building
        "distance_to_city_center_km": 8.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_prediction_with_coordinates(client):
    """Verify prediction succeeds and correctly processes optional latitude and longitude coordinates."""
    payload = {
        "state": "Telangana",
        "district": "Ranga Reddy",
        "mandal_taluk_tehsil": "Serilingampally",
        "village_locality": "Gachibowli",
        "latitude": 17.4401,
        "longitude": 78.3489,
        "property_type": "Apartment",
        "built_up_area_sqft": 1500.0,
        "bedrooms": 3,
        "bathrooms": 3,
        "distance_to_city_center_km": 10.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"]["price_inr"] > 0
