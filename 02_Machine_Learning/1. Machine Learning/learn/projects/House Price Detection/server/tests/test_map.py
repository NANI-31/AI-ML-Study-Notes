import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_map_prices_all(client):
    """Verify map prices endpoint returns locality-level statistics."""
    response = client.get("/api/v1/map/prices")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total_locations" in data
    assert data["total_locations"] > 0
    # Spot-check first item
    first = data["items"][0]
    assert "location_name" in first
    assert "median_price_inr" in first
    assert "median_price_per_sqft" in first
    assert "property_count" in first


def test_map_prices_state_filter(client):
    """Verify filtering map prices by State."""
    response = client.get("/api/v1/map/prices?state=Telangana")
    assert response.status_code == 200
    data = response.json()
    assert data["total_locations"] > 0
    for item in data["items"]:
        assert item["state"] == "Telangana"


def test_map_prices_district_level(client):
    """Verify district-level aggregation of dataset statistics."""
    response = client.get("/api/v1/map/prices?level=district")
    assert response.status_code == 200
    data = response.json()
    assert data["level"] == "district"
    assert data["total_locations"] > 0
    first = data["items"][0]
    assert first["level"] == "district"


def test_map_prices_bbox_filter(client):
    """Verify bounding box spatial filtering."""
    response = client.get("/api/v1/map/prices?min_lat=17.0&max_lat=18.0&min_lng=78.0&max_lng=79.0")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        if item["latitude"] is not None:
            assert 17.0 <= item["latitude"] <= 18.0
        if item["longitude"] is not None:
            assert 78.0 <= item["longitude"] <= 79.0


def test_map_boundaries(client):
    """Verify map boundaries returns GeoJSON FeatureCollection."""
    response = client.get("/api/v1/map/boundaries?level=locality")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) > 0
    first_feat = data["features"][0]
    assert first_feat["type"] == "Feature"
    assert "geometry" in first_feat
    assert "properties" in first_feat
