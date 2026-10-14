import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_get_states(client):
    """Verify states list returns all 36 Indian States and Union Territories."""
    response = client.get("/api/v1/locations/states")
    assert response.status_code == 200
    data = response.json()
    assert "states" in data
    assert data["total"] >= 36
    state_names = [s["name"] for s in data["states"]]
    assert "Telangana" in state_names
    assert "Andhra Pradesh" in state_names
    assert "Maharashtra" in state_names


def test_get_districts_by_state(client):
    """Verify fetching districts for Telangana."""
    response = client.get("/api/v1/locations/states/Telangana/districts")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "Telangana"
    assert data["total"] > 0
    district_names = [d["name"] for d in data["districts"]]
    assert "Hyderabad" in district_names or "Ranga Reddy" in district_names


def test_get_districts_invalid_state(client):
    """Verify 404 for non-existent state."""
    response = client.get("/api/v1/locations/states/NonExistentStateXYZ/districts")
    assert response.status_code == 404


def test_get_subdistricts_by_district(client):
    """Verify fetching subdistricts for Ranga Reddy / Rangareddy."""
    response = client.get("/api/v1/locations/districts/Rangareddy/subdistricts")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    sub_names = [s["name"] for s in data["subdistricts"]]
    assert any("Serilingampally" in s or "Gandipet" in s for s in sub_names)


def test_get_villages_by_subdistrict(client):
    """Verify fetching villages / localities under Serilingampally."""
    response = client.get("/api/v1/locations/subdistricts/Serilingampally/villages")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    v_names = [v["name"] for v in data["villages"]]
    assert any("Gachibowli" in v or "Madhapur" in v for v in v_names)


def test_reverse_geocode_valid_coordinates(client):
    """Verify reverse geocoding near Gachibowli / Ranga Reddy (17.4401, 78.3489)."""
    response = client.get("/api/v1/locations/reverse-geocode?latitude=17.4401&longitude=78.3489")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "Telangana"
    assert data["district"] == "Ranga Reddy"
    assert data["subdistrict"] == "Serilingampally"
    assert "Gachibowli" in data["village"]
    assert data["distance_km"] is not None
    assert data["distance_km"] < 5.0  # Close proximity
    assert data["state_detail"]["id"] == "TG"
    assert data["district_detail"]["name"] == "Ranga Reddy"


def test_reverse_geocode_latitude_out_of_bounds(client):
    """Verify HTTP 422 when latitude > 90."""
    response = client.get("/api/v1/locations/reverse-geocode?latitude=95.0&longitude=78.558")
    assert response.status_code == 422


def test_reverse_geocode_longitude_out_of_bounds(client):
    """Verify HTTP 422 when longitude > 180."""
    response = client.get("/api/v1/locations/reverse-geocode?latitude=17.199&longitude=190.0")
    assert response.status_code == 422


def test_reverse_geocode_outside_india(client):
    """Verify HTTP 404 when coordinates fall outside supported Indian coverage."""
    response = client.get("/api/v1/locations/reverse-geocode?latitude=51.5074&longitude=-0.1278")  # London
    assert response.status_code == 404
    err = response.json()
    assert "detail" in err
