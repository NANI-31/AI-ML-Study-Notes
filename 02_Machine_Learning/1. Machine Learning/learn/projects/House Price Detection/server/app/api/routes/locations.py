import logging
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from app.services.location_service import location_service
from app.schemas.location import (
    StateListResponse,
    DistrictListResponse,
    SubdistrictListResponse,
    VillageListResponse,
    ReverseLocationResponse,
)

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/locations", tags=["Locations"])


@router.get(
    "/states",
    response_model=StateListResponse,
    summary="Get all Indian States",
    description="Returns all 36 Indian States and Union Territories from the reference dataset.",
)
def get_states():
    states = location_service.get_states()
    return StateListResponse(states=states, total=len(states))


@router.get(
    "/states/{state}/districts",
    response_model=DistrictListResponse,
    summary="Get Districts by State",
    description="Returns all administrative districts belonging to the specified state code or name.",
)
def get_districts(state: str):
    districts = location_service.get_districts(state)
    return DistrictListResponse(state=state, districts=districts, total=len(districts))


@router.get(
    "/districts/{district}/subdistricts",
    response_model=SubdistrictListResponse,
    summary="Get Subdistricts / Mandals by District",
    description="Returns mandals, taluks, or tehsils belonging to the specified district.",
)
def get_subdistricts(district: str):
    subdistricts = location_service.get_subdistricts(district)
    return SubdistrictListResponse(district=district, subdistricts=subdistricts, total=len(subdistricts))


@router.get(
    "/districts/{district}/mandals",
    response_model=SubdistrictListResponse,
    summary="Alias: Get Mandals by District",
    description="Alias endpoint returning mandals / subdistricts for a district.",
    include_in_schema=False,
)
def get_mandals_alias(district: str):
    return get_subdistricts(district)


@router.get(
    "/subdistricts/{subdistrict}/villages",
    response_model=VillageListResponse,
    summary="Get Villages / Localities by Subdistrict",
    description="Returns all registered localities and villages under the selected subdistrict.",
)
def get_villages(subdistrict: str):
    villages = location_service.get_villages(subdistrict)
    return VillageListResponse(subdistrict=subdistrict, villages=villages, total=len(villages))


@router.get(
    "/mandals/{mandal}/villages",
    response_model=VillageListResponse,
    summary="Alias: Get Villages by Mandal",
    description="Alias endpoint returning localities / villages under a mandal.",
    include_in_schema=False,
)
def get_villages_alias(mandal: str):
    return get_villages(mandal)


@router.get(
    "/reverse-geocode",
    response_model=ReverseLocationResponse,
    summary="Reverse Geocode Coordinates to Administrative Hierarchy",
    description="Resolves geographic latitude and longitude into State -> District -> Subdistrict -> Village.",
)
def reverse_geocode(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Latitude coordinate between -90 and 90"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Longitude coordinate between -180 and 180"),
):
    return location_service.resolve_coordinates(latitude, longitude)


@router.get(
    "/reverse",
    response_model=ReverseLocationResponse,
    summary="Alias: Reverse Geocode",
    description="Alias endpoint for reverse geocoding coordinates.",
    include_in_schema=False,
)
def reverse_geocode_alias(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
):
    return reverse_geocode(latitude=latitude, longitude=longitude)
