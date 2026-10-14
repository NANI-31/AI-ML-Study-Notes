import logging
from typing import Optional
from fastapi import APIRouter, Query
from app.services.map_service import map_service
from app.schemas.map import MapPriceResponse, MapBoundaryResponse

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/map", tags=["Map"])


@router.get(
    "/prices",
    response_model=MapPriceResponse,
    summary="Get Aggregated Observed Dataset Price Statistics",
    description=(
        "Returns observed historical dataset property statistics (median price, price per sqft, "
        "min/max range, sample count). Distinct from ML model predictions."
    ),
)
def get_map_prices(
    state: Optional[str] = Query(None, description="Filter by Indian State"),
    district: Optional[str] = Query(None, description="Filter by District"),
    subdistrict: Optional[str] = Query(None, description="Filter by Subdistrict / Mandal"),
    village: Optional[str] = Query(None, description="Filter by Village / Locality"),
    min_lat: Optional[float] = Query(None, ge=-90.0, le=90.0, description="Bounding box minimum latitude"),
    max_lat: Optional[float] = Query(None, ge=-90.0, le=90.0, description="Bounding box maximum latitude"),
    min_lng: Optional[float] = Query(None, ge=-180.0, le=180.0, description="Bounding box minimum longitude"),
    max_lng: Optional[float] = Query(None, ge=-180.0, le=180.0, description="Bounding box maximum longitude"),
    level: Optional[str] = Query(None, description="Administrative level: 'locality' or 'district'"),
    zoom: Optional[int] = Query(None, description="Current map zoom level to automatically choose level"),
):
    return map_service.get_map_prices(
        state=state,
        district=district,
        subdistrict=subdistrict,
        village=village,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lng=min_lng,
        max_lng=max_lng,
        level=level,
        zoom=zoom,
    )


@router.get(
    "/boundaries",
    response_model=MapBoundaryResponse,
    summary="Get Map Boundaries as GeoJSON FeatureCollection",
    description="Returns standard GeoJSON FeatureCollection with spatial point/polygon geometries and metadata.",
)
def get_map_boundaries(
    level: str = Query("locality", description="Administrative level ('locality', 'district')"),
    state_id: Optional[str] = Query(None, description="Filter boundaries by State ID or name"),
    district_id: Optional[str] = Query(None, description="Filter boundaries by District ID or name"),
):
    return map_service.get_map_boundaries(
        level=level,
        state_id=state_id,
        district_id=district_id,
    )
