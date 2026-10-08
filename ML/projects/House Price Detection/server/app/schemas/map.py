from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class MapLocationInfo(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    subdistrict: Optional[str] = None
    village: Optional[str] = None


class MapLocationStatistics(BaseModel):
    property_count: int = Field(..., description="Number of observed residential properties in the dataset")
    median_price_inr: float = Field(..., description="Median property valuation in INR")
    median_price_per_sqft: float = Field(..., description="Median price per square foot in INR")
    minimum_price_inr: Optional[float] = Field(None, description="Minimum observed property price in INR")
    maximum_price_inr: Optional[float] = Field(None, description="Maximum observed property price in INR")


class MapPriceItem(BaseModel):
    location_id: Optional[str] = Field(None, description="Administrative location identifier", examples=["TS_RR"])
    location_name: str = Field(..., description="Location display name", examples=["Rangareddy"])
    level: str = Field(..., description="Administrative tier (state, district, subdistrict, locality)", examples=["district"])
    state: Optional[str] = None
    district: Optional[str] = None
    subdistrict: Optional[str] = None
    village: Optional[str] = None
    property_count: int = Field(..., description="Observed dataset sample count")
    median_price_inr: float = Field(..., description="Dataset median price in INR")
    median_price_per_sqft: float = Field(..., description="Dataset median rate per sqft")
    minimum_price_inr: Optional[float] = None
    maximum_price_inr: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class MapPriceResponse(BaseModel):
    items: List[MapPriceItem]
    level: str = Field(..., description="Aggregation level returned", examples=["locality", "district"])
    total_locations: int = Field(..., description="Total count of aggregated locations in result")


class MapSinglePriceResponse(BaseModel):
    location: MapLocationInfo
    statistics: MapLocationStatistics


class MapGeometry(BaseModel):
    type: str = Field(..., examples=["Point", "Polygon"])
    coordinates: Any


class MapBoundaryFeature(BaseModel):
    type: str = Field("Feature")
    geometry: Optional[MapGeometry] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class MapBoundaryResponse(BaseModel):
    type: str = Field("FeatureCollection")
    features: List[MapBoundaryFeature]
    level: Optional[str] = None
    count: int = 0
