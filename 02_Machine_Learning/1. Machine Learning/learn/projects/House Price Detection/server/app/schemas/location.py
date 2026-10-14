from typing import Optional, List
from pydantic import BaseModel, Field


class StateItem(BaseModel):
    id: str = Field(..., description="State code or identifier", examples=["TS"])
    name: str = Field(..., description="Full state name", examples=["Telangana"])
    code: Optional[str] = Field(None, description="ISO/Postal State Code", examples=["TS"])
    capital: Optional[str] = Field(None, description="State capital city", examples=["Hyderabad"])
    region: Optional[str] = Field(None, description="Geographic zone of India", examples=["South"])


class DistrictItem(BaseModel):
    id: str = Field(..., description="District code or identifier", examples=["TS-RR"])
    name: str = Field(..., description="District name", examples=["Rangareddy"])
    district_id: Optional[int] = Field(None, description="Internal numeric district ID")
    state_id: Optional[int] = Field(None, description="Associated state ID")
    state_name: Optional[str] = Field(None, description="Associated state name", examples=["Telangana"])
    tier: Optional[str] = Field(None, description="Economic / City tier", examples=["Tier-1"])
    headquarters: Optional[str] = Field(None, description="Administrative headquarters")
    base_price_sqft: Optional[float] = Field(None, description="Benchmark median price per sqft")
    min_price_sqft: Optional[float] = Field(None, description="Minimum price per sqft")
    max_price_sqft: Optional[float] = Field(None, description="Maximum price per sqft")
    median_price_inr: Optional[float] = Field(None, description="Typical 1200 sqft residential unit price")
    has_metro: Optional[bool] = Field(None, description="Presence of operational or under-construction metro rail")
    airport_dist_base: Optional[float] = Field(None, description="Average distance to airport in km")
    railway_dist_base: Optional[float] = Field(None, description="Average distance to railway station in km")


class SubdistrictItem(BaseModel):
    id: str = Field(..., description="Subdistrict code or identifier", examples=["SUB_1"])
    name: str = Field(..., description="Subdistrict / Mandal / Taluk / Tehsil name", examples=["Ibrahimpatnam"])
    subdistrict_id: Optional[int] = Field(None, description="Internal numeric subdistrict ID")
    district_id: Optional[int] = Field(None, description="Associated district ID")
    district_name: Optional[str] = Field(None, description="Associated district name", examples=["Rangareddy"])
    state_name: Optional[str] = Field(None, description="Associated state name", examples=["Telangana"])
    type: Optional[str] = Field("Mandal", description="Administrative classification (Mandal, Taluk, Tehsil)")
    base_price_sqft: Optional[float] = Field(None, description="Benchmark median price per sqft")
    min_price_sqft: Optional[float] = Field(None, description="Minimum price per sqft")
    max_price_sqft: Optional[float] = Field(None, description="Maximum price per sqft")
    median_price_inr: Optional[float] = Field(None, description="Typical 1200 sqft residential unit price")
    urban_rural: Optional[str] = Field(None, description="Urban, Semi-Urban, or Rural classification")
    has_metro: Optional[bool] = Field(None, description="Metro rail connectivity")
    airport_dist_base: Optional[float] = Field(None, description="Distance to airport in km")
    railway_dist_base: Optional[float] = Field(None, description="Distance to railway station in km")


class VillageItem(BaseModel):
    id: str = Field(..., description="Village or Locality identifier", examples=["LOC_1"])
    name: str = Field(..., description="Village or Locality name", examples=["Adibatla"])
    locality_id: Optional[int] = Field(None, description="Internal numeric locality ID")
    subdistrict_id: Optional[int] = Field(None, description="Associated subdistrict ID")
    district_id: Optional[int] = Field(None, description="Associated district ID")
    mandal_name: Optional[str] = Field(None, description="Associated subdistrict/mandal name")
    district_name: Optional[str] = Field(None, description="Associated district name")
    state_name: Optional[str] = Field(None, description="Associated state name")
    pincode: Optional[str] = Field(None, description="6-digit Indian Postal Code", examples=["501510"])
    latitude: Optional[float] = Field(None, description="Representative latitude", examples=[17.199])
    longitude: Optional[float] = Field(None, description="Representative longitude", examples=[78.558])
    urban_rural: Optional[str] = Field(None, description="Urban, Semi-Urban, or Rural classification")
    base_price_sqft: Optional[float] = Field(None, description="Reference benchmark price per sqft")
    min_price_sqft: Optional[float] = Field(None, description="Minimum price per sqft")
    max_price_sqft: Optional[float] = Field(None, description="Maximum price per sqft")
    median_price_inr: Optional[float] = Field(None, description="Typical 1200 sqft residential unit price")
    has_metro: Optional[bool] = Field(None, description="Metro rail connectivity")
    airport_dist_base: Optional[float] = Field(None, description="Distance to airport in km")
    railway_dist_base: Optional[float] = Field(None, description="Distance to railway station in km")
    hospital_dist_base: Optional[float] = Field(None, description="Distance to major hospital in km")
    school_dist_base: Optional[float] = Field(None, description="Distance to reputable school in km")
    shopping_dist_base: Optional[float] = Field(None, description="Distance to shopping center in km")


class StateListResponse(BaseModel):
    states: List[StateItem]
    total: int = Field(..., description="Total count of states")


class DistrictListResponse(BaseModel):
    state: str = Field(..., description="State name queried")
    districts: List[DistrictItem]
    total: int = Field(..., description="Total count of districts for the state")


class SubdistrictListResponse(BaseModel):
    district: str = Field(..., description="District name queried")
    subdistricts: List[SubdistrictItem]
    total: int = Field(..., description="Total count of subdistricts for the district")


class VillageListResponse(BaseModel):
    subdistrict: str = Field(..., description="Subdistrict name queried")
    villages: List[VillageItem]
    total: int = Field(..., description="Total count of villages/localities")


class LocationEntityRef(BaseModel):
    id: Optional[str] = None
    name: str


class ReverseLocationResponse(BaseModel):
    latitude: float = Field(..., description="Resolved latitude coordinate")
    longitude: float = Field(..., description="Resolved longitude coordinate")
    state: Optional[str] = Field(None, description="Resolved Indian state name")
    district: Optional[str] = Field(None, description="Resolved district name")
    subdistrict: Optional[str] = Field(None, description="Resolved subdistrict / mandal")
    mandal_taluk_tehsil: Optional[str] = Field(None, description="Alias for subdistrict")
    village: Optional[str] = Field(None, description="Resolved locality / village name")
    village_locality: Optional[str] = Field(None, description="Alias for village")
    pincode: Optional[str] = Field(None, description="Postal pincode")
    distance_km: Optional[float] = Field(None, description="Distance to nearest resolved reference point in km")
    
    # Detailed entity references with stable IDs (as recommended in server.txt Section 62)
    state_detail: Optional[LocationEntityRef] = None
    district_detail: Optional[LocationEntityRef] = None
    subdistrict_detail: Optional[LocationEntityRef] = None
    village_detail: Optional[LocationEntityRef] = None
