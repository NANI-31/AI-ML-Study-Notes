from typing import Optional, Union
from pydantic import BaseModel, Field, model_validator


class HousePredictionRequest(BaseModel):
    """
    Validated input schema for Indian residential property price estimation.
    Excludes latitude and longitude as they are deferred to later spatial iterations.
    """

    # Primary Geographic Hierarchy (Non-coordinate)
    state: str = Field(..., description="Indian State or Union Territory", examples=["Telangana"])
    district: str = Field(..., description="Administrative district name", examples=["Ranga Reddy"])
    mandal_taluk_tehsil: Optional[str] = Field(None, description="Subdistrict / Mandal / Taluk", examples=["Serilingampally"])
    village_locality: Optional[str] = Field(None, description="Neighborhood or Locality name", examples=["Gachibowli"])
    pincode: Optional[Union[str, int]] = Field(None, description="6-digit Indian Postal PIN Code", examples=["500032"])

    # Geographic Coordinates (Optional: used for spatial precision and map integration)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Geographic latitude coordinate [-90, 90]", examples=[17.199])
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Geographic longitude coordinate [-180, 180]", examples=[78.558])

    # Property Characteristics
    property_type: str = Field(..., description="Property category (Apartment, Independent House, Villa, Builder Floor, Residential Plot)", examples=["Apartment"])
    listing_type: str = Field("Sale", description="Market listing category (Sale, Resale, New Booking)", examples=["Sale"])
    property_status: str = Field("Ready to Move", description="Possession status (Ready to Move, Under Construction)", examples=["Ready to Move"])
    transaction_type: str = Field("Freehold", description="Ownership legal type (Freehold, Leasehold, Power of Attorney)", examples=["Freehold"])

    # Physical Dimensions
    built_up_area_sqft: float = Field(..., gt=0, description="Total built-up area in square feet", examples=[1250.0])
    carpet_area_sqft: Optional[float] = Field(None, gt=0, description="Net usable internal carpet area in square feet", examples=[1020.0])
    land_area_sqft: Optional[float] = Field(None, ge=0, description="Plot or land area in square feet (for standalone houses/plots)", examples=[1500.0])

    # Room and Floor Configuration
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms (BHK count)", examples=[2])
    bathrooms: int = Field(..., ge=0, description="Number of bathrooms", examples=[2])
    balconies: Optional[float] = Field(1.0, ge=0, description="Number of attached balconies", examples=[1.0])
    floor_number: int = Field(0, ge=0, description="Floor level of property (0 = ground)", examples=[5])
    total_floors: int = Field(1, ge=1, description="Total number of floors in structure", examples=[14])
    property_age_years: int = Field(0, ge=0, description="Age of building in years (0 = new/under construction)", examples=[3])

    # Amenities and Quality
    furnishing_status: str = Field("Semi-Furnished", description="Furnishing grade (Unfurnished, Semi-Furnished, Fully Furnished)", examples=["Semi-Furnished"])
    parking_spaces: Optional[float] = Field(1.0, ge=0, description="Designated vehicle parking slots", examples=[1.0])
    facing_direction: str = Field("East", description="Main entrance orientation (Vastu direction)", examples=["East"])
    road_width_ft: Optional[float] = Field(30.0, gt=0, description="Access road width in front of property in feet", examples=[40.0])
    gated_community: Union[bool, int] = Field(1, description="Whether property is in a gated community (1/True or 0/False)", examples=[1])
    construction_quality: str = Field("Standard", description="Build specification (Economy, Standard, Premium, Luxury)", examples=["Premium"])
    property_condition: str = Field("Good", description="Maintenance state (New, Excellent, Good, Needs Renovation)", examples=["New"])

    # Infrastructure and Proximity Metrics (in km)
    distance_to_city_center_km: float = Field(..., ge=0, description="Distance to city municipal center in km", examples=[12.0])
    distance_to_main_road_km: float = Field(0.5, ge=0, description="Distance to nearest main road in km", examples=[0.4])
    distance_to_highway_km: float = Field(2.5, ge=0, description="Distance to nearest highway in km", examples=[2.5])
    distance_to_metro_km: float = Field(2.0, ge=0, description="Distance to nearest metro transit in km", examples=[1.5])
    distance_to_railway_station_km: float = Field(10.0, ge=0, description="Distance to nearest railway station in km", examples=[14.0])
    distance_to_airport_km: float = Field(25.0, ge=0, description="Distance to commercial airport in km", examples=[28.0])
    distance_to_school_km: float = Field(1.0, ge=0, description="Distance to nearest school in km", examples=[1.0])
    distance_to_hospital_km: float = Field(1.5, ge=0, description="Distance to nearest hospital in km", examples=[1.5])
    distance_to_shopping_center_km: float = Field(1.5, ge=0, description="Distance to nearest shopping center in km", examples=[1.2])
    distance_to_it_hub_km: float = Field(3.0, ge=0, description="Distance to nearest IT or business hub in km", examples=[1.0])

    nearby_public_transport: str = Field("High", description="Accessibility rating (High, Moderate, Low)", examples=["High"])
    neighborhood_type: str = Field("Residential", description="Locality character (Residential, Commercial-Residential Mixed, Gated Township, Suburban Layout)", examples=["Residential"])
    urban_rural: str = Field("Urban", description="Urbanization tier (Urban, Semi-Urban, Rural)", examples=["Urban"])

    @model_validator(mode="after")
    def validate_logical_integrity(self):
        # 1. Floor number cannot exceed total floors
        if self.floor_number > self.total_floors:
            raise ValueError(f"floor_number ({self.floor_number}) cannot exceed total_floors ({self.total_floors}).")

        # 2. Carpet area defaults to reasonable fraction if not provided
        if self.carpet_area_sqft is None:
            self.carpet_area_sqft = round(self.built_up_area_sqft * 0.80, 1)
        elif self.carpet_area_sqft > self.built_up_area_sqft:
            raise ValueError(f"carpet_area_sqft ({self.carpet_area_sqft}) cannot exceed built_up_area_sqft ({self.built_up_area_sqft}).")

        # 3. Ensure gated_community is stored as an integer (0 or 1)
        if isinstance(self.gated_community, bool):
            self.gated_community = 1 if self.gated_community else 0

        return self


class PriceValuation(BaseModel):
    """Calculated property valuations in diverse currency denominations."""
    price_inr: float = Field(..., description="Estimated market valuation in Indian Rupees (INR)", examples=[6250000.0])
    price_lakh: float = Field(..., description="Valuation in Lakhs (1 Lakh = 100,000 INR)", examples=[62.5])
    price_crore: float = Field(..., description="Valuation in Crores (1 Crore = 10,000,000 INR)", examples=[0.625])


class ModelInfo(BaseModel):
    """Metadata regarding the inference model."""
    name: str = Field(..., examples=["Indian House Price Prediction Model"])
    version: Optional[str] = Field("1.0.0", examples=["1.0.0"])


class HousePredictionResponse(BaseModel):
    """API response model for property valuation predictions."""
    success: bool = Field(True, examples=[True])
    prediction: PriceValuation
    model: ModelInfo
