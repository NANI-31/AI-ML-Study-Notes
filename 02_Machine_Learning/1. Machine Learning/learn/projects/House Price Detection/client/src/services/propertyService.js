/**
 * Static metadata and categorical options matching the trained ML model pipeline
 */
export const propertyService = {
  propertyTypes: [
    'Apartment',
    'Independent House',
    'Villa',
    'Builder Floor',
    'Residential Plot',
  ],

  listingTypes: ['Sale', 'Resale', 'New Booking'],

  propertyStatuses: ['Ready to Move', 'Under Construction'],

  transactionTypes: ['Freehold', 'Leasehold', 'Power of Attorney'],

  furnishingStatuses: ['Unfurnished', 'Semi-Furnished', 'Fully Furnished'],

  facingDirections: [
    'East',
    'North',
    'North-East',
    'North-West',
    'West',
    'South',
    'South-East',
    'South-West',
  ],

  constructionQualities: ['Economy', 'Standard', 'Premium', 'Luxury'],

  propertyConditions: ['New', 'Excellent', 'Good', 'Needs Renovation'],

  nearbyTransportRatings: ['High', 'Moderate', 'Low'],

  neighborhoodTypes: [
    'Residential',
    'Commercial-Residential Mixed',
    'Gated Township',
    'Suburban Layout',
  ],

  urbanRuralTiers: ['Urban', 'Semi-Urban', 'Rural'],

  defaultPropertyForm: {
    state: 'Telangana',
    district: 'Ranga Reddy',
    mandal_taluk_tehsil: 'Serilingampally',
    village_locality: 'Gachibowli',
    pincode: '500032',
    property_type: 'Apartment',
    listing_type: 'Sale',
    property_status: 'Ready to Move',
    transaction_type: 'Freehold',
    built_up_area_sqft: 1250,
    carpet_area_sqft: 1000,
    land_area_sqft: 0,
    bedrooms: 2,
    bathrooms: 2,
    balconies: 1,
    floor_number: 4,
    total_floors: 12,
    property_age_years: 2,
    furnishing_status: 'Semi-Furnished',
    parking_spaces: 1,
    facing_direction: 'East',
    road_width_ft: 40,
    gated_community: 1,
    construction_quality: 'Premium',
    property_condition: 'New',
    distance_to_city_center_km: 12,
    distance_to_main_road_km: 0.5,
    distance_to_highway_km: 2.0,
    distance_to_metro_km: 1.5,
    distance_to_railway_station_km: 14.0,
    distance_to_airport_km: 28.0,
    distance_to_school_km: 1.0,
    distance_to_hospital_km: 1.5,
    distance_to_shopping_center_km: 1.2,
    distance_to_it_hub_km: 1.0,
    nearby_public_transport: 'High',
    neighborhood_type: 'Residential',
    urban_rural: 'Urban',
  },
};

export default propertyService;
