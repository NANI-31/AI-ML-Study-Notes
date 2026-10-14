import apiClient from './api';

export const predictionService = {
  /**
   * Submits validated property parameters to the FastAPI model inference engine
   * @param {object} propertyPayload 
   * @returns {Promise<object>}
   */
  async predictHousePrice(propertyPayload) {
    // Sanitize numeric fields before dispatch
    const payload = {
      ...propertyPayload,
      built_up_area_sqft: Number(propertyPayload.built_up_area_sqft),
      carpet_area_sqft: propertyPayload.carpet_area_sqft ? Number(propertyPayload.carpet_area_sqft) : null,
      land_area_sqft: propertyPayload.land_area_sqft ? Number(propertyPayload.land_area_sqft) : 0,
      bedrooms: parseInt(propertyPayload.bedrooms, 10),
      bathrooms: parseInt(propertyPayload.bathrooms, 10),
      balconies: Number(propertyPayload.balconies || 0),
      floor_number: parseInt(propertyPayload.floor_number || 0, 10),
      total_floors: parseInt(propertyPayload.total_floors || 1, 10),
      property_age_years: parseInt(propertyPayload.property_age_years || 0, 10),
      parking_spaces: Number(propertyPayload.parking_spaces || 0),
      road_width_ft: Number(propertyPayload.road_width_ft || 30),
      gated_community: propertyPayload.gated_community ? 1 : 0,
      distance_to_city_center_km: Number(propertyPayload.distance_to_city_center_km || 10),
      distance_to_main_road_km: Number(propertyPayload.distance_to_main_road_km || 0.5),
      distance_to_highway_km: Number(propertyPayload.distance_to_highway_km || 2.0),
      distance_to_metro_km: Number(propertyPayload.distance_to_metro_km || 2.0),
      distance_to_railway_station_km: Number(propertyPayload.distance_to_railway_station_km || 10.0),
      distance_to_airport_km: Number(propertyPayload.distance_to_airport_km || 25.0),
      distance_to_school_km: Number(propertyPayload.distance_to_school_km || 1.0),
      distance_to_hospital_km: Number(propertyPayload.distance_to_hospital_km || 1.5),
      distance_to_shopping_center_km: Number(propertyPayload.distance_to_shopping_center_km || 1.5),
      distance_to_it_hub_km: Number(propertyPayload.distance_to_it_hub_km || 3.0),
    };

    const response = await apiClient.post('/api/v1/predict', payload);
    return response.data;
  },

  /**
   * Health check endpoint to verify model availability
   */
  async checkHealth() {
    const response = await apiClient.get('/api/v1/health');
    return response.data;
  },
};

export default predictionService;
