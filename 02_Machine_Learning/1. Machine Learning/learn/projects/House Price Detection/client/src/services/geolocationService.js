import apiClient from './api';
import locationsData from '../data/locationsData.json';

/**
 * Calculates great-circle distance between two points in km (Haversine formula)
 */
function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

export const geolocationService = {
  /**
   * Performs boundary-accurate polygon reverse-geocoding via the backend server.
   * Enforces State -> District -> Mandal -> Locality boundary containment.
   * Falls back to offline Euclidean distance if network is unavailable.
   * @param {number} latitude 
   * @param {number} longitude 
   * @returns {Promise<object|null>}
   */
  async reverseGeocodeOnline(latitude, longitude) {
    if (!latitude || !longitude) return null;

    try {
      const response = await apiClient.get('/api/v1/locations/reverse-geocode', {
        params: { latitude, longitude },
      });

      const data = response.data;
      if (data && data.state) {
        const localityName = data.village || data.village_locality || data.subdistrict || 'Local Area';
        const localityObj = {
          name: localityName,
          state_name: data.state,
          district_name: data.district,
          mandal_name: data.subdistrict || data.mandal_taluk_tehsil,
          pincode: data.pincode || '522001',
          latitude: data.latitude,
          longitude: data.longitude,
          base_price_sqft: data.village_detail?.base_price_sqft || 3500,
          urban_rural: 'Urban',
          airport_dist_base: 25,
          railway_dist_base: 10,
        };

        return {
          locality: localityObj,
          distanceKm: data.distance_km || 0,
          coordinates: { latitude, longitude },
          details: data,
        };
      }
    } catch (err) {
      console.warn('Online boundary geocoding failed, falling back to local dataset:', err);
    }

    return this.reverseGeocode(latitude, longitude);
  },

  /**
   * Local Euclidean fallback: resolves nearest locality from local JSON dataset
   * @param {number} latitude 
   * @param {number} longitude 
   * @returns {object|null}
   */
  reverseGeocode(latitude, longitude) {
    if (!latitude || !longitude) return null;

    const localities = locationsData.localities || [];
    if (localities.length === 0) return null;

    let closestLocality = null;
    let minDistance = Infinity;

    for (const loc of localities) {
      const dist = haversineDistance(latitude, longitude, loc.latitude, loc.longitude);
      if (dist < minDistance) {
        minDistance = dist;
        closestLocality = loc;
      }
    }

    return {
      locality: closestLocality,
      distanceKm: Math.round(minDistance),
      coordinates: { latitude, longitude },
    };
  },
};

export default geolocationService;
