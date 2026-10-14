/**
 * Validates property prediction form inputs before submitting to FastAPI
 * @param {object} formData 
 * @returns {{ isValid: boolean, errors: Record<string, string> }}
 */
export const validatePropertyForm = (formData) => {
  const errors = {};

  // Location checks
  if (!formData.state || !formData.state.trim()) {
    errors.state = 'Please select or enter a State';
  }
  if (!formData.district || !formData.district.trim()) {
    errors.district = 'Please select or enter a District';
  }
  if (!formData.property_type || !formData.property_type.trim()) {
    errors.property_type = 'Property type is required';
  }

  // Dimension checks
  const builtUp = Number(formData.built_up_area_sqft);
  if (!builtUp || builtUp <= 0) {
    errors.built_up_area_sqft = 'Built-up area must be greater than 0 sq ft';
  } else if (builtUp > 50000) {
    errors.built_up_area_sqft = 'Built-up area exceeds maximum limit of 50,000 sq ft';
  }

  const carpet = Number(formData.carpet_area_sqft);
  if (formData.carpet_area_sqft !== '' && formData.carpet_area_sqft !== null && formData.carpet_area_sqft !== undefined) {
    if (carpet <= 0) {
      errors.carpet_area_sqft = 'Carpet area must be greater than 0';
    } else if (carpet > builtUp) {
      errors.carpet_area_sqft = 'Carpet area cannot exceed built-up area';
    }
  }

  // Room checks
  const bedrooms = Number(formData.bedrooms);
  if (bedrooms < 0 || !Number.isInteger(bedrooms)) {
    errors.bedrooms = 'Bedrooms must be a positive integer';
  }

  const bathrooms = Number(formData.bathrooms);
  if (bathrooms < 0 || !Number.isInteger(bathrooms)) {
    errors.bathrooms = 'Bathrooms must be a positive integer';
  }

  // Floor checks
  const floor = Number(formData.floor_number);
  const totalFloors = Number(formData.total_floors);
  if (floor < 0) {
    errors.floor_number = 'Floor number cannot be negative';
  }
  if (totalFloors < 1) {
    errors.total_floors = 'Total floors must be at least 1';
  }
  if (floor > totalFloors) {
    errors.floor_number = 'Floor number cannot exceed total floors';
  }

  // Age check
  const age = Number(formData.property_age_years);
  if (age < 0) {
    errors.property_age_years = 'Property age cannot be negative';
  }

  // Proximity check
  const distCity = Number(formData.distance_to_city_center_km);
  if (formData.distance_to_city_center_km === '' || isNaN(distCity) || distCity < 0) {
    errors.distance_to_city_center_km = 'Distance to city center must be >= 0 km';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
};
