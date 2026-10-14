/**
 * Formats standard numbers with Indian comma grouping
 * @param {number|string} value 
 * @returns {string}
 */
export const formatIndianNumber = (value) => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return '0';
  }
  return Number(value).toLocaleString('en-IN');
};

/**
 * Formats area in square feet
 * @param {number|string} sqft 
 * @returns {string}
 */
export const formatSqft = (sqft) => {
  if (sqft === null || sqft === undefined || isNaN(Number(sqft))) {
    return '0 sq ft';
  }
  return `${Number(sqft).toLocaleString('en-IN')} sq ft`;
};

/**
 * Formats distance in km
 * @param {number|string} km 
 * @returns {string}
 */
export const formatDistanceKm = (km) => {
  if (km === null || km === undefined || isNaN(Number(km))) {
    return '0 km';
  }
  return `${Number(km).toFixed(1)} km`;
};
