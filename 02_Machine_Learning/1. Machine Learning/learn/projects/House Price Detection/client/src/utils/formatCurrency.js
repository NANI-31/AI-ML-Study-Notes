/**
 * Formats an amount in Indian Rupees with full comma notation (e.g., ₹ 62,50,000)
 * @param {number|string} amount 
 * @returns {string}
 */
export const formatINR = (amount) => {
  if (amount === null || amount === undefined || isNaN(Number(amount))) {
    return '₹ 0';
  }
  const numeric = Math.round(Number(amount));
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(numeric);
};

/**
 * Formats amount into Lakhs or Crores representation
 * Examples:
 *  - 45,00,000 -> "₹ 45.00 L"
 *  - 1,25,00,000 -> "₹ 1.25 Cr"
 * @param {number|string} amount 
 * @returns {string}
 */
export const formatIndianLakhCrore = (amount) => {
  if (amount === null || amount === undefined || isNaN(Number(amount))) {
    return '₹ 0';
  }
  const num = Number(amount);
  if (num >= 10000000) {
    const crore = (num / 10000000).toFixed(2);
    return `₹ ${crore} Cr`;
  }
  if (num >= 100000) {
    const lakh = (num / 100000).toFixed(2);
    return `₹ ${lakh} L`;
  }
  return formatINR(num);
};

/**
 * Formats price per square foot
 * @param {number|string} price 
 * @param {number|string} sqft 
 * @returns {string}
 */
export const formatPricePerSqft = (price, sqft) => {
  if (!price || !sqft || Number(sqft) <= 0) return 'N/A';
  const rate = Math.round(Number(price) / Number(sqft));
  return `₹ ${rate.toLocaleString('en-IN')} / sq ft`;
};
