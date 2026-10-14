import axios from 'axios';

// Base URL resolves from Vite environment variables or falls back to local server
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// Centralized Response Interceptor for user-friendly error formatting
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    let message = 'Unable to connect to the prediction server. Please ensure the backend is running.';

    if (error.response) {
      const status = error.response.status;
      const detail = error.response.data?.detail;

      if (status === 422) {
        if (Array.isArray(detail)) {
          message = `Validation Error: ${detail.map((d) => d.msg || d.message).join(', ')}`;
        } else {
          message = detail || 'Invalid property specifications provided.';
        }
      } else if (status === 400) {
        message = detail || 'Bad request. Please review your property details.';
      } else if (status === 503) {
        message = 'The Machine Learning model is temporarily unavailable. Please check server status.';
      } else if (status >= 500) {
        message = detail || 'Server calculation error. Please try again in a few moments.';
      } else {
        message = detail || `Request failed with status ${status}`;
      }
    } else if (error.code === 'ECONNABORTED') {
      message = 'Request timed out while computing property valuation. Please retry.';
    }

    const customError = new Error(message);
    customError.status = error.response?.status;
    customError.originalError = error;
    return Promise.reject(customError);
  }
);

export default apiClient;
