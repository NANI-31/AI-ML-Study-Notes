# 🏢 Indian House Price Prediction — FastAPI Backend

Modular, production-grade FastAPI microservice delivering residential property valuations across Indian States and Districts using an optimized Scikit-Learn Gradient Boosting pipeline.

---

## 📋 Architecture Overview

The system operates as the predictive bridge between a React client and the pre-trained Machine Learning pipeline:

```
┌─────────────────────────┐
│      React Client       │
│  (Property Input Form)  │
└────────────┬────────────┘
             │ POST /api/v1/predict (JSON)
             ▼
┌─────────────────────────┐
│     FastAPI Server      │
│  ┌───────────────────┐  │
│  │ 1. Pydantic Model │  │
│  │    Validation     │  │
│  ├───────────────────┤  │
│  │ 2. Convert to     │  │
│  │    DataFrame      │  │
│  ├───────────────────┤  │
│  │ 3. ML Pipeline    │  │
│  │    Inference      │  │
│  └───────────────────┘  │
└────────────┬────────────┘
             │ JSON Valuation Response
             ▼
┌─────────────────────────┐
│      React Client       │
│ ₹ 62.5 Lakhs / 0.625 Cr │
└─────────────────────────┘
```

> [!IMPORTANT]
> **No Model Retraining**: The backend does **not** train or modify the model. The trained composite pipeline (`house_price_model.joblib`) contains its own preprocessing, scaling, encoding, and inverse log-transformation. FastAPI passes validated DataFrames directly to `pipeline.predict()`.

---

## 📁 Project Structure

```
ML/websites/server/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI entrypoint, lifespan & CORS
│   │
│   ├── api/
│   │   ├── __init__.py             # API v1 aggregator
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py           # GET /api/v1/health
│   │       └── prediction.py       # POST /api/v1/predict
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py           # Pydantic Request & Response models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py   # DataFrame conversion & pipeline invocation
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   └── model_loader.py         # In-memory singleton loader with startup check
│   │
│   └── core/
│       ├── __init__.py
│       └── config.py               # Pydantic Settings & path resolver
│
├── tests/
│   ├── __init__.py
│   └── test_prediction.py          # Pytest suite using TestClient
│
├── .env.example                    # Environment variable template
├── .env                            # Local configuration
├── requirements.txt                # Pinned dependencies
└── README.md                       # Service documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Navigate to the server directory and create a virtual environment:

```powershell
cd d:\Data\GitHub\interview\ML\websites\server
python -m venv venv
```

Activate the environment:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Review the artifact paths inside `.env`:
```env
APP_NAME="Indian House Price Prediction API"
APP_VERSION="1.0.0"
DEBUG=True
HOST="0.0.0.0"
PORT=8000
ALLOWED_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
MODEL_PATH="app/model/house_price_model.joblib"
MODEL_METADATA_PATH="app/model/model_metadata.json"
```

> [!NOTE]
> All paths are automatically resolved relative to the server root directory via `app/core/config.py`, ensuring reliable execution regardless of the terminal working directory.

### 3. Running the Server
Launch the development server with Uvicorn auto-reload:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Base URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Endpoints

### 1. Root Endpoint
- **Method**: `GET /`
- **Response**:
```json
{
  "message": "Indian House Price Prediction API",
  "version": "1.0.0"
}
```

---

### 2. Health & Model Status
- **Method**: `GET /api/v1/health`
- **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 3. Property Price Prediction
- **Method**: `POST /api/v1/predict`
- **Request Body (`HousePredictionRequest`)**:
```json
{
  "state": "Telangana",
  "district": "Ranga Reddy",
  "mandal_taluk_tehsil": "Serilingampally",
  "village_locality": "Gachibowli",
  "pincode": "500032",
  "property_type": "Apartment",
  "listing_type": "Sale",
  "property_status": "Ready to Move",
  "transaction_type": "Freehold",
  "built_up_area_sqft": 1250.0,
  "carpet_area_sqft": 1020.0,
  "bedrooms": 2,
  "bathrooms": 2,
  "balconies": 1.0,
  "floor_number": 5,
  "total_floors": 14,
  "property_age_years": 3,
  "furnishing_status": "Semi-Furnished",
  "parking_spaces": 1.0,
  "facing_direction": "East",
  "road_width_ft": 40.0,
  "gated_community": 1,
  "construction_quality": "Premium",
  "property_condition": "New",
  "distance_to_city_center_km": 12.0,
  "distance_to_main_road_km": 0.4,
  "distance_to_highway_km": 2.5,
  "distance_to_metro_km": 1.5,
  "distance_to_railway_station_km": 14.0,
  "distance_to_airport_km": 28.0,
  "distance_to_school_km": 1.0,
  "distance_to_hospital_km": 1.5,
  "distance_to_shopping_center_km": 1.2,
  "distance_to_it_hub_km": 1.0,
  "nearby_public_transport": "High",
  "neighborhood_type": "Residential",
  "urban_rural": "Urban"
}
```

- **Response Body (`HousePredictionResponse`)**:
```json
{
  "success": true,
  "prediction": {
    "price_inr": 18104819.62,
    "price_lakh": 181.048,
    "price_crore": 1.8105
  },
  "model": {
    "name": "Indian House Price Prediction Model",
    "version": "1.0.0"
  }
}
```

---

## ⚛️ React Integration Example

Frontend components in React can communicate with the FastAPI backend using standard `fetch` or `axios`:

```typescript
import axios from 'axios';

interface ValuationResponse {
  success: boolean;
  prediction: {
    price_inr: number;
    price_lakh: number;
    price_crore: number;
  };
  model: {
    name: string;
    version: string;
  };
}

export const estimateHousePrice = async (propertyData: Record<string, any>): Promise<ValuationResponse> => {
  try {
    const response = await axios.post<ValuationResponse>(
      'http://127.0.0.1:8000/api/v1/predict',
      propertyData,
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 5000,
      }
    );
    return response.data;
  } catch (error) {
    console.error('House price estimation failed:', error);
    throw error;
  }
};
```

---

## 🗺️ Coordinate & Spatial Roadmap

- **Current State**: Latitude and longitude are excluded from the client request schema. The backend service supplies `NaN` values, which are imputed by the saved Scikit-Learn pipeline.
- **Future Integration**: When interactive Mapbox/Leaflet pin-drop features are integrated, `latitude` and `longitude` fields will be activated in `HousePredictionRequest` without modifying the underlying prediction pipeline.

---

## 🧪 Running Unit Tests

Execute the automated test suite:
```bash
pytest tests/ -v
```
The suite verifies root endpoint availability, health status, schema validation guards, and end-to-end valuation execution.
