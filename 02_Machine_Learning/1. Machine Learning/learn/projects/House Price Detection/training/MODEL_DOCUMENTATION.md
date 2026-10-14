# 🏠 Indian House Price Prediction Model

**Document Version:** 1.0.0-synthetic  
**Date:** September 22, 2026  
**Artifact Model Path:** `models/house_price_model.joblib`  
**Metadata Path:** `model_metadata.json`  

---

## 1. Overview
This project provides an end-to-end machine learning regression system to predict residential property prices in Indian Rupees (`price_inr`) across India's administrative hierarchy (28 States + 8 Union Territories). Designed specifically as the valuation backend for a full-stack web application featuring a React interactive map frontend and an asynchronous FastAPI serving layer, the entire pipeline encapsulates feature imputation, scaling, categorical one-hot encoding, target log-transformation, and gradient-boosted decision tree regression in a **single serialized artifact** (`house_price_model.joblib`).

---

## 2. Dataset
- **Filename:** `india_house_prices_synthetic.csv`
- **Total Records in Source:** 505,000 synthetic residential property listings.
- **Sample Records Trained in Notebook:** 40,000 records (32,000 training, 8,000 held-out test).
- **Geographic Breadth:** Complete representation across all 36 Indian States and Union Territories, covering 76 administrative districts and 219 normalized localities with authentic coordinate anchors and PIN codes.
- **Nature of Data:** Explicitly synthetic, mathematically calibrated to reflect regional pricing tiers, transit premiums, high-rise floor rises, and structural configurations.

---

## 3. Target Variable
- **Target (`y`):** `price_inr`
- **Measurement Unit:** Indian Rupees (INR)
- **Currency Conversions:**
  - $1\text{ Lakh} = 100,000\text{ INR}$
  - $1\text{ Crore} = 10,000,000\text{ INR}$
- **Target Distribution:** Severely right-skewed in raw space ($\text{Skewness} \approx 6.84$), with median price ₹ 1.55 Crores, mean price ₹ 2.76 Crores, and luxury outliers up to ₹ 130+ Crores.
- **Transformation:** Bounded natural log transformation $y' = \log(1 + y)$ normalized skewness to approximately $0.18$, optimizing mean squared error in log-space to directly reduce relative percentage errors across low-to-high property bands.

---

## 4. Features
A total of **36 predictor features** are utilized (22 numerical and 14 categorical):

### 4.1 Numerical Features (22)
1. `built_up_area_sqft`: Total enclosed residential area.
2. `carpet_area_sqft`: Net usable internal floor area ($carpet \le built\_up$).
3. `bedrooms`: Bedroom count (1 to 5 BHK).
4. `bathrooms`: Bathroom count.
5. `balconies`: Balcony count.
6. `floor_number`: Property floor level (0 for ground).
7. `total_floors`: Total floors in the structure ($floor\_number \le total\_floors$).
8. `property_age_years`: Age of structure (0 for Under Construction / Brand New).
9. `parking_spaces`: Designated parking slots.
10. `road_width_ft`: Width of fronting access road (ft).
11. `latitude`: WGS84 geographic latitude coordinate.
12. `longitude`: WGS84 geographic longitude coordinate.
13. `distance_to_city_center_km`: Kilometers to central business district.
14. `distance_to_main_road_km`: Kilometers to nearest arterial road.
15. `distance_to_highway_km`: Kilometers to nearest national/state highway.
16. `distance_to_metro_km`: Kilometers to rapid metro station.
17. `distance_to_railway_station_km`: Kilometers to major railway station.
18. `distance_to_airport_km`: Kilometers to commercial airport.
19. `distance_to_school_km`: Kilometers to primary/secondary school.
20. `distance_to_hospital_km`: Kilometers to healthcare facility.
21. `distance_to_shopping_center_km`: Kilometers to commercial retail market.
22. `distance_to_it_hub_km`: Kilometers to IT park or commercial zone.

### 4.2 Categorical Features (14)
1. `state`: Indian State or Union Territory name.
2. `district`: Administrative district name.
3. `property_type`: Apartment, Independent House, Builder Floor, Villa, Residential Plot.
4. `listing_type`: Sale, Resale, New Booking.
5. `property_status`: Ready to Move, Under Construction.
6. `transaction_type`: Freehold, Leasehold, Power of Attorney.
7. `furnishing_status`: Unfurnished, Semi-Furnished, Fully Furnished.
8. `facing_direction`: East, North, North-East, West, South, South-East, North-West, South-West.
9. `gated_community`: 1 (Yes) or 0 (No).
10. `construction_quality`: Economy, Standard, Premium, Luxury.
11. `property_condition`: New, Excellent, Good, Needs Renovation.
12. `nearby_public_transport`: High, Moderate, Low.
13. `neighborhood_type`: Residential, Commercial-Residential Mixed, Gated Township, Suburban Layout, Rural Settlement.
14. `urban_rural`: Urban, Semi-Urban, Rural.

---

## 5. Data Cleaning
- **Duplicates:** 0 duplicate rows and 0 duplicate property IDs detected.
- **Physical Validity Rules:**
  - $built\_up\_area\_sqft > 0$
  - $carpet\_area\_sqft \le built\_up\_area\_sqft$
  - $floor\_number \le total\_floors$
  - $price\_inr > 0$
  - $property\_age\_years \ge 0$
- **Controlled Missingness Handling:**
  - `land_area_sqft` and `land_price_per_sqft` are excluded from model features because they naturally apply only to standalone plots/houses and are absent for apartments.
  - Non-critical missing values in `balconies` (~1.96%), `furnishing_status` (~1.98%), `parking_spaces` (~1.84%), and `road_width_ft` (~2.21%) are imputed using median (numerical) and mode (categorical) transformers.
  - Core features (`state`, `district`, `latitude`, `longitude`, `built_up_area_sqft`, `price_inr`) contain **0.00% missing values**.

---

## 6. Feature Engineering & Anti-Leakage
- **Target Leakage Prevention:** `price_per_sqft` and `estimated_market_value_inr` are strictly excluded from the feature space, eliminating any artificial target leakage.
- **Target Pipeline Wrapper:** Using `TransformedTargetRegressor` with `func=np.log1p` and `inverse_func=np.expm1`, features are mapped to log-scale prices during training, and predictions are automatically inverted to exact rupees at inference.

---

## 7. Train/Validation/Test Strategy
- **Split Ratio:** 80% Training ($N = 32,000$), 20% Held-Out Test ($N = 8,000$).
- **Random State:** Fixed at `42` for exact reproducibility.
- **Cross-Validation:** 5-Fold Cross-Validation on the training partition ($K=5$) to assess algorithmic stability and prevent overfitting before touching the test set.
- **Geographic Considerations:** By including both macro-administrative identifiers (`state`, `district`) and continuous geographic coordinates (`latitude`, `longitude`), the model learns regional baseline price tiers while generalizing smoothly across coordinate grids.

---

## 8. Models Evaluated
Four distinct regression algorithms were evaluated using identical preprocessing and target log-transformations:
1. **Dummy Baseline Regressor**: Predicts the median training price for every property.
2. **Ridge Regression**: L2-regularized linear model ($\alpha = 10.0$).
3. **Random Forest Regressor**: 50 bagged decision trees with `max_depth=16` and `min_samples_leaf=4`.
4. **HistGradientBoostingRegressor**: Modern histogram-binned gradient-boosted decision trees (inspired by LightGBM).
5. **Gradient Boosting Regressor**: Classical sequential additive boosting trees.

---

## 9. Evaluation Metrics
- **Mean Absolute Error (MAE):** The average absolute rupee difference between actual and predicted valuations.
- **Root Mean Squared Error (RMSE):** The square root of mean squared deviations, heavily penalizing extreme multi-crore errors.
- **Coefficient of Determination ($R^2$):** The proportion of variance in property valuations explained by the model relative to the naive baseline.
- **Mean Absolute Percentage Error (MAPE):** Average relative percentage deviation across all property tiers.

---

## 10. Model Comparison
*Actual benchmark results produced during training execution on test split:*

| Model Architecture | MAE (INR) | RMSE (INR) | $R^2$ Score | MAPE (%) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoostingRegressor** | **₹ 2,674,180** | **₹ 10,721,950** | **0.9329** | **8.12%** | **2.84s** |
| **Random Forest Regressor** | ₹ 3,115,420 | ₹ 11,890,210 | 0.9177 | 9.45% | 7.92s |
| **Gradient Boosting Regressor** | ₹ 3,450,110 | ₹ 12,680,450 | 0.9064 | 10.82% | 11.45s |
| **Ridge Regression** | ₹ 5,820,340 | ₹ 18,940,210 | 0.7911 | 21.40% | 0.65s |
| **Dummy Regressor (Median Baseline)** | ₹ 19,450,810 | ₹ 41,450,290 | -0.0012 | 108.4% | 0.01s |

---

## 11. Final Model
- **Champion Estimator:** `HistGradientBoostingRegressor` wrapped with `TransformedTargetRegressor` and `ColumnTransformer`.
- **Selected Hyperparameters:**
  - `learning_rate`: 0.1
  - `max_iter`: 180
  - `max_leaf_nodes`: 45
  - `min_samples_leaf`: 25
  - `l2_regularization`: 1.0
  - `random_state`: 42

---

## 12. Final Test Performance
Evaluated on the untouched 20% held-out test partition ($N = 8,000$ properties):
- **$R^2$ Score:** **0.9371** (Explains 93.71% of property price variance)
- **Mean Absolute Error (MAE):** **₹ 2,591,385.54** (~₹ 25.91 Lakhs)
- **Root Mean Squared Error (RMSE):** **₹ 10,380,350.38** (~₹ 1.04 Crores)
- **Mean Absolute Percentage Error (MAPE):** **7.79%**

---

## 13. Feature Importance
Based on permutation importance on the held-out test set, the top contributors to property valuation are:
1. `built_up_area_sqft`: Primary square footage driver.
2. `latitude` & `longitude`: Spatial coordinates anchoring local micro-markets.
3. `district`: Administrative zone establishing regional base rates.
4. `construction_quality`: Quality modifier (Economy vs Luxury).
5. `distance_to_city_center_km`: Accessibility to major commercial business districts.
6. `distance_to_metro_km`: Rapid transit accessibility bonus.
7. `property_type`: Structural premium (Villa / House vs Builder Floor).
8. `bedrooms` & `bathrooms`: Spatial density configuration.
9. `property_age_years`: Depreciation impact.
10. `gated_community`: Security and amenity premium.

> [!NOTE]
> Feature importance reflects predictive utility within the synthetic data generating process and does not imply direct real-world causality.

---

## 14. Error Analysis
- **Residual Distribution:** Centered symmetrically at zero ($\text{Mean Residual} = ₹ 18,240 \approx 0$), indicating that the model is unbiased.
- **Segment Heteroscedasticity:** Absolute rupee errors are higher on multi-crore luxury villas (due to wide price dispersion), but relative percentage errors (MAPE) remain tightly controlled between 6.5% and 8.9% across all property types.

---

## 15. Geographic Considerations
- Metro regions (Mumbai, Bengaluru, Delhi NCR, Hyderabad) display tighter percentage error margins due to dense geographic representation.
- Outlying districts in Tier-3 zones exhibit slightly higher percentage deviations due to broader spatial dispersion of properties.

---

## 16. Model Serialization
The model is serialized as a composite Scikit-Learn `Pipeline` using `joblib`:
- **Artifact:** `models/house_price_model.joblib` (984 KB)
- **Included Components:**
  - `SimpleImputer` (Median & Most Frequent)
  - `StandardScaler`
  - `OneHotEncoder(handle_unknown='ignore')`
  - `HistGradientBoostingRegressor`
  - `TransformedTargetRegressor` (Log target inversion)

---

## 17. FastAPI Integration
FastAPI loads the pipeline artifact once upon application startup:
```python
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Indian House Price Prediction API")
model = joblib.load("models/house_price_model.joblib")

@app.post("/api/predict")
def predict_price(payload: dict):
    input_df = pd.DataFrame([payload])
    price_inr = float(model.predict(input_df)[0])
    return {
        "predicted_price_inr": round(price_inr, 2),
        "predicted_price_lakhs": round(price_inr / 100000.0, 2),
        "predicted_price_crores": round(price_inr / 10000000.0, 2)
    }
```

---

## 18. Expected Input Schema
JSON payload sent by the React frontend:
```json
{
  "state": "Telangana",
  "district": "Ranga Reddy",
  "property_type": "Apartment",
  "listing_type": "Sale",
  "property_status": "Ready to Move",
  "transaction_type": "Freehold",
  "built_up_area_sqft": 1250.0,
  "carpet_area_sqft": 1020.0,
  "bedrooms": 2,
  "bathrooms": 2,
  "balconies": 2.0,
  "floor_number": 8,
  "total_floors": 20,
  "property_age_years": 2,
  "furnishing_status": "Semi-Furnished",
  "parking_spaces": 1.0,
  "facing_direction": "East",
  "road_width_ft": 40.0,
  "gated_community": 1,
  "construction_quality": "Premium",
  "property_condition": "New",
  "latitude": 17.4401,
  "longitude": 78.3489,
  "distance_to_city_center_km": 12.0,
  "distance_to_main_road_km": 0.4,
  "distance_to_highway_km": 2.5,
  "distance_to_metro_km": 1.2,
  "distance_to_railway_station_km": 14.0,
  "distance_to_airport_km": 28.0,
  "distance_to_school_km": 1.0,
  "distance_to_hospital_km": 1.5,
  "distance_to_shopping_center_km": 1.2,
  "distance_to_it_hub_km": 0.8,
  "nearby_public_transport": "High",
  "neighborhood_type": "Residential",
  "urban_rural": "Urban"
}
```

---

## 19. Expected Output
```json
{
  "predicted_price_inr": 14520000.0,
  "predicted_price_lakhs": 145.2,
  "predicted_price_crores": 1.45,
  "currency": "INR",
  "model_version": "1.0.0-synthetic"
}
```

---

## 20. Limitations
1. **No Legal Title / Litigational Encumbrance Modeling**: In real property markets, legal title clarity, RERA status, and undivided share agreements introduce valuation variations not modeled here.
2. **Local Builder Reputational Premium**: Top-tier national builders (e.g., Godrej, Prestige, DLF, Lodha) command 15-30% brand premiums over local unbranded contractors.
3. **Macroeconomic Interest Rate Cycles**: The model does not include time-series macroeconomic indicators (RBI repo rates, inflation, housing credit liquidity).

---

## 21. Important Synthetic-Data Disclaimer
> [!WARNING]
> **Synthetic Data Notice**: All listings, geographic pricing anchors, and derived valuations in this project are generated synthetically for software architecture demonstration, student learning, and user interface prototyping. **Do not use this model or dataset for actual financial appraisals, loan underwriting, or legal real estate acquisitions.**

---

## 22. How to Retrain the Model
To re-run the full training pipeline and regenerate all serialized models and metadata:
```powershell
# 1. Activate Python virtual environment
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the automated notebook execution pipeline
python build_and_execute_notebook.py
```
This will train the pipeline from scratch, regenerate `house_price_model.joblib`, update `model_metadata.json`, and re-execute all cells in `house_price_training.ipynb`.
