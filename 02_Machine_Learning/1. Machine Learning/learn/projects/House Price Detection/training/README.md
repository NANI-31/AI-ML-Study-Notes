# 🏠 House Price Prediction Using Regression

A two-stage machine-learning project covering both a **beginner-friendly synthetic pipeline** and a **real-world data pipeline** using the Ames Housing dataset (the same data behind the famous Kaggle House Prices competition).

---

## 📋 Project Description

This project demonstrates the complete machine-learning pipeline — from raw data generation through model training and evaluation — using Python and scikit-learn. Five regression algorithms are compared side-by-side with multiple evaluation metrics, and the best model is saved for future inference.

---

## 📁 Project Structure

```
House Price Detection/
│
├── 📘 NOTEBOOK 1 — Synthetic Data (Beginner)
│   └── house_price_prediction.ipynb   ← 14-section notebook (fully executed)
│
├── 📗 NOTEBOOK 2 — Real-World Data (Intermediate)
│   └── house_price_realworld.ipynb    ← Ames Housing + ColumnTransformer Pipeline
│
├── 💾 Generated Artefacts
│   ├── house_price_dataset.csv        ← Synthetic dataset (1,000 rows, 10 cols)
│   ├── house_price_model.pkl          ← Best model from Notebook 1
│   ├── house_price_scaler.pkl         ← StandardScaler from Notebook 1
│   └── ames_house_price_pipeline.pkl  ← Full Pipeline from Notebook 2
│
├── generate_dataset.py            ← Standalone dataset generation script
├── requirements.txt               ← pip dependencies
└── README.md                      ← This file
```

---

## 🆚 Notebook Comparison

| Feature | Notebook 1 (Synthetic) | Notebook 2 (Real-World) |
|---|---|---|
| Dataset | Generated with NumPy seed=42 | Ames Housing (OpenML / Kaggle) |
| Rows | 1,000 | ~1,455 (after cleaning) |
| Features | 9 numerical | 79 mixed (numerical + categorical) |
| Missing values | None | 19+ columns — imputed in Pipeline |
| Categorical cols | None | 43 columns — OHE + OrdinalEncoder |
| Outlier handling | None | IQR-based removal (factor=3.0) |
| Preprocessing | Manual StandardScaler | `ColumnTransformer` Pipeline |
| Saved artefact | model.pkl + scaler.pkl | Single all-in-one pipeline.pkl |
| Suitable for | Beginners | Intermediate learners |

---

## 📊 Dataset Descriptions

### Notebook 1 — Synthetic Dataset

Generated with NumPy (`seed=42`) for full reproducibility.

| Column | Type | Description | Range |
|---|---|---|---|
| `area_sqft` | Float | House area in square feet | 500 – 4,000 |
| `bedrooms` | Float | Number of bedrooms | 1 – 5 |
| `bathrooms` | Float | Number of bathrooms | 1 – 3 |
| `age_years` | Float | Age of the house (years) | 0 – 50 |
| `distance_to_city_km` | Float | Distance from city centre (km) | 1 – 50 |
| `garage` | Float | Garage present (0 = No, 1 = Yes) | 0 or 1 |
| `garden` | Float | Garden present (0 = No, 1 = Yes) | 0 or 1 |
| `floor` | Float | Floor number | 0 – 15 |
| `location_score` | Float | Location quality score | 1.0 – 10.0 |
| `house_price` | Float | **Target** — House price ($) | 50,000 – 1,000,000 |

### Notebook 2 — Ames Housing Dataset (Real-World — Indian Currency ₹)

Loaded from [OpenML](https://www.openml.org/d/42165) — same as the Kaggle House Prices competition. No Kaggle account needed.

- **1,460 residential properties** in Ames, Iowa (2006–2010)
- **79 features**: lot size, build quality, neighbourhood, year built, garage, basement, utilities, etc.
- **Target**: `SalePrice` — actual sale price in **Indian Rupees (INR / ₹)**, converted from USD at ₹83 / USD (₹28.97 Lakhs – ₹6.27 Crores)
- **19+ columns** have missing values
- **43 categorical columns** encoded inside the Pipeline
- **Evaluation & Predictions**: Reported in Indian Rupees (₹), with figures formatted in Lakhs and Crores


**Price formula used:**
```
price = 120 × area_sqft + 8000 × bedrooms + 6000 × bathrooms
      - 1200 × age_years - 3500 × distance_to_city_km
      + 20000 × garage + 15000 × garden + 2000 × floor
      + 12000 × location_score + 50000 + noise(0, 15000)
```

---

## 🤖 Algorithms Used

| Algorithm | Type | Scaling Required |
|---|---|---|
| **Linear Regression** | Linear | Yes (StandardScaler) |
| **Ridge Regression** | Linear + L2 regularisation | Yes |
| **Lasso Regression** | Linear + L1 regularisation | Yes |
| **Random Forest Regressor** | Ensemble (Bagging) | No |
| **Gradient Boosting Regressor** | Ensemble (Boosting) | No |

---

## 📈 Evaluation Metrics

| Metric | Symbol | Ideal Value | Description |
|---|---|---|---|
| Mean Absolute Error | MAE | Lower is better | Average absolute prediction error |
| Mean Squared Error | MSE | Lower is better | Average squared error (penalises outliers) |
| Root Mean Squared Error | RMSE | Lower is better | Square root of MSE — in the same units as price |
| R-squared | R² | Closer to 1.0 is better | Proportion of price variance explained |

---

## 🔧 Installation Requirements

### Prerequisites

- Python 3.8 or higher
- pip

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib notebook
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
joblib>=1.2
notebook>=7.0
```

---

## ▶️ How to Run the Notebook

### Option 1 — Jupyter Notebook (recommended)

```bash
# Navigate to the project folder
cd "d:/Data/GitHub/interview/ML/projects/House Price Detection"

# Start Jupyter
jupyter notebook

# Open house_price_prediction.ipynb and click Kernel > Restart & Run All
```

### Option 2 — JupyterLab

```bash
jupyter lab
```

### Option 3 — VS Code

Open the `.ipynb` file in VS Code with the Python/Jupyter extension installed and click **Run All**.

### Option 4 — Command line execution

```bash
python -m nbconvert --to notebook --execute --inplace house_price_prediction.ipynb
```

> **Note:** The notebook automatically generates `house_price_dataset.csv`, `house_price_model.pkl`, and `house_price_scaler.pkl` when executed from top to bottom.

---

## 💡 Example Prediction

```python
import joblib
import pandas as pd

# Load saved model and scaler
model  = joblib.load('house_price_model.pkl')
scaler = joblib.load('house_price_scaler.pkl')

# Define a new house
new_house = pd.DataFrame([{
    'area_sqft'           : 2500,
    'bedrooms'            : 4,
    'bathrooms'           : 2,
    'age_years'           : 10,
    'distance_to_city_km' : 15.0,
    'garage'              : 1,
    'garden'              : 1,
    'floor'               : 3,
    'location_score'      : 7.5,
}])

# For tree-based models — no scaling needed
predicted_price = model.predict(new_house)[0]
print(f'Predicted House Price: ${predicted_price:,.2f}')
```

---

## 📓 Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Project Title | Overview and objectives |
| 2 | Import Libraries | All required imports |
| 3 | Generate Dataset | Synthetic data creation & CSV save |
| 4 | Load Dataset | Read CSV, inspect shape and dtypes |
| 5 | Data Exploration (EDA) | Statistics + 6 visualisations |
| 6 | Data Preprocessing | Train/test split + StandardScaler |
| 7 | Train Models | 5 regression models trained & evaluated |
| 8 | Model Comparison | Table + 4 comparison bar charts |
| 9 | Actual vs Predicted | Scatter plot with reference line |
| 10 | Feature Importance | Tree importances + LR coefficients |
| 11 | Sample Prediction | Predict price for a new house |
| 12 | Save Model | joblib dump for model & scaler |
| 13 | Load & Test | Reload from disk and predict |
| 14 | Conclusion | Summary, limitations, next steps |

---

## ⚠️ Project Limitations

- The dataset is **synthetic** — the model cannot be deployed for real-world property valuations without training on genuine data.
- The price formula is **linear by design**, which may overfit linear models compared to a realistic non-linear dataset.
- No **temporal dynamics** are captured (housing market trends, inflation, economic cycles).
- The dataset has **no missing values or outliers** — real datasets require additional cleaning steps.

---

## 🇮🇳 Unified Indian House Price Synthetic Dataset (505,000+ Records)

Located in `dataset/`, this is a production-grade synthetic dataset modeled for full-stack ML applications (FastAPI backend, React frontend with interactive Mapbox/Leaflet GIS navigation):

```
dataset/
├── india_house_prices_synthetic.csv  ← 505,000 property records (47 columns, 167 MB)
├── data_dictionary.csv               ← Full data dictionary (types, descriptions, nullability)
├── india_states.csv                  ← 36 Indian States & Union Territories
├── india_districts.csv               ← 76 Districts with Tier classifications
├── india_subdistricts.csv            ← 87 Mandals, Taluks, Tehsils & Blocks
├── india_localities.csv              ← 219 Localities with coordinates & base valuations
├── validation_report.md              ← Comprehensive automated data quality audit
├── generate_reference_data.py        ← Script to generate geographic hierarchy tables
├── generate_dataset.py               ← Vectorized batch generator script
└── validate_dataset.py               ← Automated validation & regression audit engine
```

### Hierarchy
$$\text{India} \longrightarrow \text{State / UT (36)} \longrightarrow \text{District (76)} \longrightarrow \text{Mandal/Taluk (87)} \longrightarrow \text{Locality (219)} \longrightarrow \text{Property (505k)}$$

---

## 🚀 Possible Improvements

- Collect real housing data (e.g., from Kaggle or government open datasets).
- Apply hyperparameter tuning with `GridSearchCV` or `Optuna`.
- Add feature engineering (e.g., `price_per_sqft`, polynomial features).
- Build a simple web application with Flask or Streamlit to serve the model.
- Add cross-validation for more reliable model evaluation.

---

## 📚 Learning Objectives

By completing this project, a beginner will learn:

- ✅ How to generate and explore a dataset
- ✅ How to perform Exploratory Data Analysis (EDA)
- ✅ What train-test split and data leakage mean
- ✅ How to train and compare multiple regression models
- ✅ How to interpret MAE, RMSE, MSE, and R²
- ✅ How to visualise model performance
- ✅ How to save and load a trained model

---

*Project created for educational purposes. Random seed 42 is used throughout for full reproducibility.*

