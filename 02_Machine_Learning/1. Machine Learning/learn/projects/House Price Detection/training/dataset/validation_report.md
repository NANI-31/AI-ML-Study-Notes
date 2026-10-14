# Validation Report: Unified Indian House Price Synthetic Dataset

**Execution Date:** 2026-09-22 11:04:37  
**Target Dataset:** `india_house_prices_synthetic.csv`  
**Dataset Nature:** Explicitly Synthetic (Designed for ML Regression and Interactive Map Navigation)

---

## 1. Summary Statistics & Volume

| Metric | Value |
| :--- | :--- |
| **Total Property Records** | **505,000** |
| **Total States & UTs Represented** | **36 (All 28 States + 8 UTs)** |
| **Total Districts Represented** | **76** |
| **Unique Localities / Neighborhoods** | **219** |
| **Minimum Property Price** | ₹1,200,000 |
| **Maximum Property Price** | ₹1,307,440,000 |
| **Median Property Price** | **₹15,500,000** |
| **Mean Property Price** | ₹27,646,575 |
| **Minimum Price / Sq.Ft.** | ₹1,684.62 |
| **Maximum Price / Sq.Ft.** | ₹307,966.67 |
| **Median Price / Sq.Ft.** | **₹11,566.04** |
| **Mean Price / Sq.Ft.** | ₹17,075.18 |

---

## 2. Data Quality & Constraint Checks

| Validation Rule | Constraint Checked | Anomaly Count | Status |
| :--- | :--- | :---: | :---: |
| **Primary Key Uniqueness** | `property_id` unique across all rows | **0** | PASSED |
| **Row Uniqueness** | Zero exact duplicate rows | **0** | PASSED |
| **Coordinate Bounds** | Lat in [6.0, 37.5], Lon in [68.0, 98.0] | **0** | PASSED |
| **Hierarchy Consistency** | Locality uniquely maps to single District & State | **0** | PASSED |
| **Built-up Area Integrity** | `built_up_area_sqft > 0` | **0** | PASSED |
| **Carpet Area Constraint** | `carpet_area_sqft <= built_up_area_sqft` | **0** | PASSED |
| **Floor Level Logic** | `floor_number <= total_floors` | **0** | PASSED |
| **Positive Valuation** | `price_inr > 0` | **0** | PASSED |
| **Positive Unit Rate** | `price_per_sqft > 0` | **0** | PASSED |
| **Mathematical Derivation** | `abs(price_per_sqft - (price_inr / area)) <= 0.05` | **0** | PASSED |
| **Non-negative Age** | `property_age_years >= 0` | **0** | PASSED |
| **Valid Bedroom Count** | `bedrooms >= 0` | **0** | PASSED |
| **Valid Bathroom Count** | `bathrooms >= 0` | **0** | PASSED |

---

## 3. Property Type Distribution

| Property Type | Record Count | Percentage |
| :--- | :---: | :---: |
| **Apartment** | 277,376 | 54.93% |
| **Independent House** | 92,240 | 18.27% |
| **Builder Floor** | 71,220 | 14.10% |
| **Villa** | 34,847 | 6.90% |
| **Residential Plot** | 29,317 | 5.81% |

---

## 4. Controlled Missing Values (Non-Critical Fields)

| Field Name | Missing Records | Missing Percentage | Policy Compliance |
| :--- | :---: | :---: | :--- |
| `land_area_sqft` | 348,596 | 69.03% | Expected (Natural for Apartments) |
| `land_price_per_sqft` | 277,376 | 54.93% | Expected (Natural for Apartments) |
| `road_width_ft` | 11,180 | 2.21% | Controlled Synthetic Missingness (~2%) |
| `furnishing_status` | 10,010 | 1.98% | Controlled Synthetic Missingness (~2%) |
| `balconies` | 9,913 | 1.96% | Controlled Synthetic Missingness (~2%) |
| `parking_spaces` | 9,308 | 1.84% | Controlled Synthetic Missingness (~2%) |

> [!NOTE]
> All primary geographic and model-critical variables (`state`, `district`, `latitude`, `longitude`, `price_inr`, `built_up_area_sqft`) have **0.00% missing values**, strictly adhering to requirement 11.

---

## 5. Coverage Across Indian States & Union Territories

| State / Union Territory | State Code | Total Listings | Percentage |
| :--- | :---: | :---: | :---: |
| Maharashtra | `MH` | 82,161 | 16.27% |
| Telangana | `TG` | 72,023 | 14.26% |
| Karnataka | `KA` | 51,580 | 10.21% |
| Delhi | `DL` | 35,815 | 7.09% |
| Tamil Nadu | `TN` | 35,701 | 7.07% |
| Uttar Pradesh | `UP` | 28,995 | 5.74% |
| Gujarat | `GJ` | 27,632 | 5.47% |
| Haryana | `HR` | 23,393 | 4.63% |
| West Bengal | `WB` | 22,563 | 4.47% |
| Andhra Pradesh | `AP` | 13,840 | 2.74% |
| Rajasthan | `RJ` | 10,441 | 2.07% |
| Kerala | `KL` | 9,862 | 1.95% |
| Punjab | `PB` | 8,492 | 1.68% |
| Madhya Pradesh | `MP` | 8,489 | 1.68% |
| Jharkhand | `JH` | 7,065 | 1.40% |
| Assam | `AS` | 6,737 | 1.33% |
| Goa | `GA` | 6,591 | 1.31% |
| Bihar | `BR` | 6,534 | 1.29% |
| Odisha | `OD` | 6,131 | 1.21% |
| Jammu and Kashmir | `JK` | 5,794 | 1.15% |
| Chhattisgarh | `CG` | 5,733 | 1.14% |
| Himachal Pradesh | `HP` | 5,117 | 1.01% |
| Uttarakhand | `UK` | 4,824 | 0.96% |
| Chandigarh | `CH` | 4,317 | 0.85% |
| Puducherry | `PY` | 4,259 | 0.84% |
| Arunachal Pradesh | `AR` | 1,435 | 0.28% |
| Nagaland | `NL` | 1,405 | 0.28% |
| Andaman and Nicobar Islands | `AN` | 977 | 0.19% |
| Ladakh | `LA` | 974 | 0.19% |
| Manipur | `MN` | 966 | 0.19% |
| Tripura | `TR` | 957 | 0.19% |
| Meghalaya | `ML` | 957 | 0.19% |
| Dadra and Nagar Haveli and Daman and Diu | `DN` | 939 | 0.19% |
| Mizoram | `MZ` | 915 | 0.18% |
| Sikkim | `SK` | 904 | 0.18% |
| Lakshadweep | `LD` | 482 | 0.10% |

---
*Generated automatically by the Automated Validation Engine.*
