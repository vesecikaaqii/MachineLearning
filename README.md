<table border="0">
 <tr>
    <td><img src="https://github.com/user-attachments/assets/9002855f-3f97-4b41-a180-85d1e24ad34a" alt="University Logo" width="110" align="left"/></td>
    <td>
      <p><strong>University of Prishtina</strong></p>
      <p>Faculty of Electrical and Computer Engineering</p>
      <p>Computer and Software Engineering — Master's Program</p>
      <p>Professor: Prof. Lule Ahmedi</p>
      <p>Assistant: Prof. Mergim Hoti</p>
      <p>Course: Machine Learning</p>
    </td>
 </tr>
</table>

---

## Contributors
Vesë Cikaqi, Uranik Hodaj, Dafina Keqmezi
Academic Year: 2025 / 2026

---

# Training Weather Forecasting Models in Kosovo

A Machine Learning project that builds a complete, reproducible pipeline — from real-world data collection to model training, re-training, and evaluation — for forecasting air temperature across 27 cities of Kosovo.

---

## 📚 Table of Contents

1. [Project Goals](#-1-project-goals)
2. [Technologies Used](#-2-technologies-used)
3. [Installation and Setup](#-3-installation-and-setup)
4. [Dataset Description](#-4-dataset-description)
5. [PHASE I — Data Preparation](#-phase-i--data-preparation)
6. [Dataset Overview and Exploratory Insights](#-dataset-overview-and-exploratory-insights)
7. [Selected Algorithm](#-selected-algorithm)
8. [PHASE II — Analysis and Evaluation (Re-training)](#-phase-ii--analysis-and-evaluation-re-training)

---

## 🎯 1. Project Goals

The project is designed to fulfil both the educational and applied objectives of a Master-level Machine Learning course:

1. **Build a complete supervised ML pipeline** — from raw data collection to trained, evaluated models — for the task of **air-temperature forecasting** in Kosovo.
2. **Collect a real, domain-specific dataset** from a public meteorological API, covering all 27 municipalities of Kosovo.
3. **Apply and justify one ML algorithm** (as required by the course brief) — demonstrating supervised learning for a regression task.
4. **Demonstrate the re-training methodology** by explicitly iterating hyperparameters and comparing baseline vs. re-trained configurations.
5. **Produce publishable, reproducible artifacts** — serialised models, evaluation metrics, and visual reports — suitable for academic review.
6. **Create the foundation for Phase III** — where the same algorithm will be further optimised with automated hyperparameter search, feature engineering, and cross-validated tuning.

---

## 🛠️ 2. Technologies Used

| Category | Tool / Library | Purpose |
|----------|----------------|---------|
| **Language** | Python 3.14 | Core programming language |
| **Data Handling** | `pandas`, `numpy` | Tabular manipulation, numeric operations |
| **Visualisation** | `matplotlib`, `seaborn` | Plots, heat-maps, feature-importance charts |
| **Machine Learning** | `scikit-learn` | Random Forest, StandardScaler, train/test split, cross-validation, metrics |
| **Serialisation** | `joblib` | Saving trained models + scalers |
| **Data Source** | OpenWeatherMap API (via `requests`) | Live meteorological data (forecast + current) |
| **Version Control** | Git + GitHub | Source control, collaboration |
| **OS / Platform** | Windows 11, bash shell | Development environment |

---

## ⚙️ 3. Installation and Setup

### Prerequisites
- Python ≥ 3.10
- Git
- A free [OpenWeatherMap](https://openweathermap.org/api) API key (only required if you want to re-collect the dataset)

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/<user>/MachineLearning.git
cd MachineLearning

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate         # Linux / macOS / Git-Bash on Windows
# or
.venv\Scripts\activate            # PowerShell

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib requests

# 4. (Optional) Re-collect data from OpenWeatherMap
#    Replace the API_KEY inside weather_data_scraper.py with your own,
#    then:
python weather_data_scraper.py
#    This (re)creates / appends to kosovo_weather_dataset.csv

# 5. Run the Phase II training + evaluation pipeline
python phase2_model_training.py
```

### Expected artifacts after running Phase II

```
models/
├── rf_baseline.pkl          # Baseline Random Forest (100 trees)
├── rf_retrained.pkl         # Re-trained Random Forest (300 trees, tuned)
└── scaler_phase2.pkl        # Fitted StandardScaler (pipeline-compatible)

reports/
├── phase2_metrics.txt              # Full training log
├── phase2_summary.json             # Machine-readable metric summary
├── phase2_correlation_heatmap.png  # Feature correlation heat-map
├── phase2_feature_importance.png   # RF feature-importance chart
└── phase2_pred_vs_true.png         # Predicted vs. actual scatter
```

---

## 🗂️ 4. Dataset Description

| Property | Value |
|----------|-------|
| **Source** | [OpenWeatherMap API](https://openweathermap.org/api) — public, free tier |
| **Geographic scope** | 27 municipalities of the Republic of Kosovo |
| **Instances (rows)** | **1107** |
| **Attributes (columns)** | **20** |
| **File size** | ≈ 142 KB (CSV) |
| **Temporal resolution** | every 3 hours |
| **Temporal coverage** | ~5-day forecast horizon + one live snapshot per city |
| **File format** | CSV (UTF-8, `pandas`-compatible) |
| **Collection script** | [`weather_data_scraper.py`](weather_data_scraper.py) |
| **Raw data file** | [`kosovo_weather_dataset.csv`](kosovo_weather_dataset.csv) |

### Attributes (20)

`city`, `type`, `datetime`, `temperature`, `feels_like`, `temp_min`, `temp_max`, `pressure`, `humidity`, `weather`, `description`, `wind_speed`, `wind_deg`, `wind_gust`, `clouds`, `visibility`, `pop`, `hour`, `day`, `month`.

---

# 🧩 PHASE I — Data Preparation

## 🎯 Objective of the Phase
Phase I lays the foundation of the whole project: **collecting, structuring, and performing the initial cleaning of a real meteorological dataset for Kosovo.** No ML model can produce reliable results without high-quality, well-structured data, and this phase is therefore considered the cornerstone of the project.

## 🛠️ Tasks Performed

1. **Identification of the data source** — [OpenWeatherMap API](https://openweathermap.org/api) was chosen as a trusted, publicly accessible source for global meteorological forecasts.
2. **Selection of 27 municipalities of Kosovo** with their (lat, lon) coordinates to cover all regions.
3. **Development of the script [`weather_data_scraper.py`](weather_data_scraper.py)** which, for each city:
   - fetches the 5-day forecast at a 3-hour interval (`forecast` endpoint),
   - fetches the current weather snapshot (`weather` endpoint).
4. **Persistence to CSV** as [`kosovo_weather_dataset.csv`](kosovo_weather_dataset.csv), automatically appending the columns `hour`, `day`, and `month` for temporal analysis.
5. **Integrity verification** (no duplicates, no empty rows in the primary target columns).

## 🎯 Defined Machine-Learning Tasks

The dataset built in this phase is designed to support three main ML tasks across the later phases:

| # | Task | Type | Target (output) | Main input features |
|---|------|------|-----------------|---------------------|
| 1 | Temperature forecasting | **Regression (supervised)** | `temperature` (°C, numeric) | humidity, pressure, clouds, wind_speed |
| 2 | Weather-state classification | Classification (supervised) | `weather` (Clear / Clouds / Rain / Snow) | temperature, humidity, clouds, pop |
| 3 | Sequential time-series forecasting | Time-series (future work) | `temperature[t+1]` | 6-step × 5-feature sequence |

## 🗂️ Attribute Types

Out of 20 total columns, the structural split is:

| Type | Count | Attributes |
|------|-------|-----------|
| **Numeric (continuous)** | 10 | `temperature`, `feels_like`, `temp_min`, `temp_max`, `pressure`, `humidity`, `wind_speed`, `wind_gust`, `visibility`, `pop` |
| **Numeric (discrete / temporal)** | 5 | `wind_deg`, `clouds`, `hour`, `day`, `month` |
| **Categorical** | 3 | `city` (27 levels), `type` (2 levels), `weather` (4 levels) |
| **Textual** | 1 | `description` (free-text, not used for training) |
| **Datetime** | 1 | `datetime` (ISO 8601) |

## 📊 Descriptive Statistics (numeric attributes)

| Attribute | min | mean | std | max |
|-----------|-----|------|-----|-----|
| `temperature` (°C) | −1.99 | 9.47 | 4.52 | 20.00 |
| `feels_like` (°C) | −4.27 | 8.27 | 4.74 | 19.05 |
| `humidity` (%) | 29 | 68.06 | 17.01 | 100 |
| `pressure` (hPa) | 1011 | 1015.98 | 1.91 | 1021 |
| `wind_speed` (m/s) | 0.08 | 2.33 | 1.58 | 10.29 |
| `clouds` (%) | 0 | 60.72 | 36.61 | 100 |
| `visibility` (m) | 0 | 9717 | 1394 | 10000 |
| `pop` (prob.) | 0.00 | 0.15 | 0.31 | 1.00 |

### Class distribution for the categorical target `weather`

| Class | Count | Share |
|-------|-------|-------|
| Clouds | 686 | 61.97 % |
| Rain   | 242 | 21.86 % |
| Clear  | 170 | 15.36 % |
| Snow   | 9   | **0.81 %** ⚠️ |

➡️ A clear **class imbalance** is observed (the `Snow` class is heavily under-represented) — a limitation that will be addressed in Phase III using oversampling techniques.

## 🧼 Missing Values

| Column | Missing | Reason | Treatment |
|--------|---------|--------|-----------|
| `pop` | 27 | API does not return `pop` for `type = current` rows | Imputed with `0.0` in Phase II |
| all others | **0** | — | — |

**Total NaN in the dataset: 27 / (1107 × 20 = 22,140 cells) → 0.12 %** — very high data quality.

## 💡 Why these attributes?

- **Core meteorological variables** (`temperature`, `humidity`, `pressure`, `wind_*`, `clouds`) — standard physical inputs for atmospheric modelling.
- **`pop` and `visibility`** — precipitation / visibility indicators, useful for weather classification.
- **`hour`, `day`, `month`** — automatically derived to capture **temporal cycles** (diurnal / seasonal).
- **`city`** — enables per-city modelling or regional climate grouping.
- **lat / lon coordinates** are not stored in the CSV because they are static per city and can be re-joined from `weather_data_scraper.py`.

## ✅ Phase I Outcome

A complete, clean, and well-structured dataset for Kosovo:
- **1107 instances × 20 attributes**, only **0.12 % NaN** (easily handled),
- **3 ML tasks clearly defined** (regression, classification, time-series),
- descriptive statistics and class distribution fully documented,
- ready to feed the Phase II and Phase III models without requiring further major cleaning.

---

## 📁 Dataset Overview and Exploratory Insights

### Dataset at a Glance

| Parameter | Actual Value |
|-----------|--------------|
| Number of cities | 27 |
| Forecasts per city | 40 |
| Current snapshots per city | 1 |
| Total rows per run | ~1107 |
| Number of columns | 20 |
| Temporal resolution | 3 hours |
| Coverage horizon | 5 days |

### 🌍 Cities Analysed (sample)

| City | Region |
|------|--------|
| Pristina | Prishtinë |
| Prizren | Prizren |
| Peja | Pejë |
| Gjakova | Gjakovë |
| Mitrovica | Mitrovicë |
| Ferizaj | Ferizaj |
| Gjilan | Gjilan |

### 📈 Real Statistics Extracted from the Dataset (sample)

| City | Min Temp (°C) | Max Temp (°C) | Humidity (%) | Weather |
|------|---------------|---------------|--------------|---------|
| Pristina | 5.60 | 13.58 | 44 – 90 | Clouds / Clear |
| Prizren  | ~6.00 | ~15.00 | 40 – 85 | Clear / Clouds |
| Peja     | ~5.00 | ~14.00 | 50 – 88 | Clouds |
| Gjakova  | ~6.50 | ~15.50 | 45 – 80 | Clear |
| Mitrovica | ~5.50 | ~13.00 | 50 – 85 | Clouds |

### ⏱️ Temporal Structure

| Date | Intervals |
|------|-----------|
| 2026-03-23 | every 3 hours |
| 2026-03-24 | every 3 hours |
| 2026-03-25 | every 3 hours |
| 2026-03-26 | every 3 hours |
| 2026-03-27 | every 3 hours |

### 📊 Sample Raw Records

| City | Type | Datetime | Temp (°C) | Humidity (%) | Weather |
|------|------|----------|-----------|--------------|---------|
| Pristina | forecast | 2026-03-23 19:00 | 9.65 | 60 | Clouds |
| Pristina | forecast | 2026-03-24 01:00 | 5.60 | 90 | Clouds |
| Pristina | forecast | 2026-03-25 13:00 | 13.58 | 44 | Clear |

### 🌡️ Temperature Analysis

| Category | Result |
|----------|--------|
| Warmest cities | Pristina, Prizren |
| Coldest cities | Dragash, Deçan |
| Average difference | ~7 °C – 10 °C |
| Observed phenomenon | Urban Heat Island effect |

### 💨 Wind and Pressure Analysis

| Parameter | Range |
|-----------|-------|
| Wind Speed | 0.5 – 6 m/s |
| Pressure | 1010 – 1018 hPa |
| Wind Direction | 0° – 360° |

### 💧 Humidity Analysis

| Parameter | Value |
|-----------|-------|
| Minimum humidity | ~40 % |
| Maximum humidity | ~90 % |
| Mean | ~65 % |

### 🌧️ Probability of Precipitation

| Parameter | Value |
|-----------|-------|
| POP min | 0.0 |
| POP max | 1.0 |
| Mean | ~0.3 |

---

## 🤖 Selected Algorithm

In accordance with the professor's brief (*"Students must implement **any one** of the applicable ML algorithms..."*), the project focuses on a **single algorithm**: **Random Forest Regressor** — an ensemble of decision trees for regression (supervised learning).

| Phase | Random Forest Configuration | Status |
|-------|----------------------------|--------|
| Phase II | Baseline (100 trees) + Re-trained (300 trees, leaf ≥ 2, `max_features='sqrt'`) | ✅ Completed |
| Phase III | Optimised RF (automated hyperparameter tuning with GridSearchCV + feature engineering) | 🔜 Planned |

---

# 🧪 PHASE II — Analysis and Evaluation (Re-training)

## 🎯 Objective of the Phase

Phase II applies a **single algorithm — Random Forest Regressor** — to the task of predicting air temperature (°C) from meteorological features, and explicitly demonstrates the iterative process of **re-training**: a *baseline* configuration followed by a *re-trained* configuration with tuned, anti-overfitting hyperparameters. This directly addresses the phase title "Re-training".

- **Training script:** [`phase2_model_training.py`](phase2_model_training.py)
- **Full metrics log:** [`reports/phase2_metrics.txt`](reports/phase2_metrics.txt)
- **Machine-readable summary:** [`reports/phase2_summary.json`](reports/phase2_summary.json)

## 📸 Phase II Visualisations

Three key visualisations summarise the outputs of this phase:

<table>
  <tr>
    <td align="center"><b>🔥 Correlation Heat-map</b></td>
    <td align="center"><b>📈 Predicted vs. Actual</b></td>
    <td align="center"><b>🌲 Feature Importance</b></td>
  </tr>
  <tr>
    <td><img src="reports/phase2_correlation_heatmap.png" alt="Correlation Heatmap" width="320"/></td>
    <td><img src="reports/phase2_pred_vs_true.png" alt="Predicted vs Actual" width="320"/></td>
    <td><img src="reports/phase2_feature_importance.png" alt="Feature Importance" width="320"/></td>
  </tr>
  <tr>
    <td align="center"><sub>Correlations across meteorological features</sub></td>
    <td align="center"><sub>Baseline vs. Re-trained against the ideal diagonal</sub></td>
    <td align="center"><sub>Humidity, hour_sin, and pressure dominate</sub></td>
  </tr>
</table>

## 🌳 Why Random Forest Regressor?

| Reason | Explanation |
|--------|-------------|
| **Nature of the problem** | Temperature forecasting is a **supervised regression** problem — Random Forest is one of the most robust and battle-tested choices for it. |
| **Non-linearity** | Relationships among humidity, pressure, clouds, and temperature are non-linear; Random Forest captures them through deep, multi-way trees. |
| **Outlier robustness** | Trees split on thresholds, not distances, so extreme values do not distort the model as they would a linear regressor. |
| **No feature scaling required** | Trees are scale-invariant — this simplifies the pipeline and reduces the risk of pre-processing mistakes. |
| **Interpretability** | Provides built-in **feature importances**, helping verify that the model learned physically meaningful relationships, not artefacts. |
| **Natural fit for "Re-training"** | Exposes clear hyperparameters (`n_estimators`, `min_samples_leaf`, `max_features`) that allow explicit, documentable iteration. |

## 🧹 Data Preprocessing

1. **Drop rows with missing values** in `temperature`, `humidity`, `pressure` (the primary features).
2. **Fill `pop`** with `0.0` for `type = current` rows (where the API does not return this field).
3. **Cyclic encoding of time** using `sin / cos` for `hour` and `month`:
   - Reason: `23:00` and `00:00` are adjacent in time but appear numerically far apart. `sin / cos` preserves the cyclic adjacency.
4. **80 / 20 train / test split** with `random_state = 42` for reproducibility.
5. **`StandardScaler`** is fitted and saved purely for future pipeline compatibility; Random Forest itself does not require scaling.

### 📊 Split sizes

| Split | Row count | Share |
|-------|-----------|-------|
| **Train** | **885** | 80 % |
| **Test**  | **222** | 20 % |
| **Total** | 1107 | 100 % |

### 🧾 Input features (11)

`humidity, pressure, wind_speed, wind_deg, clouds, visibility, pop, hour_sin, hour_cos, month_sin, month_cos`

**Target:** `temperature` (°C, numeric).

## 🔎 Exploratory Data Analysis (EDA)

### Correlation Heat-map

![Correlation Heatmap](reports/phase2_correlation_heatmap.png)

| Feature | |corr| with `temperature` |
|---------|---------------------------|
| `humidity`   | **0.758** (strongest) |
| `pressure`   | 0.406 |
| `wind_speed` | 0.371 |
| `clouds`     | 0.199 |
| `visibility` | 0.173 |
| `pop`        | 0.147 |

➡️ Humidity is the strongest predictor — a physically expected result, since warmer air typically holds less relative humidity.

## 🔄 Re-training — two explicit iterations

| Parameter | Iteration 1 (baseline) | Iteration 2 (re-trained) |
|-----------|------------------------|--------------------------|
| `n_estimators` | 100 | **300** |
| `min_samples_leaf` | 1 | **2** |
| `max_features` | `None` (all) | **`sqrt`** |
| `max_depth` | `None` (unlimited) | `None` |
| `random_state` | 42 | 42 |

### 💡 Justification of the changes
- **300 trees instead of 100** → reduces variance by averaging more trees (stronger bagging).
- **`min_samples_leaf = 2`** → prevents the creation of leaves containing a single sample, a classic sign of overfitting.
- **`max_features = 'sqrt'`** → each tree sees only ~√11 ≈ 3 features per split → increases *decorrelation* among trees → a better-generalising ensemble.
- The primary goal is **reducing overfitting**, not maximising test-set R² on a single split.

## 📊 Results

| Model | MAE ↓ | RMSE ↓ | R² (train) | R² (test) | R² (5-fold CV) |
|-------|-------|--------|------------|-----------|----------------|
| **Baseline RF**   | **1.029 °C** | **1.519 °C** | 0.9828 | 0.8831 | 0.8677 |
| **Re-trained RF** | 1.132 °C | 1.546 °C | **0.9561** | 0.8789 | **0.8742** |

### 📈 Predicted vs. Actual

![Predicted vs Actual Temperature](reports/phase2_pred_vs_true.png)

The points cluster tightly along the ideal diagonal (dashed line) — the model matches the actual temperature closely. Larger deviations appear only at the extremes (very hot / very cold), which are under-represented in the dataset.

### 🔥 Feature Importance (re-trained model)

![Feature Importance](reports/phase2_feature_importance.png)

| Feature | Importance |
|---------|-----------|
| `humidity`   | **0.342** |
| `hour_sin`   | 0.153 |
| `pressure`   | 0.132 |
| `wind_speed` | 0.119 |
| `clouds`     | 0.095 |
| `hour_cos`   | 0.083 |
| `wind_deg`   | 0.053 |
| `pop`        | 0.018 |
| `visibility` | 0.005 |
| `month_sin`  | ~0.000 |
| `month_cos`  | ~0.000 |

➡️ Humidity, hour of day (sin/cos), and pressure are the three most informative features. `month_*` are effectively zero because the dataset spans only ~5 days, leaving no month-level variance.

## 💬 Discussion of Results

1. **R² (test) ≈ 0.88** — the model explains ~88 % of the temperature variance with **MAE ≈ 1 °C** — a strong outcome for a small dataset (1107 rows, 5 days).

2. **Baseline vs. Re-trained — an important lesson:**
   - The baseline shows **R² (train) = 0.9828** vs. **R² (test) = 0.8831** → a ~10-point gap → clear **overfitting**.
   - The re-trained model shows **R² (train) = 0.9561** vs. **R² (test) = 0.8789** → a ~7.7-point gap → overfitting noticeably reduced.
   - **5-fold CV R² rose** from 0.8677 → **0.8742** → the re-trained model **generalises better**.
   - This is the classic case where a single fixed test split can mislead: the test set sometimes hides overfitting that CV reveals, which is why both were reported.

3. **Feature importance confirms the physics** — humidity dominates, followed by hour of day and pressure, so the model learned meaningful relationships rather than artefacts.

4. **Main limitation** — the dataset spans only 5 days, with no monthly/seasonal variation, which is why `month_sin/cos` have zero importance. Phase III will address this through a longer data-collection window and more advanced tuning.

## 🧾 Phase II Conclusions

1. **A single algorithm — Random Forest Regressor — was implemented**, as required by the professor ("any one of the algorithms").
2. **The re-training process** was demonstrated through two explicit iterations, with a documented justification for every hyperparameter change.
3. **R² (test) = 0.88** and **MAE = 1.03 °C** were achieved by the baseline; the re-trained model improved generalisation (CV R² rose from 0.868 → 0.874).
4. The **train / test split (885 / 222)** is explicitly documented, and a train-vs-test comparison exposes the overfitting pattern.
5. **Feature importance** confirms a physically interpretable model.
6. All models and reports are serialised in [`models/`](models/) and [`reports/`](reports/), ready for direct comparison in Phase III.
