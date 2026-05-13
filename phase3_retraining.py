import os
import json
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
np.random.seed(42)

DATA_PATH   = "kosovo_weather_dataset.csv"
MODELS_DIR  = "models"
REPORTS_DIR = "reports/phase3_retraining"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("PHASE III  -  Re-training (feature eng + tuning for multiple models)")
log("=" * 70)

df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

df = df.dropna(subset=["temperature_2m", "relative_humidity_2m", "surface_pressure"]).reset_index(drop=True)
df["precipitation"] = df["precipitation"].fillna(0.0)

df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24.0)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24.0)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

log(f"Loaded rows : {len(df)}")

log("\n" + "-" * 70)
log("2. FEATURE ENGINEERING (lag / rolling / delta / interaction / city OHE)")
log("-" * 70)

g_temp = df.groupby("city")["temperature_2m"]
g_hum  = df.groupby("city")["relative_humidity_2m"]
g_pre  = df.groupby("city")["surface_pressure"]

df["temp_lag_1h"]     = g_temp.shift(1)
df["temp_lag_3h"]     = g_temp.shift(3)
df["temp_lag_24h"]    = g_temp.shift(24)
df["humidity_lag_1h"] = g_hum.shift(1)
df["pressure_lag_1h"] = g_pre.shift(1)

df["temp_roll_3h_mean"]  = g_temp.shift(1).groupby(df["city"]).rolling(3).mean().reset_index(level=0, drop=True)
df["temp_roll_24h_mean"] = g_temp.shift(1).groupby(df["city"]).rolling(24).mean().reset_index(level=0, drop=True)
df["hum_roll_3h_mean"]   = g_hum.shift(1).groupby(df["city"]).rolling(3).mean().reset_index(level=0, drop=True)
df["pre_roll_3h_mean"]   = g_pre.shift(1).groupby(df["city"]).rolling(3).mean().reset_index(level=0, drop=True)

df["pressure_delta_3h"] = g_pre.shift(1) - g_pre.shift(4)
df["humidity_delta_3h"] = g_hum.shift(1) - g_hum.shift(4)
df["temp_delta_1h"]     = g_temp.shift(1) - g_temp.shift(2)

df["hum_x_clouds"]      = df["relative_humidity_2m"] * df["cloud_cover"]
df["wind_x_pre_delta"]  = df["wind_speed_10m"] * df["pressure_delta_3h"]

city_ohe = pd.get_dummies(df["city"], prefix="city", dtype=int)
df = pd.concat([df, city_ohe], axis=1)

LAG_FEATURES = [
    "temp_lag_1h", "temp_lag_3h", "temp_lag_24h",
    "humidity_lag_1h", "pressure_lag_1h",
    "temp_roll_3h_mean", "temp_roll_24h_mean", "hum_roll_3h_mean", "pre_roll_3h_mean",
    "pressure_delta_3h", "humidity_delta_3h", "temp_delta_1h",
    "hum_x_clouds", "wind_x_pre_delta",
]
BASE_FEATURES = [
    "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "cloud_cover", "precipitation",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
CITY_FEATURES = list(city_ohe.columns)

ALL_FEATURES = BASE_FEATURES + LAG_FEATURES + CITY_FEATURES
TARGET = "temperature_2m"

df = df.dropna(subset=LAG_FEATURES).reset_index(drop=True)

log("\n" + "-" * 70)
log("3. CHRONOLOGICAL SPLIT")
log("-" * 70)

df_sorted = df.sort_values("datetime").reset_index(drop=True)
cutoff = df_sorted["datetime"].quantile(0.80)
train_df = df_sorted[df_sorted["datetime"] <  cutoff].copy()
test_df  = df_sorted[df_sorted["datetime"] >= cutoff].copy()

X_train = train_df[ALL_FEATURES].values
y_train = train_df[TARGET].values
X_test  = test_df[ALL_FEATURES].values
y_test  = test_df[TARGET].values

log("\n" + "-" * 70)
log("4. REFERENCE BASELINES")
log("-" * 70)

gm_pred  = np.full(len(y_test), y_train.mean())
gm_mae   = mean_absolute_error(y_test, gm_pred)
pc_pred  = test_df["city"].map(train_df.groupby("city")[TARGET].mean()).values
pc_mae   = mean_absolute_error(y_test, pc_pred)
persist_pred = test_df["temp_lag_1h"].values
persist_mae  = mean_absolute_error(y_test, persist_pred)

log(f"Global mean      : MAE={gm_mae:.3f}")
log(f"Per-city mean    : MAE={pc_mae:.3f}")
log(f"1-h persistence  : MAE={persist_mae:.3f}")

log("\n" + "-" * 70)
log("5. GRIDSEARCHCV (TimeSeriesSplit, MAE scoring)")
log("-" * 70)

model_configs = {
    "RandomForest": {
        "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
        "param_grid": {"n_estimators": [100, 300], "max_depth": [None, 20], "min_samples_leaf": [1, 5]}
    },
    "GradientBoosting": {
        "estimator": GradientBoostingRegressor(random_state=42),
        "param_grid": {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    },
    "LinearRegression": {
        "estimator": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "param_grid": {"lr__fit_intercept": [True, False]}
    }
}

tscv = TimeSeriesSplit(n_splits=3)
best_models = {}
tuned_metrics = {}

for name, config in model_configs.items():
    log(f"\nTuning {name}...")
    t0 = time.time()
    gs = GridSearchCV(
        config["estimator"], config["param_grid"], cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0
    
    best_model = gs.best_estimator_
    best_models[name] = best_model
    
    pred_final = best_model.predict(X_test)
    final_mae  = mean_absolute_error(y_test, pred_final)
    final_rmse = np.sqrt(mean_squared_error(y_test, pred_final))
    final_r2   = r2_score(y_test, pred_final)
    
    tuned_metrics[name] = {"MAE": final_mae, "RMSE": final_rmse, "R2": final_r2, "BestParams": gs.best_params_}

    log(f"  Best params : {gs.best_params_}")
    log(f"  Test MAE    : {final_mae:.3f} C")
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"{name.lower()}_retrained.pkl"))

log("\n" + "-" * 70)
log("6. MULTI-HORIZON FORECASTING (On Best Models)")
log("-" * 70)

HORIZONS = [1, 3, 6, 12, 24, 48]
df_h = df.sort_values(["city", "datetime"]).reset_index(drop=True)
g_target = df_h.groupby("city")["temperature_2m"]
for h in HORIZONS:
    df_h[f"target_t+{h}"] = g_target.shift(-h)

df_h_sorted = df_h.sort_values("datetime").reset_index(drop=True)
cutoff_h = df_h_sorted["datetime"].quantile(0.80)

horizon_results = {name: {} for name in best_models.keys()}

for h in HORIZONS:
    target_col = f"target_t+{h}"
    df_h_h = df_h_sorted.dropna(subset=[target_col])
    tr, te = df_h_h[df_h_h["datetime"] <  cutoff_h], df_h_h[df_h_h["datetime"] >= cutoff_h]
    
    if len(te) == 0: continue
    
    X_tr, y_tr = tr[ALL_FEATURES].values, tr[target_col].values
    X_te, y_te = te[ALL_FEATURES].values, te[target_col].values
    
    for name, best_model in best_models.items():
        # Clone architecture to refit cleanly on multi-horizon targets
        from sklearn.base import clone
        m = clone(best_model)
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        horizon_results[name][h] = mean_absolute_error(y_te, p)

log("\n" + "-" * 70)
log("7. PLOTS")
log("-" * 70)

plt.figure(figsize=(8, 5))
for name in best_models.keys():
    hs = sorted(horizon_results[name].keys())
    maes = [horizon_results[name][h] for h in hs]
    plt.plot(hs, maes, "o-", label=name)
plt.xlabel("Forecast horizon [hours ahead]")
plt.ylabel("MAE [C]")
plt.title("Phase III - Forecast Error vs Horizon (All Models)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "phase3_multihorizon_comparison.png"), dpi=150)
plt.close()

labels = ["Global\nmean", "Per-city\nmean", "1-h\npersist"] + list(best_models.keys())
maes   = [gm_mae, pc_mae, persist_mae] + [tuned_metrics[name]["MAE"] for name in best_models.keys()]

plt.figure(figsize=(10, 5))
bars = plt.bar(labels, maes, color=["lightgray", "lightgray", "lightgray", "steelblue", "darkorange", "forestgreen"])
for bar, v in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
plt.ylabel("MAE [C]")
plt.title("Phase III - Final Models vs Reference Baselines")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "phase3_baselines_comparison.png"), dpi=150)
plt.close()

all_importances = {}
for name, model in best_models.items():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.named_steps["lr"].coef_)
        
    imp = pd.Series(importances, index=ALL_FEATURES).sort_values(ascending=False)
    all_importances[name] = imp.head(10).to_dict()
    
    plt.figure(figsize=(8, 8))
    imp.head(20)[::-1].plot(kind="barh", color="steelblue")
    plt.title(f"Phase III - Top-20 Feature Importance ({name})")
    plt.xlabel("Importance / Absolute Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{name.lower()}_feature_importance.png"), dpi=150)
    plt.close()

summary = {
    "phase": "III - Re-training",
    "rows_used": int(len(df)),
    "features": {
        "base": BASE_FEATURES,
        "engineered": LAG_FEATURES,
        "city_ohe": CITY_FEATURES,
        "total_count": len(ALL_FEATURES),
    },
    "tuned_metrics": tuned_metrics,
    "multi_horizon_mae": horizon_results,
    "top10_feature_importances": all_importances
}

with open(os.path.join(REPORTS_DIR, "phase3_retraining_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(os.path.join(REPORTS_DIR, "phase3_retraining_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log("\n" + "=" * 70)
log(f"Phase III re-training complete.")
log("=" * 70)