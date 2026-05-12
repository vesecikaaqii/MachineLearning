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
from sklearn.ensemble import RandomForestRegressor
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
log("PHASE III  -  Re-training (feature engineering + hyperparameter tuning)")
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

log(f"Base features        : {len(BASE_FEATURES)}")
log(f"New engineered feats : {len(LAG_FEATURES)}")
log(f"City one-hot dummies : {len(CITY_FEATURES)}")
log(f"Total features       : {len(ALL_FEATURES)}")

df = df.dropna(subset=LAG_FEATURES).reset_index(drop=True)
log(f"Rows after dropping lag-NaN : {len(df)}")

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

log(f"Train period : {train_df['datetime'].min()} -> {train_df['datetime'].max()}  ({len(train_df)} rows)")
log(f"Test  period : {test_df['datetime'].min()}  -> {test_df['datetime'].max()}   ({len(test_df)} rows)")

log("\n" + "-" * 70)
log("4. REFERENCE BASELINES (on chronological test set)")
log("-" * 70)

gm_pred  = np.full(len(y_test), y_train.mean())
gm_mae   = mean_absolute_error(y_test, gm_pred)
gm_rmse  = np.sqrt(mean_squared_error(y_test, gm_pred))
gm_r2    = r2_score(y_test, gm_pred)

city_means = train_df.groupby("city")[TARGET].mean()
pc_pred = test_df["city"].map(city_means).values
pc_mae  = mean_absolute_error(y_test, pc_pred)
pc_rmse = np.sqrt(mean_squared_error(y_test, pc_pred))
pc_r2   = r2_score(y_test, pc_pred)

persist_pred = test_df["temp_lag_1h"].values
persist_mae  = mean_absolute_error(y_test, persist_pred)
persist_rmse = np.sqrt(mean_squared_error(y_test, persist_pred))
persist_r2   = r2_score(y_test, persist_pred)

log(f"Global mean      : MAE={gm_mae:.3f}  RMSE={gm_rmse:.3f}  R^2={gm_r2:.4f}")
log(f"Per-city mean    : MAE={pc_mae:.3f}  RMSE={pc_rmse:.3f}  R^2={pc_r2:.4f}")
log(f"1-h persistence  : MAE={persist_mae:.3f}  RMSE={persist_rmse:.3f}  R^2={persist_r2:.4f}")

log("\n" + "-" * 70)
log("5. GRIDSEARCHCV (TimeSeriesSplit, MAE scoring)")
log("-" * 70)

param_grid = {
    "n_estimators":     [100, 300],
    "max_depth":        [None, 20],
    "min_samples_leaf": [1, 5],
}
tscv = TimeSeriesSplit(n_splits=3)

t0 = time.time()
gs = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_train, y_train)
elapsed = time.time() - t0

best_params = gs.best_params_
log(f"Search completed in {elapsed:.1f}s")
log(f"Best params : {best_params}")
log(f"Best CV MAE : {-gs.best_score_:.3f}")

best_model = gs.best_estimator_
pred_final = best_model.predict(X_test)

final_mae  = mean_absolute_error(y_test, pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, pred_final))
final_r2   = r2_score(y_test, pred_final)
final_r2_tr = r2_score(y_train, best_model.predict(X_train))

log(f"Final test  : MAE={final_mae:.3f}  RMSE={final_rmse:.3f}  R^2={final_r2:.4f}  (train R^2={final_r2_tr:.4f}, gap={final_r2_tr-final_r2:.4f})")

joblib.dump(best_model, os.path.join(MODELS_DIR, "rf_model_v2.pkl"))

log("\n" + "-" * 70)
log("6. MULTI-HORIZON FORECASTING")
log("-" * 70)

HORIZONS = [1, 3, 6, 12, 24, 48]
horizon_results = {}

df_h = df.sort_values(["city", "datetime"]).reset_index(drop=True)
g_target = df_h.groupby("city")["temperature_2m"]

for h in HORIZONS:
    df_h[f"target_t+{h}"] = g_target.shift(-h)

df_h_sorted = df_h.sort_values("datetime").reset_index(drop=True)
cutoff_h = df_h_sorted["datetime"].quantile(0.80)

for h in HORIZONS:
    target_col = f"target_t+{h}"
    df_h_h = df_h_sorted.dropna(subset=[target_col])

    tr = df_h_h[df_h_h["datetime"] <  cutoff_h]
    te = df_h_h[df_h_h["datetime"] >= cutoff_h]

    if len(te) == 0:
        log(f"  h=+{h:>2}h : skipped (no test rows after cutoff)")
        continue

    X_tr = tr[ALL_FEATURES].values
    y_tr = tr[target_col].values
    X_te = te[ALL_FEATURES].values
    y_te = te[target_col].values

    m = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)

    h_mae  = mean_absolute_error(y_te, p)
    h_rmse = np.sqrt(mean_squared_error(y_te, p))
    h_r2   = r2_score(y_te, p)
    horizon_results[h] = dict(MAE=h_mae, RMSE=h_rmse, R2=h_r2,
                              train_size=int(len(tr)), test_size=int(len(te)))
    log(f"  h=+{h:>2}h : MAE={h_mae:.3f}  RMSE={h_rmse:.3f}  R^2={h_r2:.4f}")

log("\n" + "-" * 70)
log("7. PLOTS")
log("-" * 70)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, pred_final, alpha=0.45, s=10, color="steelblue", label="predictions")
lo, hi = float(y_test.min()), float(y_test.max())
plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
plt.xlabel("Actual temperature [C]")
plt.ylabel("Predicted temperature [C]")
plt.title(f"Phase III - Predicted vs Actual (MAE={final_mae:.2f} C, R²={final_r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "phase3_pred_vs_true.png"), dpi=150)
plt.close()

if horizon_results:
    hs   = sorted(horizon_results.keys())
    maes = [horizon_results[h]["MAE"] for h in hs]
    plt.figure(figsize=(7, 5))
    plt.plot(hs, maes, "o-", color="steelblue")
    plt.xlabel("Forecast horizon [hours ahead]")
    plt.ylabel("MAE [C]")
    plt.title("Phase III - Forecast error vs horizon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "phase3_multihorizon.png"), dpi=150)
    plt.close()

labels = ["Global\nmean", "Per-city\nmean", "1-h\npersistence", "Phase III\nfinal RF"]
maes   = [gm_mae, pc_mae, persist_mae, final_mae]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, maes, color=["lightgray", "lightgray", "lightgray", "steelblue"])
for bar, v in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
plt.ylabel("MAE [C]")
plt.title("Phase III - Final model vs reference baselines")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "phase3_baselines.png"), dpi=150)
plt.close()

imp = pd.Series(best_model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
top20 = imp.head(20)
plt.figure(figsize=(8, 8))
top20[::-1].plot(kind="barh", color="steelblue")
plt.title("Phase III - Top-20 feature importance (impurity)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "phase3_feature_importance.png"), dpi=150)
plt.close()

log("Top-10 features:")
log(imp.head(10).to_string())

summary = {
    "phase": "III - Re-training",
    "rows_used": int(len(df)),
    "train_size": int(len(train_df)),
    "test_size":  int(len(test_df)),
    "features": {
        "base":   BASE_FEATURES,
        "engineered": LAG_FEATURES,
        "city_ohe":   CITY_FEATURES,
        "total_count": len(ALL_FEATURES),
    },
    "target": TARGET,
    "best_params": best_params,
    "best_cv_mae": float(-gs.best_score_),
    "baselines": {
        "global_mean":     {"MAE": gm_mae,      "RMSE": gm_rmse,      "R2": gm_r2},
        "per_city_mean":   {"MAE": pc_mae,      "RMSE": pc_rmse,      "R2": pc_r2},
        "persistence_1h":  {"MAE": persist_mae, "RMSE": persist_rmse, "R2": persist_r2},
    },
    "phase3_final": {
        "MAE": final_mae, "RMSE": final_rmse,
        "R2_train": final_r2_tr, "R2_test": final_r2,
        "train_test_gap": final_r2_tr - final_r2,
    },
    "multi_horizon": {str(k): v for k, v in horizon_results.items()},
    "top10_feature_importance": imp.head(10).to_dict(),
}

with open(os.path.join(REPORTS_DIR, "phase3_retraining_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(os.path.join(REPORTS_DIR, "phase3_retraining_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log("\n" + "=" * 70)
log(f"Phase III re-training complete.")
log(f"Model    -> models/rf_model_v2.pkl")
log(f"Reports  -> {REPORTS_DIR}/")
log("=" * 70)
