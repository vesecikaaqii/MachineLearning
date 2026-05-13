import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from sklearn.model_selection import KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
np.random.seed(42)

DATA_PATH    = "kosovo_weather_dataset.csv"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports/phase3_evaluation"
os.makedirs(REPORTS_DIR, exist_ok=True)

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("PHASE III  -  Rigorous Evaluation of the Phase II Baseline")
log("=" * 70)

df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

df = df.dropna(subset=["temperature_2m", "relative_humidity_2m", "surface_pressure"]).reset_index(drop=True)
df["precipitation"] = df["precipitation"].fillna(0.0)

df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24.0)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24.0)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

FEATURES = [
    "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "cloud_cover", "precipitation",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
TARGET = "temperature_2m"

cfg = dict(n_estimators=100, max_depth=None, min_samples_leaf=1,
           random_state=42, n_jobs=-1)

log(f"Rows after cleaning : {len(df)}")
log(f"Features ({len(FEATURES)})       : {FEATURES}")
log(f"Target              : {TARGET} (deg C)")
log(f"RF baseline config  : {cfg}")

log("\n" + "-" * 70)
log("1. CHRONOLOGICAL SPLIT (honest forecasting evaluation)")
log("-" * 70)

df_sorted = df.sort_values("datetime").reset_index(drop=True)
cutoff_time = df_sorted["datetime"].quantile(0.80)

train_df = df_sorted[df_sorted["datetime"] <  cutoff_time]
test_df  = df_sorted[df_sorted["datetime"] >= cutoff_time]

X_train_c = train_df[FEATURES].values
y_train_c = train_df[TARGET].values
X_test_c  = test_df[FEATURES].values
y_test_c  = test_df[TARGET].values

model_c = RandomForestRegressor(**cfg)
model_c.fit(X_train_c, y_train_c)
pred_c = model_c.predict(X_test_c)

mae_c  = mean_absolute_error(y_test_c, pred_c)
rmse_c = np.sqrt(mean_squared_error(y_test_c, pred_c))
r2_c   = r2_score(y_test_c, pred_c)
r2_tr_c = r2_score(y_train_c, model_c.predict(X_train_c))

log(f"Train period   : {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
log(f"Test  period   : {test_df['datetime'].min()} -> {test_df['datetime'].max()}")
log(f"Split sizes    : {len(train_df)} / {len(test_df)}")
log(f"MAE  (test)    : {mae_c:.3f} C")
log(f"RMSE (test)    : {rmse_c:.3f} C")
log(f"R^2  (train)   : {r2_tr_c:.4f}")
log(f"R^2  (test)    : {r2_c:.4f}")
log(f"Train-test gap : {r2_tr_c - r2_c:.4f}")

log("\n" + "-" * 70)
log("2. 5-FOLD CROSS-VALIDATION")
log("-" * 70)

X = df[FEATURES].values
y = df[TARGET].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
    m = RandomForestRegressor(**cfg)
    m.fit(X[tr_idx], y[tr_idx])
    p = m.predict(X[te_idx])
    fm = dict(
        fold=fold,
        MAE=mean_absolute_error(y[te_idx], p),
        RMSE=np.sqrt(mean_squared_error(y[te_idx], p)),
        R2=r2_score(y[te_idx], p),
    )
    fold_metrics.append(fm)
    log(f"  Fold {fold}: MAE={fm['MAE']:.3f}  RMSE={fm['RMSE']:.3f}  R^2={fm['R2']:.4f}")

cv_mae  = np.mean([f["MAE"] for f in fold_metrics])
cv_mae_std  = np.std([f["MAE"] for f in fold_metrics])
cv_rmse = np.mean([f["RMSE"] for f in fold_metrics])
cv_r2   = np.mean([f["R2"]  for f in fold_metrics])
cv_r2_std = np.std([f["R2"] for f in fold_metrics])

log(f"  CV mean  : MAE={cv_mae:.3f} (+/- {cv_mae_std:.3f})  RMSE={cv_rmse:.3f}  R^2={cv_r2:.4f} (+/- {cv_r2_std:.4f})")

log("\n" + "-" * 70)
log("3. RESIDUAL DIAGNOSTICS")
log("-" * 70)

residuals = y_test_c - pred_c

plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=60, color="steelblue", edgecolor="white")
plt.axvline(0, color="black", linestyle="--", lw=1)
plt.title("Phase III - Residual histogram (chronological hold-out)")
plt.xlabel("Residual (actual - predicted) [C]")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "residuals.png"), dpi=150)
plt.close()

plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Phase III - Q-Q plot of residuals")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "qq_plot.png"), dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(pred_c, residuals, alpha=0.4, s=10, color="steelblue")
plt.axhline(0, color="black", linestyle="--", lw=1)
plt.title("Phase III - Residuals vs predicted")
plt.xlabel("Predicted temperature [C]")
plt.ylabel("Residual [C]")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "residuals_vs_pred.png"), dpi=150)
plt.close()

log(f"Residual mean   : {residuals.mean():.4f}")
log(f"Residual std    : {residuals.std():.4f}")
log(f"Residual min/max: {residuals.min():.3f} / {residuals.max():.3f}")

log("\n" + "-" * 70)
log("4. ERROR BREAKDOWN BY DIMENSION")
log("-" * 70)

err_df = test_df.copy()
err_df["pred"] = pred_c
err_df["abs_err"] = np.abs(err_df[TARGET] - err_df["pred"])

hour_err = err_df.groupby("hour")["abs_err"].agg(["mean", "median", "count"]).round(3)
hour_err.to_csv(os.path.join(REPORTS_DIR, "error_by_hour.csv"))

plt.figure(figsize=(8, 4))
hour_err["mean"].plot(kind="bar", color="steelblue")
plt.title("Phase III - Mean absolute error by hour of day")
plt.xlabel("Hour")
plt.ylabel("MAE [C]")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "error_by_hour.png"), dpi=150)
plt.close()

city_err = err_df.groupby("city")["abs_err"].agg(["mean", "median", "count"]).round(3).sort_values("mean")
city_err.to_csv(os.path.join(REPORTS_DIR, "error_by_city.csv"))

plt.figure(figsize=(8, 7))
city_err["mean"].plot(kind="barh", color="steelblue")
plt.title("Phase III - Mean absolute error by city")
plt.xlabel("MAE [C]")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "error_by_city.png"), dpi=150)
plt.close()

err_df["temp_quartile"] = pd.qcut(err_df[TARGET], 4, labels=["Q1 (cold)", "Q2", "Q3", "Q4 (warm)"])
quartile_err = err_df.groupby("temp_quartile", observed=True)["abs_err"].agg(["mean", "median", "count"]).round(3)
quartile_err.to_csv(os.path.join(REPORTS_DIR, "error_by_quartile.csv"))

log("Error by hour (top 3 worst hours):")
log(hour_err.sort_values("mean", ascending=False).head(3).to_string())
log("\nError by city (top 3 worst cities):")
log(city_err.sort_values("mean", ascending=False).head(3).to_string())
log("\nError by temperature quartile:")
log(quartile_err.to_string())

log("\n" + "-" * 70)
log("5. LEARNING CURVE")
log("-" * 70)

train_sizes = np.linspace(0.1, 1.0, 6)
lc_sizes, lc_train, lc_val = learning_curve(
    RandomForestRegressor(**cfg),
    X, y,
    train_sizes=train_sizes,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=42,
)

lc_train_mae = -lc_train.mean(axis=1)
lc_val_mae   = -lc_val.mean(axis=1)

plt.figure(figsize=(7, 5))
plt.plot(lc_sizes, lc_train_mae, "o-", label="Train MAE", color="steelblue")
plt.plot(lc_sizes, lc_val_mae,   "o-", label="CV    MAE", color="orangered")
plt.title("Phase III - Learning curve")
plt.xlabel("Training-set size")
plt.ylabel("MAE [C]")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "learning_curve.png"), dpi=150)
plt.close()

log(f"Learning-curve sizes : {lc_sizes.tolist()}")
log(f"CV MAE at each size  : {[round(v, 3) for v in lc_val_mae]}")

log("\n" + "-" * 70)
log("6. PERMUTATION IMPORTANCE (on chronological test set)")
log("-" * 70)

perm = permutation_importance(
    model_c, X_test_c, y_test_c,
    n_repeats=10, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error",
)

perm_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std":  perm.importances_std,
}).sort_values("importance_mean", ascending=False)

perm_df.to_csv(os.path.join(REPORTS_DIR, "permutation_importance.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.barh(perm_df["feature"][::-1], perm_df["importance_mean"][::-1], xerr=perm_df["importance_std"][::-1],
         color="steelblue")
plt.title("Phase III - Permutation importance (chronological test set)")
plt.xlabel("Increase in MAE when feature is shuffled [C]")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "permutation_importance.png"), dpi=150)
plt.close()

log(perm_df.to_string(index=False))

summary = {
    "phase": "III - Evaluation",
    "rows_used": int(len(df)),
    "features": FEATURES,
    "target": TARGET,
    "config": cfg,
    "chronological_split": {
        "train_size": int(len(train_df)),
        "test_size":  int(len(test_df)),
        "train_period": [str(train_df["datetime"].min()), str(train_df["datetime"].max())],
        "test_period":  [str(test_df["datetime"].min()),  str(test_df["datetime"].max())],
        "metrics": {"MAE": mae_c, "RMSE": rmse_c, "R2_train": r2_tr_c, "R2_test": r2_c},
    },
    "cv_5fold": {
        "folds": fold_metrics,
        "mean": {"MAE": cv_mae, "RMSE": cv_rmse, "R2": cv_r2},
        "std":  {"MAE": cv_mae_std, "R2": cv_r2_std},
    },
    "residuals": {
        "mean": float(residuals.mean()),
        "std":  float(residuals.std()),
        "min":  float(residuals.min()),
        "max":  float(residuals.max()),
    },
    "learning_curve": {
        "sizes": lc_sizes.tolist(),
        "cv_mae": [float(v) for v in lc_val_mae],
        "train_mae": [float(v) for v in lc_train_mae],
    },
    "permutation_importance": perm_df.to_dict(orient="records"),
}

with open(os.path.join(REPORTS_DIR, "phase3_evaluation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(os.path.join(REPORTS_DIR, "phase3_evaluation_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log("\n" + "=" * 70)
log(f"Phase III evaluation complete. Artefacts -> {REPORTS_DIR}/")
log("=" * 70)
