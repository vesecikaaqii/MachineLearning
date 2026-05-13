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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
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
log("PHASE III  -  Rigorous Evaluation of Multiple Baselines")
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

log(f"Rows after cleaning : {len(df)}")
log(f"Features ({len(FEATURES)})       : {FEATURES}")
log(f"Target              : {TARGET} (deg C)")

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    "LinearRegression": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
}

overall_summary = {
    "phase": "III - Evaluation",
    "rows_used": int(len(df)),
    "features": FEATURES,
    "target": TARGET,
    "models_evaluated": list(models.keys()),
    "results": {}
}

# 1. Prepare Data for Chronological Split (Done once for all models)
df_sorted = df.sort_values("datetime").reset_index(drop=True)
cutoff_time = df_sorted["datetime"].quantile(0.80)

train_df = df_sorted[df_sorted["datetime"] <  cutoff_time]
test_df  = df_sorted[df_sorted["datetime"] >= cutoff_time]

X_train_c = train_df[FEATURES].values
y_train_c = train_df[TARGET].values
X_test_c  = test_df[FEATURES].values
y_test_c  = test_df[TARGET].values

X = df[FEATURES].values
y = df[TARGET].values

for name, model_instance in models.items():
    log("\n" + "=" * 70)
    log(f"EVALUATING: {name}")
    log("=" * 70)
    
    model_summary = {}
    
    log("\n1. CHRONOLOGICAL SPLIT (honest forecasting evaluation)")
    
    model_c = clone(model_instance)
    model_c.fit(X_train_c, y_train_c)
    pred_c = model_c.predict(X_test_c)

    mae_c  = mean_absolute_error(y_test_c, pred_c)
    rmse_c = np.sqrt(mean_squared_error(y_test_c, pred_c))
    r2_c   = r2_score(y_test_c, pred_c)
    r2_tr_c = r2_score(y_train_c, model_c.predict(X_train_c))

    log(f"MAE  (test)    : {mae_c:.3f} C")
    log(f"RMSE (test)    : {rmse_c:.3f} C")
    log(f"R^2  (train)   : {r2_tr_c:.4f}")
    log(f"R^2  (test)    : {r2_c:.4f}")
    log(f"Train-test gap : {r2_tr_c - r2_c:.4f}")
    
    model_summary["chronological_split"] = {
        "metrics": {"MAE": mae_c, "RMSE": rmse_c, "R2_train": r2_tr_c, "R2_test": r2_c}
    }

    log("\n2. 5-FOLD CROSS-VALIDATION")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
        m = clone(model_instance)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict(X[te_idx])
        fm = dict(
            fold=fold,
            MAE=mean_absolute_error(y[te_idx], p),
            RMSE=np.sqrt(mean_squared_error(y[te_idx], p)),
            R2=r2_score(y[te_idx], p),
        )
        fold_metrics.append(fm)

    cv_mae  = np.mean([f["MAE"] for f in fold_metrics])
    cv_mae_std  = np.std([f["MAE"] for f in fold_metrics])
    cv_rmse = np.mean([f["RMSE"] for f in fold_metrics])
    cv_r2   = np.mean([f["R2"]  for f in fold_metrics])
    cv_r2_std = np.std([f["R2"] for f in fold_metrics])

    log(f"CV mean : MAE={cv_mae:.3f} (+/- {cv_mae_std:.3f})  RMSE={cv_rmse:.3f}  R^2={cv_r2:.4f}")
    
    model_summary["cv_5fold"] = {
        "mean": {"MAE": cv_mae, "RMSE": cv_rmse, "R2": cv_r2},
        "std":  {"MAE": cv_mae_std, "R2": cv_r2_std},
    }

    log("\n3. RESIDUAL DIAGNOSTICS")
    residuals = y_test_c - pred_c

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=60, color="steelblue", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", lw=1)
    plt.title(f"Phase III - Residuals ({name})")
    plt.xlabel("Residual (actual - predicted) [C]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{name.lower()}_residuals.png"), dpi=150)
    plt.close()

    model_summary["residuals"] = {
        "mean": float(residuals.mean()),
        "std":  float(residuals.std()),
        "min":  float(residuals.min()),
        "max":  float(residuals.max()),
    }

    log("\n4. ERROR BREAKDOWN BY DIMENSION")
    err_df = test_df.copy()
    err_df["pred"] = pred_c
    err_df["abs_err"] = np.abs(err_df[TARGET] - err_df["pred"])

    hour_err = err_df.groupby("hour")["abs_err"].agg(["mean", "median", "count"]).round(3)
    hour_err.to_csv(os.path.join(REPORTS_DIR, f"{name.lower()}_error_by_hour.csv"))

    city_err = err_df.groupby("city")["abs_err"].agg(["mean", "median", "count"]).round(3).sort_values("mean")
    city_err.to_csv(os.path.join(REPORTS_DIR, f"{name.lower()}_error_by_city.csv"))

    log("\n5. LEARNING CURVE")
    train_sizes = np.linspace(0.1, 1.0, 6)
    
    lc_sizes, lc_train, lc_val = learning_curve(
        clone(model_instance), X, y,
        train_sizes=train_sizes, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=42
    )

    lc_train_mae = -lc_train.mean(axis=1)
    lc_val_mae   = -lc_val.mean(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(lc_sizes, lc_train_mae, "o-", label="Train MAE", color="steelblue")
    plt.plot(lc_sizes, lc_val_mae,   "o-", label="CV    MAE", color="orangered")
    plt.title(f"Phase III - Learning Curve ({name})")
    plt.xlabel("Training-set size")
    plt.ylabel("MAE [C]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{name.lower()}_learning_curve.png"), dpi=150)
    plt.close()

    log("\n6. PERMUTATION IMPORTANCE (on chronological test set)")
    perm = permutation_importance(
        model_c, X_test_c, y_test_c,
        n_repeats=10, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error",
    )

    perm_df = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    perm_df.to_csv(os.path.join(REPORTS_DIR, f"{name.lower()}_permutation_importance.csv"), index=False)
    
    plt.figure(figsize=(8, 5))
    plt.barh(perm_df["feature"][::-1], perm_df["importance_mean"][::-1], xerr=perm_df["importance_std"][::-1], color="steelblue")
    plt.title(f"Phase III - Permutation Importance ({name})")
    plt.xlabel("Increase in MAE when feature is shuffled [C]")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{name.lower()}_permutation_importance.png"), dpi=150)
    plt.close()
    
    overall_summary["results"][name] = model_summary

with open(os.path.join(REPORTS_DIR, "phase3_evaluation_summary.json"), "w") as f:
    json.dump(overall_summary, f, indent=2, default=str)

with open(os.path.join(REPORTS_DIR, "phase3_evaluation_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log("\n" + "=" * 70)
log(f"Phase III evaluation complete. Artefacts -> {REPORTS_DIR}/")
log("=" * 70)