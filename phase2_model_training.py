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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
np.random.seed(42)

DATA_PATH = "kosovo_weather_dataset.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("PHASE II  -  Model Training  (Random Forest Regressor)")
log("=" * 70)

df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.dropna(subset=["temperature", "humidity", "pressure"]).reset_index(drop=True)
df["pop"] = df["pop"].fillna(0.0)

df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24.0)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24.0)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

FEATURES = [
    "humidity", "pressure", "wind_speed", "wind_deg",
    "clouds", "visibility", "pop",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
TARGET = "temperature"

log(f"Rows after cleaning : {len(df)}")
log(f"Features (11)       : {FEATURES}")
log(f"Target              : {TARGET} (deg C)")

corr_cols = ["temperature", "humidity", "pressure", "wind_speed",
             "clouds", "visibility", "pop"]
corr = df[corr_cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Phase II - Correlation heat-map")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "phase2_correlation_heatmap.png"), dpi=150)
plt.close()

log("\nStrongest absolute correlations with temperature:")
log(corr["temperature"].drop("temperature").abs().sort_values(ascending=False).to_string())


X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_phase2.pkl"))

log(f"\nTrain / Test split  : {len(X_train)} / {len(X_test)}  (80% / 20%)")


cfg = dict(n_estimators=100, max_depth=None, min_samples_leaf=1,
           random_state=42, n_jobs=-1)

log("\n" + "-" * 70)
log("TRAINING")
log("-" * 70)
log(f"Config : {cfg}")

model = RandomForestRegressor(**cfg)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

mae   = mean_absolute_error(y_test,  y_pred_test)
rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_te = r2_score(y_test,  y_pred_test)
r2_tr = r2_score(y_train, y_pred_train)

log("\nTraining results:")
log(f"  MAE         : {mae:.3f} C")
log(f"  RMSE        : {rmse:.3f} C")
log(f"  R^2 (train) : {r2_tr:.4f}")
log(f"  R^2 (test)  : {r2_te:.4f}")


joblib.dump(model, os.path.join(MODELS_DIR, "rf_model.pkl"))


imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(7, 5))
imp.plot(kind="barh", color="steelblue")
plt.title("Phase II - Random Forest feature importance")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "phase2_feature_importance.png"), dpi=150)
plt.close()

log("\nFeature importance:")
log(imp.sort_values(ascending=False).to_string())

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.55, s=18, color="steelblue",
            label="predictions")
lo, hi = y_test.min(), y_test.max()
plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
plt.xlabel("Actual temperature (C)")
plt.ylabel("Predicted temperature (C)")
plt.title("Phase II - Predicted vs Actual temperature")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "phase2_pred_vs_true.png"), dpi=150)
plt.close()

summary = {
    "phase": "II - Model Training",
    "algorithm": "RandomForestRegressor",
    "rows_used": int(len(df)),
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
    "features":   FEATURES,
    "target":     TARGET,
    "config":     cfg,
    "metrics":    {"MAE": mae, "RMSE": rmse,
                   "R2_train": r2_tr, "R2_test": r2_te},
    "feature_importance": imp.sort_values(ascending=False).to_dict(),
}
with open(os.path.join(REPORTS_DIR, "phase2_training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(REPORTS_DIR, "phase2_training_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log("\n" + "=" * 70)
log("Phase II complete.  Models -> ./models/   Reports -> ./reports/")
log("Next step (Phase III): analysis, evaluation, and re-training.")
log("=" * 70)
