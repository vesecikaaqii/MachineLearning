"""
Phase II - Model Training
Trains an LSTM (Long Short-Term Memory) network for next-step temperature
forecasting on the Kosovo weather dataset.

Why LSTM?
  - The project's core task is *weather forecasting*, which is a time-series problem.
  - Weather state at time t is strongly correlated with state at t-1, t-2, ... (autocorrelation).
  - LSTM gates (forget / input / output) are designed exactly to learn long- and
    short-term temporal dependencies in sequences.
  - Outperforms classical regression on the same data because it consumes the
    whole 18h history window, not isolated snapshots.

Outputs:
  - models/lstm_model.pt        - trained PyTorch state_dict
  - models/scaler_lstm.pkl      - StandardScaler fitted on the 5 input features
  - models/metrics.txt          - full training log (metrics, shapes, epochs)
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "kosovo_weather_dataset.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

metrics_log = []


def log(msg):
    print(msg)
    metrics_log.append(str(msg))


# ---------------------------------------------------------------------------
# 1. Load & preprocess
# ---------------------------------------------------------------------------
log("=" * 60)
log("LOADING DATASET")
log("=" * 60)

df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.dropna(subset=["temperature"]).reset_index(drop=True)
df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

log(f"Rows: {len(df)}  |  Columns: {df.shape[1]}")
log(f"Cities: {df['city'].nunique()}")


# ---------------------------------------------------------------------------
# 2. Build sequences (per city, no cross-city windows)
# ---------------------------------------------------------------------------
log("\n" + "=" * 60)
log("BUILDING SEQUENCES")
log("=" * 60)

seq_features = ["temperature", "humidity", "pressure", "wind_speed", "clouds"]
SEQ_LEN = 6  # 6 steps x 3h = 18h history -> predict next 3h step

forecast_df = df[df["type"] == "forecast"].copy()

scaler_lstm = StandardScaler()
scaler_lstm.fit(forecast_df[seq_features].values)

X_seq, y_seq = [], []
for _, group in forecast_df.groupby("city"):
    group = group.sort_values("datetime")
    vals = scaler_lstm.transform(group[seq_features].values)
    for i in range(len(vals) - SEQ_LEN):
        X_seq.append(vals[i : i + SEQ_LEN])
        y_seq.append(vals[i + SEQ_LEN, 0])  # next-step (scaled) temperature

X_seq = np.asarray(X_seq, dtype=np.float32)
y_seq = np.asarray(y_seq, dtype=np.float32)
log(f"Total sequences: {len(X_seq)}  |  Shape: {X_seq.shape}")

# Temporal 80/20 split (no shuffling - prevents future leakage)
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
log(f"Train: {len(X_train)}  |  Test: {len(X_test)}")


# ---------------------------------------------------------------------------
# 3. LSTM model
# ---------------------------------------------------------------------------
log("\n" + "=" * 60)
log("LSTM  ->  next-step temperature (3h ahead)")
log("=" * 60)

torch.manual_seed(42)
np.random.seed(42)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


device = torch.device("cpu")
model = LSTMRegressor(n_features=len(seq_features)).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

log(f"Architecture: LSTM(64) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1)")
log(f"Optimizer: Adam(lr=1e-3)  |  Loss: MSE")

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)

# Hold out last 10% of train for validation / early stopping
val_split = int(len(X_train_t) * 0.9)
X_tr, X_val = X_train_t[:val_split], X_train_t[val_split:]
y_tr, y_val = y_train_t[:val_split], y_train_t[val_split:]

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

best_val = float("inf")
best_state = None
patience, bad = 10, 0
for epoch in range(60):
    model.train()
    for xb, yb in train_loader:
        optim.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()


# ---------------------------------------------------------------------------
# 4. Evaluate (in original Celsius units)
# ---------------------------------------------------------------------------
temp_mean = scaler_lstm.mean_[0]
temp_scale = scaler_lstm.scale_[0]
y_pred_real = y_pred_scaled * temp_scale + temp_mean
y_test_real = y_test * temp_scale + temp_mean

mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test_real, y_pred_real)

log(f"Epochs trained (early-stopped): {epoch + 1}")
log(f"Best validation loss: {best_val:.4f}")
log(f"\nTest metrics (Celsius):")
log(f"  MAE  = {mae:.3f} C")
log(f"  RMSE = {rmse:.3f} C")
log(f"  R2   = {r2:.3f}")


# ---------------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_model.pt")
joblib.dump(scaler_lstm, f"{MODELS_DIR}/scaler_lstm.pkl")

with open(f"{MODELS_DIR}/metrics.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(metrics_log))

log("\n" + "=" * 60)
log("LSTM TRAINED & SAVED IN ./models/")
log("=" * 60)
