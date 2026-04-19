import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. PROJECT PATHS
# =========================================================
# This script is expected to be inside: project_root/notebooks/
project_root = Path(__file__).resolve().parent.parent
data_sources = project_root / "data"

reports_dir = project_root / "reports"
tables_dir = reports_dir / "tables"

tables_dir.mkdir(parents=True, exist_ok=True)

print("=== PROJECT PATHS ===")
print("project_root:", project_root)
print("data_sources:", data_sources)
print("tables_dir  :", tables_dir)

# Check required data file
tracks_path = data_sources / "spotify_tracks.csv"

if not tracks_path.exists():
    raise FileNotFoundError(
        f"Required file not found: {tracks_path}\n"
        f"Make sure this file exists inside the cloned project's 'data' folder."
    )

# =========================================================
# 2. LOAD DATA
# =========================================================
tracks = pd.read_csv(tracks_path)

# Clean column names
tracks.columns = [c.strip().lower() for c in tracks.columns]
unnamed_cols = [c for c in tracks.columns if c.startswith("unnamed")]
if unnamed_cols:
    tracks = tracks.drop(columns=unnamed_cols)

# =========================================================
# 3. SELECT VARIABLES FOR RQ2
# =========================================================
rq2_vars = [
    "popularity",
    "danceability",
    "energy",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms"
]

df = tracks[rq2_vars].copy().dropna()

print("\n=== RQ2 DATASET INFO ===")
print("Shape:", df.shape)

# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================
X = df.drop(columns=["popularity"])
y = df["popularity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Train size:", len(X_train))
print("Test size :", len(X_test))

# =========================================================
# 5. NAIVE BASELINE
# =========================================================
baseline_value = y_train.mean()
y_pred_baseline = np.full(shape=len(y_test), fill_value=baseline_value)

print("\nBaseline mean popularity:", round(baseline_value, 3))

# =========================================================
# 6. PERFORMANCE METRICS
# =========================================================
mae = mean_absolute_error(y_test, y_pred_baseline)
rmse = mean_squared_error(y_test, y_pred_baseline, squared=False)
r2 = r2_score(y_test, y_pred_baseline)

print("\n=== NAIVE BASELINE RESULTS ===")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R^2 :", round(r2, 3))

# =========================================================
# 7. SAVE RESULTS
# =========================================================
results_df = pd.DataFrame({
    "metric": ["MAE", "RMSE", "R2"],
    "naive_baseline_score": [mae, rmse, r2]
})

results_df.to_csv(tables_dir / "step4_naive_baseline_results.csv", index=False)

summary_df = pd.DataFrame({
    "item": ["dataset_rows", "train_size", "test_size", "baseline_mean_popularity"],
    "value": [len(df), len(X_train), len(X_test), baseline_value]
})

summary_df.to_csv(tables_dir / "step4_baseline_summary.csv", index=False)

print("\n=== BASELINE STAGE COMPLETED ===")
print("Saved files:")
print(tables_dir / "step4_naive_baseline_results.csv")
print(tables_dir / "step4_baseline_summary.csv")