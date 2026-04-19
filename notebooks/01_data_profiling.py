import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 1. PROJECT PATHS
# =========================================================
# This script is expected to be inside: project_root/notebooks/
project_root = Path(__file__).resolve().parent.parent
data_sources = project_root / "data"

reports_dir = project_root / "reports"
figures_dir = reports_dir / "figures"
tables_dir = reports_dir / "tables"

figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

print("=== PROJECT PATHS ===")
print("project_root:", project_root)
print("data_sources:", data_sources)
print("figures_dir :", figures_dir)
print("tables_dir  :", tables_dir)

# Check required data files before loading
tracks_path = data_sources / "spotify_tracks.csv"
albums_path = data_sources / "spotify_albums.csv"
artists_path = data_sources / "spotify_artists.csv"

for path in [tracks_path, albums_path, artists_path]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Make sure the file exists inside the 'data' folder of the cloned project."
        )

# =========================================================
# 2. DATA LOADING
# =========================================================
tracks = pd.read_csv(tracks_path)
albums = pd.read_csv(albums_path)
artists = pd.read_csv(artists_path)

print("\n=== DATASET SHAPES ===")
print("tracks:", tracks.shape)
print("albums:", albums.shape)
print("artists:", artists.shape)

# =========================================================
# 3. COLUMN STANDARDIZATION / CLEANING
# =========================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    unnamed_cols = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df

tracks = clean_columns(tracks)
albums = clean_columns(albums)
artists = clean_columns(artists)

print("\n=== CLEANED COLUMN NAMES ===")
print("tracks:", tracks.columns.tolist())

# =========================================================
# 4. SELECTED VARIABLES
# =========================================================
# RQ1
rq1_vars = ["popularity", "energy", "danceability"]

# RQ2
rq2_vars = [
    "popularity",
    "danceability",
    "energy",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms"
]

# Main selected dataframe
analysis_df = tracks[["id"] + rq2_vars].copy()

# =========================================================
# 5. INITIAL INSPECTION
# =========================================================
print("\n=== DTYPES: ANALYSIS DF ===")
print(analysis_df.dtypes)

print("\n=== HEAD: ANALYSIS DF ===")
print(analysis_df.head(3))

# =========================================================
# 6. KEY CHECKS / DUPLICATES
# =========================================================
def check_key(df: pd.DataFrame, col: str, name: str) -> dict:
    result = {
        "dataset": name,
        "key_column": col,
        "rows": len(df),
        "missing": int(df[col].isna().sum()),
        "unique": int(df[col].nunique()),
        "is_unique": bool(df[col].is_unique),
        "duplicate_keys": int(df[col].duplicated().sum())
    }
    print(f"\n{name} - {col}")
    for k, v in result.items():
        if k not in ["dataset", "key_column"]:
            print(f"{k}: {v}")
    return result

key_results = []
key_results.append(check_key(tracks, "id", "tracks"))
key_results.append(check_key(albums, "track_id", "albums"))
key_results.append(check_key(artists, "track_id", "artists"))

key_df = pd.DataFrame(key_results)
key_df.to_csv(tables_dir / "key_checks.csv", index=False)

duplicate_results = pd.DataFrame({
    "dataset": ["tracks", "albums", "artists"],
    "duplicate_rows": [
        tracks.duplicated().sum(),
        albums.duplicated().sum(),
        artists.duplicated().sum()
    ]
})
print("\n=== DUPLICATE ROWS ===")
print(duplicate_results)
duplicate_results.to_csv(tables_dir / "duplicate_rows.csv", index=False)

# =========================================================
# 7. MISSING VALUE CHECKS
# =========================================================
missing_df = analysis_df[rq2_vars].isna().sum().reset_index()
missing_df.columns = ["variable", "missing_count"]

print("\n=== MISSING VALUES: SELECTED VARIABLES ===")
print(missing_df)

missing_df.to_csv(tables_dir / "selected_vars_missing_values.csv", index=False)

# =========================================================
# 8. DESCRIPTIVE STATISTICS
# =========================================================
desc_rq1 = analysis_df[rq1_vars].describe().T
desc_rq1 = desc_rq1[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]

desc_rq2 = analysis_df[rq2_vars].describe().T
desc_rq2 = desc_rq2[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]

print("\n=== RQ1 DESCRIPTIVE STATISTICS ===")
print(desc_rq1)

print("\n=== RQ2 DESCRIPTIVE STATISTICS ===")
print(desc_rq2)

desc_rq1.to_csv(tables_dir / "desc_rq1.csv")
desc_rq2.to_csv(tables_dir / "desc_rq2.csv")

# =========================================================
# 9. REPORT TABLE
# =========================================================
report_vars = [
    "popularity",
    "energy",
    "danceability",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms"
]

final_summary = analysis_df[report_vars].agg(["mean", "std", "min", "median", "max"]).T.round(3)

print("\n=== FINAL SUMMARY TABLE FOR REPORT ===")
print(final_summary)

final_summary.to_csv(tables_dir / "step3_summary_table.csv")

# =========================================================
# 10. DISTRIBUTION PLOTS
# =========================================================
plot_vars = ["popularity", "energy", "danceability"]

for col in plot_vars:
    plt.figure(figsize=(6, 4))
    plt.hist(analysis_df[col].dropna(), bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{col}_histogram.png", dpi=300)
    plt.close()

for col in plot_vars:
    plt.figure(figsize=(5, 4))
    plt.boxplot(analysis_df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{col}_boxplot.png", dpi=300)
    plt.close()

print("\n=== HISTOGRAMS AND BOXPLOTS SAVED ===")

# =========================================================
# 11. CORRELATION ANALYSIS
# =========================================================
corr_df = analysis_df[rq2_vars].corr().round(3)

print("\n=== CORRELATION MATRIX ===")
print(corr_df)

print("\n=== POPULARITY ROW ===")
print(corr_df.loc["popularity"])

corr_df.to_csv(tables_dir / "correlation_matrix.csv")

# =========================================================
# 12. SKEWNESS ANALYSIS
# =========================================================
skew_df = analysis_df[rq2_vars].skew(numeric_only=True).sort_values(
    key=lambda s: s.abs(), ascending=False
).round(3)

print("\n=== SKEWNESS: SELECTED VARIABLES ===")
print(skew_df)

skew_df.to_csv(tables_dir / "skewness_selected_vars.csv")

# =========================================================
# 13. IQR-BASED OUTLIER ANALYSIS
# =========================================================
def iqr_outlier_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    results = []
    for col in cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()

        results.append({
            "variable": col,
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(iqr, 3),
            "lower_bound": round(lower, 3),
            "upper_bound": round(upper, 3),
            "outlier_count": int(outliers),
            "outlier_pct": round(outliers / len(series) * 100, 2)
        })
    return pd.DataFrame(results)

iqr_df = iqr_outlier_summary(
    analysis_df,
    ["popularity", "energy", "danceability", "speechiness", "tempo", "valence", "duration_ms"]
)

print("\n=== IQR OUTLIER SUMMARY ===")
print(iqr_df)

iqr_df.to_csv(tables_dir / "iqr_outliers_selected_vars.csv", index=False)

# =========================================================
# 14. SAVE MAIN ANALYSIS DATAFRAME
# =========================================================
analysis_df.to_csv(reports_dir / "analysis_df.csv", index=False)

print("\n=== STEP 3 PIPELINE COMPLETED ===")
print("Outputs saved in:")
print("figures ->", figures_dir)
print("tables  ->", tables_dir)
print("dataframe->", reports_dir / "analysis_df.csv")