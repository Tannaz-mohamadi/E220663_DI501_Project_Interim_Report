# Spotify-Based Song Popularity Study

## About the Project 
Github Link: https://github.com/Tannaz-mohamadi/E220663_DI501_Project_Interim_Report/settings
This repository contains the interim work for a DI501 project on song popularity. The analysis is based on Spotify track-level data from the SpotGenTrack dataset.

The study has two connected goals. First, it looks at whether popularity changes across songs with different energy and danceability profiles. Second, it uses a small set of Spotify audio variables to predict popularity in a regression setting.

## Files Included

- `spotify_tracks.csv`  
  No adjustments are made on this file. Due to the size I could not upload it to github, so please copy paste the original file to the data folder. Main dataset      used in the analysis. It includes popularity and Spotify audio features.

- `spotify_albums.csv`  
  Used for dataset inspection and key checks.

- `spotify_artists.csv`  
  Used for dataset inspection, duplicate checks, and key structure review.

- `01_data_profiling.ipynb`  
  Covers data inspection, variable selection, descriptive statistics, missing-value checks, visual analysis, correlation analysis, skewness, and outlier review.

- `02_naive_baseline.ipynb`  
  Builds and evaluates a simple mean-based baseline for the popularity prediction task.

## Main Steps

### 1. Initial Inspection
The datasets are loaded and checked in terms of structure, column names, and identifier fields.

### 2. Data Preparation
The analysis keeps only the variables needed for the research questions. Column names are cleaned, unnamed columns are removed, and duplicates and missing values are reviewed.

### 3. Exploratory Analysis
The selected variables are summarized with descriptive statistics. Their distributions are examined through plots, and additional checks are carried out for correlation, skewness, and potential outliers.

### 4. Baseline Modeling
A simple regression baseline is defined by predicting the training-set mean popularity for every test observation. Its performance is evaluated before moving to more advanced models.

## Selected Variables

### For group-based analysis
- `popularity`
- `energy`
- `danceability`

### For prediction
- `popularity`
- `danceability`
- `energy`
- `speechiness`
- `tempo`
- `valence`
- `duration_ms`

## Evaluation Measures
The baseline model is assessed with:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²

These measures are appropriate because popularity is treated as a continuous variable.

## Project Choices
- The interim study focuses on Spotify audio variables rather than lyric-based features.
- Outliers are not removed automatically, since they may represent real variation among tracks.
- A simple mean-based baseline is used as the reference point for later regression models.

## Next Stage
The next phase of the project will:
- test popularity differences across audio-profile groups more formally,
- train additional regression models,
- compare them with the naïve baseline,
- and interpret which audio features contribute most to prediction.
