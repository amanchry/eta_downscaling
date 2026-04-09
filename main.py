"""
Full end-to-end pipeline for WaPOR v3 ETa downscaling.

Steps
-----
  0.  Load AOI from GeoJSON
  1.  Download WaPOR L1 AETI (300 m)          — FAO direct URL (GDAL vsicurl)
  2.  Download WaPOR L3 AETI (20 m)           — FAO direct URL (validation)
  3.  Download Landsat 8+9 index stack (30 m)  — GEE Python API
        6 bands: NDVI, EVI, SAVI, NDWI, NDMI, LST
        Cloud-masked (QA_PIXEL) + scale factors applied in GEE
  4.  Extract training data in memory
        Aggregate Landsat 30 m → 300 m (mean), align to WaPOR L1 grid,
        pair each pixel with its ETa value — no CSV written to disk
  5.  Preprocess: clean, print stats, correlation matrix
  6.  Train/test split (80/20) + StandardScaler
  7.  Train four models: Linear Regression, Random Forest, XGBoost, MLP
  8.  Compare models — metrics table + bar chart + scatter plots
  9.  Apply all models to 30 m raster → downscaled ETa GeoTIFFs
  10. Validate downscaled maps against WaPOR L3
"""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ── Download helpers ──────────────────────────────────────────
from helpers.ETa_wapor_v3_download import (
    download_wapor_v3_L1_eta_data,
    download_wapor_v3_L3_eta_data,
)
from helpers.Landsat_download import download_landsat_indices_30m
from helpers.training_samples import extract_training_data

# ── ML helpers ────────────────────────────────────────────────
import helpers.models as _models       # module-level import to set MONTH_LABEL
from helpers.models import (
    print_section,
    preprocess_training_data,
    split_and_scale,
    split_and_scale_spatial_multilevel,
    train_decision_tree,
    train_linear_regression,
    train_random_forest,  
    train_xgboost,        
    train_mlp,            
    compare_models,
    predict_raster,
    validate_against_l3,
    run_stability_analysis,
    plot_stability_distributions,
    visualize_spatial_split,
)


# =============================================================
# CONFIGURATION — edit only this block
# =============================================================

YEAR            = 2018
MONTH           = 10

AOI_GEOJSON        = Path("Mwea_Scheme.geojson")  # study area boundary
AOI_BUFFER_M       = 0                             # buffer around AOI in metres
STABILITY_TEST     = False                          # Run 100 iterations for spread analysis
USE_SPATIAL_SPLIT  = True                         # Set to True for Multi-level Spatial Split
N_ITERATIONS       = 100
WAPOR_L3_REGION    = "KMW"                         # WaPOR L3 region code for Mwea
MAX_CLOUD          = 5                             # max cloud cover % per Landsat image

INPUT_FOLDER    = Path("input_data")            # downloaded rasters
OUTPUT_FOLDER   = Path("output_data_10m")       # models, GeoTIFFs, figures

# =============================================================
# Derived paths 
# =============================================================

_TAG = f"{YEAR}_{MONTH:02d}"
_models.MONTH_LABEL = _TAG          # sets the suffix used in all output filenames

WAPOR_L1_PATH = INPUT_FOLDER / f"WAPOR3_L1_AETI_M_{YEAR}_{MONTH:02d}.tif"
WAPOR_L3_PATH = INPUT_FOLDER / f"WAPOR3_L3_AETI_M_{YEAR}_{MONTH:02d}.tif"
LANDSAT_PATH  = INPUT_FOLDER / f"landsat_indices_30m_{YEAR}-{MONTH:02d}.tif"


# =============================================================
# AOI UTILITIES
# =============================================================

def load_aoi(aoi_geojson: Path, buffer_m: float = 0) -> gpd.GeoDataFrame:
    """
    Load the study-area GeoJSON, dissolve to a single geometry, and
    optionally buffer by buffer_m metres.

    Parameters
    ----------
    aoi_geojson : Path   Path to the GeoJSON file
    buffer_m    : float  Buffer in metres (0 = no buffer)

    Returns
    -------
    gpd.GeoDataFrame  Single-row GeoDataFrame in EPSG:4326
    """
    if not aoi_geojson.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_geojson.resolve()}")

    gdf = gpd.read_file(aoi_geojson)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    aoi = gdf.dissolve().reset_index(drop=True)

    if aoi.crs is None:
        aoi = aoi.set_crs("EPSG:4326")

    aoi_4326 = aoi.to_crs("EPSG:4326")

    if buffer_m > 0:
        utm_crs = aoi_4326.estimate_utm_crs()
        aoi_utm = aoi_4326.to_crs(utm_crs)
        aoi_utm["geometry"] = aoi_utm.geometry.buffer(buffer_m).buffer(0)
        aoi_4326 = aoi_utm.to_crs("EPSG:4326")
        print(f"  Buffer applied : {buffer_m} m")

    print(f"  AOI bounds : {aoi_4326.total_bounds.round(4)}")
    return aoi_4326


def _aoi_geojson_dict(aoi_gdf: gpd.GeoDataFrame) -> dict:
    """Return AOI geometry as a GeoJSON dict for GDAL cutline."""
    return aoi_gdf.to_crs("EPSG:4326").geometry.iloc[0].__geo_interface__


# =============================================================
# MAIN PIPELINE
# =============================================================

def main() -> None:
    """
    Run all steps end-to-end.
    Training data is kept in memory.
    Edit the CONFIGURATION block above to change study area or period.
    """
    print("=" * 60)
    print("  WaPOR ETa Downscaling — Full End-to-End Pipeline")
    print(f"  Study area : {AOI_GEOJSON}")
    print(f"  Period     : {YEAR}-{MONTH:02d}")
    print("=" * 60)

    os.makedirs(INPUT_FOLDER,  exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ----------------------------------------------------------
    # Step 0 — Load AOI
    # ----------------------------------------------------------
    print_section("STEP 0 — Load AOI")
    aoi         = load_aoi(AOI_GEOJSON, buffer_m=AOI_BUFFER_M)
    aoi_geojson = _aoi_geojson_dict(aoi)
    print(f"  File : {AOI_GEOJSON}")

    # ----------------------------------------------------------
    # Step 1 — Download WaPOR L1 AETI (300 m)
    # ----------------------------------------------------------
    print_section("STEP 1 — Download WaPOR L1 AETI (300 m)")
    if WAPOR_L1_PATH.exists():
        print(f"  Already exists, skipping: {WAPOR_L1_PATH.name}")
    else:
        download_wapor_v3_L1_eta_data(
            year          =YEAR,
            month         =MONTH,
            output_folder =str(INPUT_FOLDER),
            geojson_obj   =aoi_geojson,
        )
    print(f"  L1 AETI : {WAPOR_L1_PATH}")

    # ----------------------------------------------------------
    # Step 2 — Download WaPOR L3 AETI (20 m) — for validation
    # ----------------------------------------------------------
    print_section("STEP 2 — Download WaPOR L3 AETI (20 m)")
    if WAPOR_L3_PATH.exists():
        print(f"  Already exists, skipping: {WAPOR_L3_PATH.name}")
    else:
        download_wapor_v3_L3_eta_data(
            year          =YEAR,
            month         =MONTH,
            output_folder =str(INPUT_FOLDER),
            geojson_obj   =aoi_geojson,
            region_id     =WAPOR_L3_REGION,
        )
    print(f"  L3 AETI : {WAPOR_L3_PATH}")

    # ----------------------------------------------------------
    # Step 3 — Download Landsat 8+9 index stack (30 m)
    #   Cloud-masked via QA_PIXEL, scale factors applied,
    #   6 index bands computed in GEE: NDVI EVI SAVI NDWI NDMI LST
    # ----------------------------------------------------------
    print_section("STEP 3 — Download Landsat 8+9 Index Stack (30 m)")
    if LANDSAT_PATH.exists():
        print(f"  Already exists, skipping: {LANDSAT_PATH.name}")
    else:
        download_landsat_indices_30m(
            aoi_gdf     =aoi,
            year        =YEAR,
            month       =MONTH,
            max_cloud   =MAX_CLOUD,
            output_file =LANDSAT_PATH,
        )
    print(f"  Landsat : {LANDSAT_PATH}")

    # ----------------------------------------------------------
    # Step 4 — Extract training data
    #   Aggregates Landsat 30 m → 300 m (mean), aligns to WaPOR
    #   L1 pixel grid, returns a DataFrame.
    # ----------------------------------------------------------
    print_section("STEP 4 — Extract Training Data (in memory)")
    df_raw = extract_training_data(
        landsat_indices_path =str(LANDSAT_PATH),
        wapor_l1_path        =str(WAPOR_L1_PATH),
    )
    # df_raw.to_csv("training_data_pixel.csv")

    # ----------------------------------------------------------
    # Step 5 — Preprocess: clean + print stats + correlation plot
    # ----------------------------------------------------------
    X, y, _ = preprocess_training_data(df_raw, out_dir=str(OUTPUT_FOLDER))

    # ----------------------------------------------------------
    # Step 6 — Train/test split and feature scaling
    # ----------------------------------------------------------
    if USE_SPATIAL_SPLIT:
        data = split_and_scale_spatial_multilevel(df_raw, seed=42)
        
        # ── Step 6.1 — Visualize the Split ────────────────
        print_section("STEP 6.1 — Visualize Current Spatial Split")
        split_map_path = os.path.join(str(OUTPUT_FOLDER), "spatial_split_visualization.png")
        visualize_spatial_split(
            data['vis_df'], 
            f"Multi-level Spatial Split (Seed 42)", 
            split_map_path
        )
    else:
        # Standard random split
        data = split_and_scale(X, y)
        
        # ── Step 6.1 — Visualize the Random Split ────────
        print_section("STEP 6.1 — Visualize Current Random Split")
        # We create a visualization dataframe for the random split
        vis_df_random = df_raw.copy()
        # Initialise with 'Test'
        vis_df_random['split'] = 'Test'
        
        # We need to map the training indices back. 
        # Note: data['X_train_raw'] matches the first 80% if shuffle=False, 
        # but since we use shuffle=True, we'll just show the distribution 
        # by recreating the mask with the same seed.
        from sklearn.model_selection import train_test_split
        train_idx, _ = train_test_split(df_raw.index, test_size=0.2, random_state=42, shuffle=True)
        vis_df_random.loc[train_idx, 'split'] = 'Train'
        
        split_map_path = os.path.join(str(OUTPUT_FOLDER), "spatial_split_visualization.png")
        visualize_spatial_split(
            vis_df_random, 
            "Random Pixel-wise Split (80/20)", 
            split_map_path
        )

    # ----------------------------------------------------------
    # Step 7 — Train & Stability Analysis
    # ----------------------------------------------------------
    if STABILITY_TEST:
        print_section(f"STEP 7 — Stability Analysis ({N_ITERATIONS} iterations)")
        
        # 1. Run stability tests (re-splitting in each iteration)
        split_type = 'spatial_multilevel' if USE_SPATIAL_SPLIT else 'random'
        
        lr_stability  = run_stability_analysis(train_linear_regression, df_raw, str(OUTPUT_FOLDER), N_ITERATIONS, split_type=split_type)
        dt_stability  = run_stability_analysis(train_decision_tree, df_raw, str(OUTPUT_FOLDER), N_ITERATIONS, split_type=split_type)
        rf_stability  = run_stability_analysis(train_random_forest, df_raw, str(OUTPUT_FOLDER), N_ITERATIONS, split_type=split_type)
        xgb_stability = run_stability_analysis(train_xgboost, df_raw, str(OUTPUT_FOLDER), N_ITERATIONS, split_type=split_type)
        mlp_stability = run_stability_analysis(train_mlp, df_raw, str(OUTPUT_FOLDER), N_ITERATIONS, split_type=split_type)
        
        # 2. Combine results and plot distributions
        all_stability_results = lr_stability + dt_stability + rf_stability + xgb_stability + mlp_stability
        plot_stability_distributions(all_stability_results, str(OUTPUT_FOLDER))
        
        # 3. Calculate Mean Metrics for final comparison (to show in bar chart)
        def get_mean_metrics(stability_list):
            df_s = pd.DataFrame(stability_list)
            return {
                "R2": df_s['R2'].mean(),
                "RMSE": df_s['RMSE'].mean(),
                "rRMSE_pct": df_s['rRMSE_pct'].mean(),
                "MAE": df_s['MAE'].mean(),
                "Bias": df_s['Bias'].mean()
            }
        
        # 4. Train one final set of models for the prediction step
        lr_result  = train_linear_regression(data, str(OUTPUT_FOLDER))
        dt_result  = train_decision_tree(data, str(OUTPUT_FOLDER))
        rf_result  = train_random_forest(data, str(OUTPUT_FOLDER))
        xgb_result = train_xgboost(data, str(OUTPUT_FOLDER))
        mlp_result = train_mlp(data, str(OUTPUT_FOLDER))
        
        # 5. OVERRIDE the single-run metrics with the 100-run AVERAGE
        lr_result['metrics']  = get_mean_metrics(lr_stability)
        dt_result['metrics']  = get_mean_metrics(dt_stability)
        rf_result['metrics']  = get_mean_metrics(rf_stability)
        xgb_result['metrics'] = get_mean_metrics(xgb_stability)
        mlp_result['metrics'] = get_mean_metrics(mlp_stability)
        
    else:
        lr_result  = train_linear_regression(data, str(OUTPUT_FOLDER))
        dt_result  = train_decision_tree(data, str(OUTPUT_FOLDER))
        rf_result  = train_random_forest(data, str(OUTPUT_FOLDER))
        xgb_result = train_xgboost(data, str(OUTPUT_FOLDER))
        mlp_result = train_mlp(data, str(OUTPUT_FOLDER))

    all_results = [lr_result, dt_result, rf_result, xgb_result, mlp_result]

    # ----------------------------------------------------------
    # Step 8 — Evaluate model
    # ----------------------------------------------------------
    best_name = compare_models(all_results, data, str(OUTPUT_FOLDER))

    # ----------------------------------------------------------
    # Step 9 — Apply models to 30 m raster → downscaled ETa maps
    # ----------------------------------------------------------
    predict_raster(
        raster_path  =str(LANDSAT_PATH),
        model_results=all_results,
        scaler       =data["scaler"],
        best_name    =best_name,
        out_dir      =str(OUTPUT_FOLDER),
        aoi_path     =str(AOI_GEOJSON),   # clip to original scheme boundary (no buffer)
    )

    # ----------------------------------------------------------
    # Step 10 — Validate against WaPOR L3
    # ----------------------------------------------------------
    if WAPOR_L3_PATH.exists():
        model_names = [r["name"] for r in all_results if r]
        validate_against_l3(str(WAPOR_L3_PATH), str(OUTPUT_FOLDER), model_names)
    else:
        print(f"\n[WARN] L3 raster not found: {WAPOR_L3_PATH} — skipping Step 10.")

    # ----------------------------------------------------------
    # Done
    # ----------------------------------------------------------
    print_section("Pipeline Complete")
    print(f"  Best model : {best_name}")
    print(f"  All outputs: {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
