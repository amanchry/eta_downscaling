"""
ml_pipeline.py
==============
Full end-to-end pipeline for WaPOR ETa downscaling.

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

Run:
    python ml_pipeline.py
"""

import os
from pathlib import Path

import geopandas as gpd

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
    train_linear_regression,
    train_random_forest,  
    train_xgboost,        
    train_mlp,            
    compare_models,
    predict_raster,
    validate_against_l3,
)


# =============================================================
# CONFIGURATION — edit only this block
# =============================================================

YEAR            = 2018
MONTH           = 10

AOI_GEOJSON     = Path("Mwea_Scheme.geojson")  # study area boundary
AOI_BUFFER_M    = 0                             # buffer around AOI in metres
WAPOR_L3_REGION = "KMW"                         # WaPOR L3 region code for Mwea
MAX_CLOUD       = 5                             # max cloud cover % per Landsat image

INPUT_FOLDER    = Path("input_data")            # downloaded rasters
OUTPUT_FOLDER   = Path("output_data_10m")       # models, GeoTIFFs, figures

# =============================================================
# Derived paths — do not edit
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
    Run all pipeline steps end-to-end.
    Training data is kept in memory — no intermediate CSV is written.
    Edit the CONFIGURATION block above to change study area or period.
    """
    print("=" * 60)
    print("  WaPOR ETa Downscaling — Full End-to-End Pipeline")
    print(f"  Study area : Mwea Irrigation Scheme, Kenya")
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
    # Step 2 — Download WaPOR L3 AETI (20 m) — validation only
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
    # Step 4 — Extract training data in memory
    #   Aggregates Landsat 30 m → 300 m (mean), aligns to WaPOR
    #   L1 pixel grid, returns a DataFrame — no CSV written.
    # ----------------------------------------------------------
    print_section("STEP 4 — Extract Training Data (in memory)")
    df_raw = extract_training_data(
        landsat_indices_path =str(LANDSAT_PATH),
        wapor_l1_path        =str(WAPOR_L1_PATH),
    )

    # ----------------------------------------------------------
    # Step 5 — Preprocess: clean + print stats + correlation plot
    # ----------------------------------------------------------
    X, y, _ = preprocess_training_data(df_raw, out_dir=str(OUTPUT_FOLDER))

    # ----------------------------------------------------------
    # Step 6 — Train/test split and feature scaling
    # ----------------------------------------------------------
    data = split_and_scale(X, y)

    # ----------------------------------------------------------
    # Step 7 — Train model
    #   Currently: Linear Regression only (baseline).
    #   To add more models later, uncomment the other lines and
    #   add them to all_results.
    # ----------------------------------------------------------
    lr_result = train_linear_regression(data, str(OUTPUT_FOLDER))

    rf_result  = train_random_forest(data, str(OUTPUT_FOLDER))
    xgb_result = train_xgboost(data, str(OUTPUT_FOLDER))
    # mlp_result = train_mlp(data, str(OUTPUT_FOLDER))

    all_results = [lr_result,rf_result,xgb_result]
    # all_results = [lr_result, rf_result, xgb_result, mlp_result]

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
