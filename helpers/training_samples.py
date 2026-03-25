"""
helpers/training_samples.py
============================
Build the training dataset in memory by aggregating the 30 m Landsat
index stack to 300 m and pairing each pixel with WaPOR L1 ETa.

No intermediate CSV is written — the result is returned directly as
numpy arrays ready for model training.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling


FEATURE_NAMES = ["NDVI", "EVI", "SAVI", "NDWI", "NDMI", "LST"]
NODATA        = -9999


def extract_training_data(
    landsat_indices_path: str,
    wapor_l1_path: str,
) -> pd.DataFrame:
    """
    Aggregate the 30 m Landsat index stack to 300 m (mean), align to the
    WaPOR L1 pixel grid, and return a DataFrame of pixel-pair samples.

    No file is written to disk — all processing happens in memory.

    Process
    -------
    1. Open WaPOR L1 (300 m) — defines the reference CRS, transform,
       and spatial extent that all data must align to.
    2. Reproject each of the 6 Landsat index bands from 30 m to the
       WaPOR 300 m grid using mean aggregation (Resampling.average).
    3. Build a valid-pixel mask: ETa and all 6 bands must be finite,
       non-NoData, and ETa must be positive.
    4. Return a DataFrame with columns: NDVI, EVI, SAVI, NDWI, NDMI, LST, ETa.
       Each row is one 300 m pixel pair.

    Parameters
    ----------
    landsat_indices_path : str
        6-band 30 m GeoTIFF (NDVI, EVI, SAVI, NDWI, NDMI, LST)
        produced by download_landsat_indices_30m().
    wapor_l1_path : str
        WaPOR v3 L1 AETI GeoTIFF at 300 m.

    Returns
    -------
    pd.DataFrame
        Columns: NDVI, EVI, SAVI, NDWI, NDMI, LST, ETa
        Rows   : one valid 300 m pixel per row
    """
    print("\n" + "=" * 60)
    print("  Building Training Data (30 m → 300 m aggregation)")
    print("=" * 60)
    print(f"  Landsat index stack : {landsat_indices_path}")
    print(f"  WaPOR L1 ETa        : {wapor_l1_path}")

    # ------------------------------------------------------------------
    # 1. Read WaPOR L1 — defines the 300 m reference grid
    # ------------------------------------------------------------------
    with rasterio.open(wapor_l1_path) as wapor_src:
        ref_crs       = wapor_src.crs
        ref_transform = wapor_src.transform
        ref_height    = wapor_src.height
        ref_width     = wapor_src.width
        wapor_nodata  = wapor_src.nodata
        eta           = wapor_src.read(1).astype(np.float64)

    nd_eta = wapor_nodata if wapor_nodata is not None else NODATA
    print(f"  WaPOR L1 grid : {ref_height} × {ref_width} pixels  |  CRS: {ref_crs}")

    # ------------------------------------------------------------------
    # 2. Read Landsat 30 m and reproject each band to 300 m (mean)
    # ------------------------------------------------------------------
    with rasterio.open(landsat_indices_path) as ls_src:
        ls_data      = ls_src.read().astype(np.float64)   # (6, H, W)
        ls_transform = ls_src.transform
        ls_crs       = ls_src.crs
        ls_nodata    = ls_src.nodata
        n_bands      = ls_src.count

    if n_bands < 6:
        raise ValueError(
            f"Expected 6 bands in Landsat index stack, found {n_bands}."
        )

    nd_ls = ls_nodata if ls_nodata is not None else NODATA
    ls_data[ls_data == nd_ls] = np.nan   # NoData → NaN so average ignores them

    print(f"  Landsat 30 m  : {ls_data.shape[1]} × {ls_data.shape[2]} pixels")
    print("  Aggregating 30 m → 300 m (mean) ...")

    landsat_300m = np.full((6, ref_height, ref_width), np.nan, dtype=np.float64)

    for i, name in enumerate(FEATURE_NAMES):
        reproject(
            source        = ls_data[i],
            destination   = landsat_300m[i],
            src_transform = ls_transform,
            src_crs       = ls_crs,
            dst_transform = ref_transform,
            dst_crs       = ref_crs,
            resampling    = Resampling.average,
            src_nodata    = np.nan,
            dst_nodata    = np.nan,
        )
        print(f"    {name}: {int(np.isfinite(landsat_300m[i]).sum())} valid pixels")

    # ------------------------------------------------------------------
    # 3. Build valid-pixel mask
    # ------------------------------------------------------------------
    valid = np.isfinite(eta) & (eta != nd_eta) & (eta > 0)
    for i in range(6):
        valid &= np.isfinite(landsat_300m[i])

    n_valid = int(valid.sum())
    print(f"  Valid pixel pairs : {n_valid}")

    if n_valid == 0:
        raise RuntimeError(
            "No valid pixel pairs found — check that rasters overlap "
            "and share a compatible CRS."
        )

    # ------------------------------------------------------------------
    # 4. Extract valid pixels → DataFrame (stays in memory)
    # ------------------------------------------------------------------
    data = {name: landsat_300m[i][valid] for i, name in enumerate(FEATURE_NAMES)}
    data["ETa"] = eta[valid]

    df = pd.DataFrame(data)
    print(f"  Dataset shape : {df.shape}")
    return df
