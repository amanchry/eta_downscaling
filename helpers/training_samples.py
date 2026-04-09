"""
helpers/training_samples.py
============================
Build the training dataset in memory by aggregating the 30 m Landsat
index stack to 300 m and pairing each pixel with WaPOR L1 ETa.

The result is returned directly as a DataFrame ready for model training.
No CSV is written to disk — all processing happens in RAM.

WHY THIS STEP EXISTS
--------------------
We have two rasters at incompatible resolutions:
    • Landsat indices  →  30 m pixels  (our predictors X)
    • WaPOR L1 ETa     → 300 m pixels  (our target y)

A machine learning model needs X and y at the SAME spatial resolution
and perfectly pixel-aligned. We cannot pair a 30 m Landsat pixel with
a 300 m WaPOR pixel directly — they don't represent the same ground area.

Solution: aggregate Landsat 30 m → 300 m by averaging all ~100 small
pixels that fall inside each large WaPOR pixel. Now both X and y describe
the same 300 m × 300 m ground area, and we can build a row in the
training table: [NDVI_mean, EVI_mean, ..., LST_mean] → ETa.

The model learns at 300 m. We then APPLY it at 30 m (Step 9 in main.py)
because the 30 m predictors still exist at full resolution — we just
couldn't train on them directly without 30 m ETa labels.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import xy as transform_xy


FEATURE_NAMES = ["NDVI", "EVI", "SAVI", "NDWI", "NDMI", "LST"]
NODATA        = -9999


def extract_training_data(
    landsat_indices_path: str,
    wapor_l1_path: str,
) -> pd.DataFrame:
    """
    Aggregate the 30 m Landsat index stack to 300 m (mean), align every
    aggregated pixel to the WaPOR L1 pixel grid, and return a DataFrame
    of matched pixel pairs ready for ML training.

    HOW THE 30 m → 300 m AGGREGATION WORKS
    ----------------------------------------
    One 300 m WaPOR pixel covers a 300 × 300 m ground area.
    One 30 m Landsat pixel covers a  30 ×  30 m ground area.

    So each WaPOR pixel is covered by a 10 × 10 block of Landsat pixels
    (300 / 30 = 10 along each axis → 100 Landsat pixels per WaPOR pixel).

    We use rasterio's reproject() with Resampling.average to:
        1. Take each 30 m Landsat band as the source.
        2. Project it onto the exact WaPOR 300 m pixel grid (same CRS,
           same affine transform, same dimensions).
        3. For each destination 300 m pixel, compute the MEAN of all
           source 30 m pixels that map into it (NaN pixels are excluded
           from the average, so cloud gaps don't pull the mean down).

    The result is a 300 m grid of averaged Landsat indices that aligns
    perfectly — pixel for pixel — with the WaPOR ETa grid.


    Parameters
    ----------
    landsat_indices_path : str
        6-band 30 m GeoTIFF (NDVI, EVI, SAVI, NDWI, NDMI, LST).
    wapor_l1_path : str
        WaPOR v3 L1 AETI GeoTIFF at 300 m (the training target).

    Returns
    -------
    pd.DataFrame
        Columns : NDVI, EVI, SAVI, NDWI, NDMI, LST, ETa
        Each row : one 300 m pixel with averaged Landsat indices + ETa value
    """
    print("\n" + "=" * 60)
    print("  Building Training Data (30 m → 300 m aggregation)")
    print("=" * 60)
    print(f"  Landsat index stack : {landsat_indices_path}")
    print(f"  WaPOR L1 ETa        : {wapor_l1_path}")


    with rasterio.open(wapor_l1_path) as wapor_src:
        ref_crs       = wapor_src.crs        # e.g. EPSG:4326
        ref_transform = wapor_src.transform  # affine: origin + pixel size
        ref_height    = wapor_src.height     # number of rows
        ref_width     = wapor_src.width      # number of columns
        wapor_nodata  = wapor_src.nodata     # fill value (usually -9999)
        eta           = wapor_src.read(1).astype(np.float64)  # shape: (H, W)

    # Determine the nodata value to use when masking ETa
    nd_eta = wapor_nodata if wapor_nodata is not None else NODATA
    print(f"  WaPOR L1 grid : {ref_height} × {ref_width} pixels  |  CRS: {ref_crs}")

    with rasterio.open(landsat_indices_path) as ls_src:
        ls_data      = ls_src.read().astype(np.float64)  # shape: (6, H30, W30)
        ls_transform = ls_src.transform
        ls_crs       = ls_src.crs
        ls_nodata    = ls_src.nodata
        n_bands      = ls_src.count

    if n_bands < 6:
        raise ValueError(
            f"Expected 6 bands in Landsat index stack, found {n_bands}."
        )

    # ------------------------------------------------------------------
    # Replace Landsat nodata values with NaN BEFORE averaging.
    #
    # WHY: Cloud-masked pixels have a nodata fill value (e.g. -9999).
    # If we average -9999 along with real index values we get a garbage mean.
    # By converting to NaN first, numpy's averaging (and rasterio's
    # Resampling.average) automatically skips those pixels — so a 300 m
    # cell is the average of only the VALID 30 m pixels inside it.
    # If ALL 30 m pixels inside a 300 m cell are NaN (fully clouded block),
    # the destination 300 m pixel also becomes NaN → excluded later.
    # ------------------------------------------------------------------
    nd_ls = ls_nodata if ls_nodata is not None else NODATA
    ls_data[ls_data == nd_ls] = np.nan   # nodata → NaN

    print(f"  Landsat 30 m  : {ls_data.shape[1]} × {ls_data.shape[2]} pixels")
    print("  Aggregating 30 m → 300 m (mean) ...")

    # ==================================================================
    # THE ACTUAL RESAMPLING: 30 m → 300 m
    #
    # Allocate an empty output array the same shape as the WaPOR grid.
    # We fill it band-by-band using rasterio's reproject() function.
    #
    # Shape: (6 bands, ref_height rows, ref_width cols)
    # Pre-filled with NaN so any cell that gets no valid source pixels
    # stays NaN (not zero, which would be a false value).
    # ==================================================================
    landsat_300m = np.full((6, ref_height, ref_width), np.nan, dtype=np.float64)

    for i, name in enumerate(FEATURE_NAMES):
        # --------------------------------------------------------------
        # rasterio.warp.reproject() — the core of the aggregation
        #
        # This function reprojects AND resamples a source raster array
        # onto a destination grid. Here we use it purely for resampling
        # (source and destination CRS are the same or handled internally),
        # but its real power is that it geometrically maps source pixels
        # to destination pixels using the affine transforms.
        #
        #
        #   resampling    = Resampling.average
        #       HOW to combine multiple source pixels into one destination
        #       pixel. Options include nearest, bilinear, cubic, average...
        #
        #       We use AVERAGE because:
        #       • Each 300 m destination pixel covers ~100 source 30 m pixels.
        #       • We want the mean spectral signal over that 300 m ground area.
        #       • This matches the physical interpretation of WaPOR ETa,
        #         which represents the average ET over a 300 m × 300 m cell.
        #       • NaN pixels (cloud gaps) are automatically excluded from
        #         the average, so partial cloud cover doesn't bias the mean.
        #
        # --------------------------------------------------------------
        reproject(
            source        = ls_data[i],       # 30 m band array
            destination   = landsat_300m[i],  # 300 m output array (written in-place)
            src_transform = ls_transform,     # 30 m pixel geometry
            src_crs       = ls_crs,
            dst_transform = ref_transform,    # WaPOR 300 m pixel geometry (MASTER grid)
            dst_crs       = ref_crs,
            resampling    = Resampling.average,  # mean of all ~100 source pixels
            src_nodata    = np.nan,           # ignore NaN source pixels
            dst_nodata    = np.nan,           # NaN output if no valid source pixels
        )
        print(f"    {name}: {int(np.isfinite(landsat_300m[i]).sum())} valid pixels")

    # Visually, what happened after the loop:
    #
    #   BEFORE (Landsat 30 m, one band):
    #   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
    #   │  │  │  │  │  │  │  │  │  │  │  ← 10 columns of 30 m pixels
    #   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    #   │  │  │  │  │NaN│  │  │  │  │  │  ← NaN = cloud gap
    #   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    #   │  │  │  │  │  │  │  │  │  │  │
    #   ... (10 rows total for this one WaPOR pixel)
    #
    #   AFTER (aggregated to 300 m, one cell):
    #   ┌──────────────────────────────┐
    #   │   mean of 99 valid values    │  ← NaN excluded from average
    #   └──────────────────────────────┘
    #   This one 300 m value now aligns with the WaPOR ETa pixel.

    # ==================================================================
    # Build a valid-pixel mask
    #
    # Not every 300 m pixel can be used for training. We require:
    #
    #   ETa is finite        → not NaN (wasn't a nodata pixel in WaPOR)
    #   ETa != nd_eta        → not the fill value (-9999)
    #   ETa > 0              → physically ETa cannot be negative.
    #                          Zero or negative usually means water body,
    #                          urban surface, or a WaPOR data gap.
    #
    #   All 6 Landsat bands are finite → no cloud gap in any index.
    #                          If even one band is NaN (e.g. LST had cloud
    #                          while NDVI was clear), we drop that pixel
    #                          entirely — a partial feature vector is useless.
    #
    # valid is a boolean array, shape (ref_height, ref_width).
    # True = use this pixel for training. False = skip it.
    # ==================================================================
    valid = np.isfinite(eta) & (eta != nd_eta) & (eta > 0)
    for i in range(6):
        valid &= np.isfinite(landsat_300m[i])  # all 6 bands must be valid

    n_valid = int(valid.sum())
    print(f"  Valid pixel pairs : {n_valid}")

    if n_valid == 0:
        raise RuntimeError(
            "No valid pixel pairs found — check that rasters overlap "
            "and share a compatible CRS."
        )

    # ==================================================================
    # Extract valid pixels → DataFrame (stays in memory)
    #
    # valid is a 2D boolean mask. Using it as an index on a 2D array
    # flattens and filters in one step:
    #
    #   landsat_300m[i][valid]  →  1D array of length n_valid
    #   eta[valid]              →  1D array of length n_valid
    #
    # We build a dict of these arrays and wrap in a DataFrame.
    # Each ROW in the DataFrame is one training sample:
    #   [NDVI_300m, EVI_300m, SAVI_300m, NDWI_300m, NDMI_300m, LST_300m, ETa_300m]
    #
    # This is the (X, y) dataset the model will learn from.
    # ==================================================================
    # Pixel-center coordinates on the WaPOR grid (used for spatial splitting/plots)
    # Note: these are in the WaPOR CRS units (e.g., lon/lat if EPSG:4326).
    rows, cols = np.where(valid)
    xs, ys = transform_xy(ref_transform, rows, cols, offset="center")

    data = {name: landsat_300m[i][valid] for i, name in enumerate(FEATURE_NAMES)}
    data["ETa"] = eta[valid]
    data["x"] = np.asarray(xs, dtype=np.float64)
    data["y"] = np.asarray(ys, dtype=np.float64)

    df = pd.DataFrame(data)
    print(f"  Dataset shape : {df.shape}")
    return df
