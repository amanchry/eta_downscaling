"""
helpers/Landsat_download.py
============================
Downloads Landsat 8 and 9 imagery from Google Earth Engine (GEE),
applies cloud masking and radiometric scaling per-image, computes
6 spectral/thermal indices, reduces to a monthly median composite,
and exports the result as a 6-band GeoTIFF at 30 m resolution.

Why Landsat?
    WaPOR L1 ETa is at 300 m. To predict ETa at 30 m we need 30 m
    predictor variables. Landsat 8/9 is the only freely available
    satellite with both 30 m optical bands AND a thermal (LST) band
    — LST is directly tied to the energy balance that drives ET.

Output band order: NDVI | EVI | SAVI | NDWI | NDMI | LST
"""

import ee
import geemap
import geopandas as gpd
import os
from pathlib import Path
import calendar
from .gee_auth import ee_authenticate


# =============================================================
# FUNCTION 1 — Cloud Masking
# =============================================================

def _mask_cloud(image: ee.Image) -> ee.Image:
    """
    Remove cloud, cloud shadow, dilated cloud, and snow pixels from a
    single Landsat image using the QA_PIXEL quality-flag band.

    WHY WE DO THIS
    --------------
    Clouds are white and bright — completely wrong spectral signal.
    A cloud pixel will give NDVI ≈ -0.3 and LST ≈ 0°C even over a
    green irrigated field. Including those pixels in training data or
    the final prediction map would corrupt every index and every model.

    HOW QA_PIXEL WORKS
    ------------------
    Every Landsat C2 L2 image includes a band called QA_PIXEL.
    It is NOT reflectance — it is a 16-bit integer per pixel where
    individual BITS encode yes/no quality flags:

        Bit 1 → Dilated cloud  (pixels adjacent to a cloud — still hazy)
        Bit 3 → Cloud          (actual cloud pixel)
        Bit 4 → Cloud shadow   (dark stripe cast on the ground by a cloud)
        Bit 5 → Snow / ice     (bright like cloud)

    We keep a pixel only if ALL FOUR flags are OFF (0).

    Parameters
    ----------
    image : ee.Image
        Raw Landsat C2 L2 image straight from the ImageCollection.

    Returns
    -------
    ee.Image
        Same image with flagged pixels set to transparent (masked).
        Masked pixels are automatically ignored by .median() later.
    """
    # Pull only the quality flag band — it holds all the bitmask information
    qa = image.select("QA_PIXEL")

    # Build the keep-mask: True = clean pixel, False = bad pixel
    #
    # How bitwiseAnd works:
    #   "1 << 1" = binary 000010 = decimal 2  → isolates bit 1
    #   "1 << 3" = binary 001000 = decimal 8  → isolates bit 3
    #   .eq(0)   = that bit must be OFF (no cloud / shadow / snow)
    #
    # All four conditions are ANDed: a pixel passes only if every flag is clear.
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)          # bit 1 OFF → no dilated cloud
          .And(qa.bitwiseAnd(1 << 3).eq(0))  # bit 3 OFF → no cloud
          .And(qa.bitwiseAnd(1 << 4).eq(0))  # bit 4 OFF → no cloud shadow
          .And(qa.bitwiseAnd(1 << 5).eq(0))  # bit 5 OFF → no snow / ice
    )

    # Apply the mask: pixels where mask=0 become transparent (NoData)
    # They will be excluded when the collection is reduced to a median composite
    return image.updateMask(mask)


# =============================================================
# FUNCTION 2 — Radiometric Scaling + Index Computation
# =============================================================

def _scale_and_compute_indices(image: ee.Image) -> ee.Image:
    """
    Convert raw Landsat digital numbers to physical units and compute
    6 spectral / thermal indices from a single cloud-masked image.

    WHY WE DO THIS
    --------------
    Raw Landsat bands are stored as integers for compact file size.
    They carry no physical meaning until scaled. The 6 indices compress
    the raw bands into vegetation, moisture, and temperature signals
    that are directly mechanistically linked to evapotranspiration.

    Together the 6 indices cover three physical ET pathways:
        • Vegetation amount (transpiration source)  → NDVI, EVI, SAVI
        • Water availability (soil + canopy moisture) → NDWI, NDMI
        • Energy balance (latent vs sensible heat)    → LST

    SCALE FACTORS (Landsat Collection 2 Level-2 standard)
    -----------------------------------------------------
    Surface Reflectance bands : DN × 0.0000275 + (−0.2)  → dimensionless (0–1)
    Land Surface Temperature   : DN × 0.00341802 + 149.0 − 273.15 → degrees Celsius

    Landsat 8/9 Band Mapping
    ------------------------
    SR_B2 = Blue   SR_B3 = Green   SR_B4 = Red
    SR_B5 = NIR    SR_B6 = SWIR1   ST_B10 = Thermal IR

    Parameters
    ----------
    image : ee.Image
        Cloud-masked Landsat C2 L2 image (output of _mask_cloud).

    Returns
    -------
    ee.Image
        6-band image with bands named: NDVI, EVI, SAVI, NDWI, NDMI, LST
        The acquisition timestamp is preserved for temporal operations.
    """

    # ------------------------------------------------------------------
    # STEP 1 — Apply surface reflectance scale factors
    # Raw DN × 0.0000275 − 0.2 → physical surface reflectance (0 to ~1)
    # We select only the 5 optical bands we need (skip thermal here)
    # ------------------------------------------------------------------
    optical = (image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"])
                    .multiply(0.0000275).add(-0.2))

    # Give each band a meaningful alias for the index formulas below
    blue  = optical.select("SR_B2")   # ~0.45–0.51 µm  — absorbs atmosphere
    green = optical.select("SR_B3")   # ~0.53–0.59 µm  — water reflects this
    red   = optical.select("SR_B4")   # ~0.64–0.67 µm  — absorbed by chlorophyll
    nir   = optical.select("SR_B5")   # ~0.85–0.88 µm  — strongly reflected by leaves
    swir1 = optical.select("SR_B6")   # ~1.57–1.65 µm  — absorbed by liquid water in leaves

    # ------------------------------------------------------------------
    # STEP 2 — Convert thermal band to Land Surface Temperature in Celsius
    # Formula: DN × 0.00341802 + 149.0  → Kelvin,  then − 273.15 → Celsius
    # Cool pixels = high latent heat (evaporation consuming energy) = high ET
    # Warm pixels = energy going into sensible heat = low ET
    # ------------------------------------------------------------------
    lst = (image.select("ST_B10")
                .multiply(0.00341802).add(149.0).subtract(273.15)
                .rename("LST"))

    # ------------------------------------------------------------------
    # INDEX 1 — NDVI: Normalised Difference Vegetation Index
    # Formula: (NIR − Red) / (NIR + Red)
    #
    # Why: Healthy green vegetation strongly reflects NIR and absorbs Red
    #      (chlorophyll uses red light for photosynthesis).
    #      Range: −1 (water) to +1 (dense canopy).
    #      High NDVI → dense crop canopy → high transpiration → high ETa.
    # ------------------------------------------------------------------
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    # ------------------------------------------------------------------
    # INDEX 2 — EVI: Enhanced Vegetation Index
    # Formula: 2.5 × (NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1)
    #
    # Why: NDVI saturates in dense canopies and is sensitive to soil
    #      background and atmospheric haze. EVI fixes both:
    #      • The "−7.5·Blue" term corrects for atmospheric Rayleigh scattering
    #        (blue light is scattered most by the atmosphere).
    #      • The "+ 1" in the denominator stabilises against bright bare soil.
    #      • The "2.5" multiplier scales back to the NDVI range.
    #      Better than NDVI in dense rice paddies where NDVI flattens out.
    # ------------------------------------------------------------------
    evi = (nir.subtract(red).multiply(2.5)
              .divide(
                  nir.add(red.multiply(6))
                     .subtract(blue.multiply(7.5))
                     .add(1)
              ).rename("EVI"))

    # ------------------------------------------------------------------
    # INDEX 3 — SAVI: Soil-Adjusted Vegetation Index  (L = 0.5)
    # Formula: 1.5 × (NIR − Red) / (NIR + Red + 0.5)
    #
    # Why: some fields may have exposed soil between crop rows
    #      or during inter-season fallow periods. Bright soil increases
    #      the NIR denominator and biases NDVI upward even with no crops.
    #      The constant 0.5 (the soil adjustment factor L) suppresses
    #      this soil-brightness effect. The 1.5 multiplier re-scales the
    #      result back to the same numerical range as NDVI.
    # ------------------------------------------------------------------
    savi = (nir.subtract(red).multiply(1.5)
               .divide(nir.add(red).add(0.5))
               .rename("SAVI"))

    # ------------------------------------------------------------------
    # INDEX 4 — NDWI: Normalised Difference Water Index
    # Formula: (Green − NIR) / (Green + NIR)
    #
    # Why: Open water and flooded surfaces absorb NIR but reflect green,
    #      making NDWI positive (> 0) over water / flooded paddies.
    # ------------------------------------------------------------------
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

    # ------------------------------------------------------------------
    # INDEX 5 — NDMI: Normalised Difference Moisture Index
    # Formula: (NIR − SWIR1) / (NIR + SWIR1)
    #
    # Why: SWIR1 (~1.6 µm) is absorbed by liquid water held inside plant
    #      leaves. High NDMI = leaves full of water = plant not stressed
    #      = high actual ET. Low NDMI = water-stressed crop = lower ETa.
    #      This is mechanistically the most directly linked index to ETa:
    #      leaf water content controls stomatal conductance which controls
    #      transpiration.
    # ------------------------------------------------------------------
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")

    # ------------------------------------------------------------------
    # Stack all 6 bands into one image and carry the timestamp forward
    # (the timestamp is needed if we ever do temporal operations later)
    # ------------------------------------------------------------------
    return (ee.Image([ndvi, evi, savi, ndwi, ndmi, lst])
              .copyProperties(image, ["system:time_start"]))


# =============================================================
# FUNCTION 3 — Main Download (used by the pipeline)
# =============================================================

def download_landsat_indices_30m(
    aoi_gdf: gpd.GeoDataFrame,
    year: int,
    month: int,
    max_cloud: int,
    output_file: str | Path,
) -> Path:
    """
    Download a cloud-masked, scaled Landsat 8+9 median composite at 30 m
    with 6 spectral/thermal index bands: NDVI, EVI, SAVI, NDWI, NDMI, LST.

    OVERALL STRATEGY
    ----------------
    1. Load all Landsat 8 and 9 images for the target month over the AOI.
    2. Keep only images with < max_cloud% cloud cover.
    3. For every image: mask clouds → scale DN → compute 6 indices.
    4. Reduce the collection to a MEDIAN composite.
       (Median is robust: even if 2 of 5 images have a cloud edge that
        slipped through masking, the median picks a clear-sky value.)
    5. Clip to the AOI boundary and export as a 30 m GeoTIFF.

    WHY BOTH L8 AND L9
    ------------------
    Landsat 8 and 9 have near-identical sensors (cross-calibrated) and
    are offset in their orbits, giving a combined ~4-day revisit vs ~8-day
    for either alone. More revisits → more valid observations per month →
    less chance that clouds block every pass → more reliable median.
    (Note: for 2018 only Landsat 8 contributes — L9 launched Sept 2021.)

    Parameters
    ----------
    aoi_gdf     : GeoDataFrame  Study area in EPSG:4326
    year        : int           Target year  (e.g. 2018)
    month       : int           Target month (1–12)
    max_cloud   : int           Maximum per-image cloud cover % (e.g. 5)
    output_file : path          Output GeoTIFF path

    Returns
    -------
    Path  Path to the saved 6-band GeoTIFF
    """
    # Authenticate and initialise Google Earth Engine
    # (uses service account JSON if present, otherwise browser OAuth)
    ee_authenticate()

    # Build the date range: first day to last day of the target month
    # calendar.monthrange handles variable month lengths and leap years
    last_day   = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date   = f"{year}-{month:02d}-{last_day:02d}"
    print(f"\n  Period : {start_date} to {end_date}")

    # Convert the geopandas AOI to a GEE geometry for spatial filtering
    # Ensure the CRS is EPSG:4326 first (GEE expects geographic coordinates)
    gdf = aoi_gdf.to_crs("EPSG:4326") if aoi_gdf.crs.to_string() != "EPSG:4326" else aoi_gdf
    roi = geemap.geopandas_to_ee(gdf).geometry()

    # Cloud filter: exclude any scene where more than max_cloud% is cloudy
    # This is a scene-level filter applied BEFORE loading individual images
    cloud_filter = ee.Filter.lt("CLOUD_COVER", max_cloud)

    # ------------------------------------------------------------------
    # Load Landsat 8 Collection 2 Level-2 Surface Reflectance
    # Available: 2013 – present
    # filterBounds → only scenes that overlap our AOI
    # filterDate   → only scenes within the target month
    # filter       → only scenes with acceptable cloud cover
    # ------------------------------------------------------------------
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(cloud_filter))

    # ------------------------------------------------------------------
    # Load Landsat 9 Collection 2 Level-2 Surface Reflectance
    # Available: 2021 – present  (will be empty for years before 2021)
    # ------------------------------------------------------------------
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(cloud_filter))

    # Merge L8 and L9 into one collection
    # For 2018 this is just L8; for 2022+ this is both satellites
    merged = l8.merge(l9)

    # Check how many images were found — raise an error early if zero
    count = merged.size().getInfo()
    print(f"  Images found (L8+L9) : {count}")
    if count == 0:
        raise ValueError("No Landsat images found — try increasing max_cloud or widening the date range.")

    # ------------------------------------------------------------------
    # Processing chain applied to every image in the collection:
    #   1. _mask_cloud              → set cloud/shadow/snow pixels to NoData
    #   2. _scale_and_compute_indices → DN → physical units → 6 index bands
    #
    # Then reduce the entire collection to a MEDIAN composite:
    #   .median() computes the median value per pixel per band across
    #   all valid (unmasked) observations in the month.
    #
    # Finally clip to the AOI boundary to remove pixels outside the scheme.
    # ------------------------------------------------------------------
    composite = (merged
                 .map(_mask_cloud)
                 .map(_scale_and_compute_indices)
                 .median()
                 .clip(roi))

    os.makedirs(Path(output_file).parent, exist_ok=True)

    # Export the composite from GEE to a local GeoTIFF
    # scale=30 → 30 m pixel size  |  file_per_band=False → single multi-band file
    # This is where the actual data transfer happens — all prior steps are
    # lazy GEE computation graphs that only execute at export time.
    print(f"  Exporting 6-band index composite → {output_file}")
    geemap.ee_export_image(
        composite,
        filename=str(output_file),
        scale=30,
        region=roi,
        file_per_band=False,
    )
    print(f"  Landsat indices download complete: {output_file}")
    return Path(output_file)


# =============================================================
# FUNCTION 4 — Raw Band Download (utility, NOT used by pipeline)
# =============================================================

def download_landsat_30m(
    aoi_gdf: gpd.GeoDataFrame,
    year: int,
    month: int,
    max_cloud: int,
    satellite: str,
    output_file: str | Path,
    composite_method: str = "median",
):
    """
    Download raw multispectral Landsat bands (NOT computed indices)
    for a single satellite (L8 or L9) as a 30 m GeoTIFF.

    This is a secondary utility function — NOT called by main.py.
    Useful for visual inspection, alternative index computation,
    or feeding a different downstream workflow.

    DIFFERENCE FROM download_landsat_indices_30m
    --------------------------------------------
    • Only one satellite at a time (no L8+L9 merge)
    • Outputs raw DN bands, not computed indices
    • No cloud masking or radiometric scaling applied
    • Supports "mosaic" compositing (least cloudy pixel wins)
      in addition to "median"

    Parameters
    ----------
    aoi_gdf          : GeoDataFrame  (EPSG:4326)
    year             : int           e.g. 2018
    month            : int           1–12
    max_cloud        : int           maximum cloud cover %
    satellite        : str           "L8" or "L9"
    output_file      : path          Output GeoTIFF path
    composite_method : str           "median" — average across scenes
                                     "mosaic" — pick least-cloudy pixel per location
    """
    ee_authenticate()

    # Build date range for the target month
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day:02d}"
    print(f"\nSelected period: {start_date} to {end_date}")

    gdf = aoi_gdf.to_crs("EPSG:4326") if aoi_gdf.crs.to_string() != "EPSG:4326" else aoi_gdf
    roi = geemap.geopandas_to_ee(gdf).geometry()

    # Select the correct GEE collection ID based on satellite choice
    if satellite == "L8":
        collection_id = "LANDSAT/LC08/C02/T1_L2"
    elif satellite == "L9":
        collection_id = "LANDSAT/LC09/C02/T1_L2"
    else:
        raise ValueError("satellite must be 'L8' or 'L9'")

    # Load the collection with spatial, temporal, and cloud filters
    # sort("CLOUD_COVER") puts the clearest scenes first (used by mosaic)
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
        .sort("CLOUD_COVER")
    )

    count = collection.size().getInfo()
    print(f"Landsat images found: {count}")
    if count == 0:
        raise ValueError("No Landsat images found for the given AOI/date/cloud filters.")

    # Composite method:
    # "median" → pixel-wise median across all scenes (robust to outliers)
    # "mosaic" → first valid pixel from the sorted (least cloudy) stack
    if composite_method == "median":
        image = collection.median()
    elif composite_method == "mosaic":
        image = collection.sort("CLOUD_COVER").mosaic()
    else:
        raise ValueError("composite_method must be 'median' or 'mosaic'.")

    # Clip to the AOI boundary
    image = image.clip(roi)

    os.makedirs(Path(output_file).parent, exist_ok=True)

    # Export to local disk at 30 m resolution
    geemap.ee_export_image(
        image,
        filename=str(output_file),
        scale=30,
        region=roi,
        file_per_band=False,
    )

    print(f"Landsat download complete: {output_file}")
