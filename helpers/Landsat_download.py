import ee
import geemap
import geopandas as gpd
import os
from pathlib import Path
import calendar
from .gee_auth import ee_authenticate


def _mask_cloud(image: ee.Image) -> ee.Image:
    """
    Mask cloud, cloud shadow, dilated cloud, and snow pixels using
    the QA_PIXEL band from Landsat Collection 2 Level-2.

    Bits masked:
        1 — Dilated cloud
        3 — Cloud
        4 — Cloud shadow
        5 — Snow / ice

    Parameters
    ----------
    image : ee.Image  Raw Landsat C2 L2 image

    Returns
    -------
    ee.Image  Same image with cloud pixels masked
    """
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)   # dilated cloud
          .And(qa.bitwiseAnd(1 << 3).eq(0))   # cloud
          .And(qa.bitwiseAnd(1 << 4).eq(0))   # cloud shadow
          .And(qa.bitwiseAnd(1 << 5).eq(0))   # snow
    )
    return image.updateMask(mask)


def _scale_and_compute_indices(image: ee.Image) -> ee.Image:
    """
    Apply Landsat C2 L2 scale factors and compute 6 spectral / thermal
    indices from a single (already cloud-masked) image.

    Scale factors:
        SR bands   : × 0.0000275 + (−0.2)
        ST_B10     : × 0.00341802 + 149.0 − 273.15  → °C

    Indices returned (6 bands):
        NDVI  = (NIR − Red) / (NIR + Red)
        EVI   = 2.5 × (NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1)
        SAVI  = 1.5 × (NIR − Red) / (NIR + Red + 0.5)
        NDWI  = (Green − NIR) / (Green + NIR)
        NDMI  = (NIR − SWIR1) / (NIR + SWIR1)
        LST   = Land Surface Temperature in °C

    Landsat 8/9 band mapping:
        B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1

    Parameters
    ----------
    image : ee.Image  Cloud-masked Landsat C2 L2 image

    Returns
    -------
    ee.Image  6-band image with bands named NDVI, EVI, SAVI, NDWI, NDMI, LST
    """
    # Apply surface reflectance scale factors
    optical = (image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"])
                    .multiply(0.0000275).add(-0.2))

    blue  = optical.select("SR_B2")
    green = optical.select("SR_B3")
    red   = optical.select("SR_B4")
    nir   = optical.select("SR_B5")
    swir1 = optical.select("SR_B6")

    # Land Surface Temperature in Celsius
    lst = (image.select("ST_B10")
                .multiply(0.00341802).add(149.0).subtract(273.15)
                .rename("LST"))

    # NDVI — Normalized Difference Vegetation Index
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    # EVI — Enhanced Vegetation Index
    evi = (nir.subtract(red).multiply(2.5)
              .divide(
                  nir.add(red.multiply(6))
                     .subtract(blue.multiply(7.5))
                     .add(1)
              ).rename("EVI"))

    # SAVI — Soil-Adjusted Vegetation Index (L = 0.5)
    savi = (nir.subtract(red).multiply(1.5)
               .divide(nir.add(red).add(0.5))
               .rename("SAVI"))

    # NDWI — Normalized Difference Water Index
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

    # NDMI — Normalized Difference Moisture Index
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")

    return (ee.Image([ndvi, evi, savi, ndwi, ndmi, lst])
              .copyProperties(image, ["system:time_start"]))


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

    Both Landsat 8 and 9 are merged for the target month, cloud-masked
    using QA_PIXEL, scaled to physical units, and reduced to a median
    composite before exporting the 6-band index stack.

    Parameters
    ----------
    aoi_gdf     : GeoDataFrame  Study area in EPSG:4326
    year        : int           Target year  (e.g. 2018)
    month       : int           Target month (1–12)
    max_cloud   : int           Maximum per-image cloud cover %
    output_file : path          Output GeoTIFF path

    Returns
    -------
    Path  Path to the saved GeoTIFF
    """
    ee_authenticate()

    last_day   = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date   = f"{year}-{month:02d}-{last_day:02d}"
    print(f"\n  Period : {start_date} to {end_date}")

    gdf = aoi_gdf.to_crs("EPSG:4326") if aoi_gdf.crs.to_string() != "EPSG:4326" else aoi_gdf
    roi = geemap.geopandas_to_ee(gdf).geometry()

    cloud_filter = ee.Filter.lt("CLOUD_COVER", max_cloud)

    # Load Landsat 8 and Landsat 9 separately then merge
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(roi).filterDate(start_date, end_date)
            .filter(cloud_filter))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(roi).filterDate(start_date, end_date)
            .filter(cloud_filter))

    merged = l8.merge(l9)
    count  = merged.size().getInfo()
    print(f"  Images found (L8+L9) : {count}")
    if count == 0:
        raise ValueError("No Landsat images found — try increasing max_cloud or widening the date range.")

    # Apply cloud mask and compute indices per image, then take median
    composite = (merged
                 .map(_mask_cloud)
                 .map(_scale_and_compute_indices)
                 .median()
                 .clip(roi))

    os.makedirs(Path(output_file).parent, exist_ok=True)

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
    Download a Landsat 30m multispectral composite clipped to the AOI.

    Parameters
    ----------
    aoi_gdf  : GeoDataFrame  (EPSG:4326)
    year     : int  e.g. 2018
    month    : int  1–12
    max_cloud: int  maximum cloud cover %
    satellite: str  "L8" or "L9"
    output_file      : path for the output GeoTIFF
    composite_method : "median" or "mosaic"
    """
    ee_authenticate()

    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day:02d}"
    print(f"\nSelected period: {start_date} to {end_date}")


    gdf = aoi_gdf.to_crs("EPSG:4326") if aoi_gdf.crs.to_string() != "EPSG:4326" else aoi_gdf
    roi = geemap.geopandas_to_ee(gdf).geometry()

    if satellite == "L8":
        collection_id = "LANDSAT/LC08/C02/T1_L2"
    elif satellite == "L9":
        collection_id = "LANDSAT/LC09/C02/T1_L2"
    else:
        raise ValueError("satellite must be 'L8' or 'L9'")

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

    if composite_method == "median":
        image = collection.median()
    elif composite_method == "mosaic":
        image = collection.sort("CLOUD_COVER").mosaic()
    else:
        raise ValueError("composite_method must be 'median' or 'mosaic'.")

    image = image.clip(roi)

    os.makedirs(Path(output_file).parent, exist_ok=True)

    geemap.ee_export_image(
        image,
        filename=str(output_file),
        scale=30,
        region=roi,
        file_per_band=False,
    )

    print(f"Landsat download complete: {output_file}")
