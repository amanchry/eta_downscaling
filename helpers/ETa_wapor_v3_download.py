import os
import numpy as np
import rasterio
import json
from osgeo import gdal


def download_wapor_v3_L1_eta_data(
    year,
    month,
    output_folder,
    geojson_obj,
):
    """
    Download and clip WAPOR v3 L1 monthly ETa (300m, global) for a month.

    Parameters:
    - year  : int, e.g. 2018
    - month : int, 1–12
    - output_folder : str or Path, folder to save processed GeoTIFF
    - geojson_obj   : dict, GeoJSON geometry for clipping
    """
    os.makedirs(output_folder, exist_ok=True)

    url = (
        f"https://gismgr.fao.org/DATA/WAPOR-3/MAPSET/L1-AETI-M/"
        f"WAPOR-3.L1-AETI-M.{year}-{month:02d}.tif"
    )
    output_filename = f"WAPOR3_L1_AETI_M_{year}_{month:02d}.tif"
    output_path = os.path.join(output_folder, output_filename)
    temp_clip = os.path.join(output_folder, f"temp_{output_filename}")


    if os.path.exists(output_path):
        print(f"✅ {output_filename} already exists, skipping...")
        pass
        


    try:
        gdal.UseExceptions()
    except Exception:
        pass

    cutline_path = "/vsimem/cutline_L1.geojson"
    gdal.FileFromMemBuffer(cutline_path, json.dumps(geojson_obj).encode("utf-8"))

    vsicurl_url = f"/vsicurl/{url}"
    print(f"Downloading... {url}")

    warp_options = gdal.WarpOptions(
        cutlineDSName=cutline_path,
        cropToCutline=True,
        dstNodata=-9999,
    )
    gdal.Warp(destNameOrDestDS=temp_clip, srcDSOrSrcDSTab=vsicurl_url, options=warp_options)

    try:
        with rasterio.open(temp_clip) as src:
            profile = src.profile
            data = src.read(1)
            nodata = src.nodata

            data = np.where(data == nodata, -9999, data)
            scaled_data = np.where(data != -9999, data * 0.1, -9999)

            profile.update(dtype=rasterio.float32, nodata=-9999, compress="LZW")

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(scaled_data.astype(rasterio.float32), 1)

        if os.path.exists(temp_clip):
            os.remove(temp_clip)

        print(f"Saved: {output_filename}")

    except Exception as e:
        if os.path.exists(temp_clip):
            os.remove(temp_clip)
        raise e

    return output_path


def download_wapor_v3_L3_eta_data(
    year,
    month,
    output_folder,
    geojson_obj,
    region_id,
):
    """
    Download and clip WAPOR v3 L3 monthly ETa (20m, regional) for  month.

    Parameters:
    - year      : int, e.g. 2018
    - month     : int, 1–12
    - output_folder : str or Path, folder to save processed GeoTIFF
    - geojson_obj   : dict, GeoJSON geometry for clipping
    - region_id : str, WAPOR L3 region code, e.g. "KMW"
    """
    os.makedirs(output_folder, exist_ok=True)

    url = (
        f"https://storage.googleapis.com/fao-gismgr-wapor-3-data/DATA/WAPOR-3/MOSAICSET/L3-AETI-M/"
        f"WAPOR-3.L3-AETI-M.{region_id}.{year}-{month:02d}.tif"
    )
    output_filename = f"WAPOR3_L3_AETI_M_{year}_{month:02d}.tif"
    output_path = os.path.join(output_folder, output_filename)
    temp_clip = os.path.join(output_folder, f"temp_{output_filename}")

    if os.path.exists(output_path):
        print(f"✅ {output_filename} already exists, skipping...")
        pass
    

    try:
        gdal.UseExceptions()
    except Exception:
        pass

    cutline_path = "/vsimem/cutline_L3.geojson"
    gdal.FileFromMemBuffer(cutline_path, json.dumps(geojson_obj).encode("utf-8"))

    vsicurl_url = f"/vsicurl/{url}"
    print(f"Downloading... {url}")

    warp_options = gdal.WarpOptions(
        cutlineDSName=cutline_path,
        cropToCutline=True,
        dstNodata=-9999,
    )
    gdal.Warp(destNameOrDestDS=temp_clip, srcDSOrSrcDSTab=vsicurl_url, options=warp_options)

    try:
        with rasterio.open(temp_clip) as src:
            profile = src.profile
            data = src.read(1)
            nodata = src.nodata

            data = np.where(data == nodata, -9999, data)
            scaled_data = np.where(data != -9999, data * 0.1, -9999)

            profile.update(dtype=rasterio.float32, nodata=-9999, compress="LZW")

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(scaled_data.astype(rasterio.float32), 1)

        if os.path.exists(temp_clip):
            os.remove(temp_clip)

        print(f"Saved: {output_filename}")

    except Exception as e:
        if os.path.exists(temp_clip):
            os.remove(temp_clip)
        raise e

    return output_path
