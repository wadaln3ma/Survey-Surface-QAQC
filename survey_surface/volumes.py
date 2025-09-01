from typing import Optional, Dict
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def _pixel_area(transform) -> float:
    return abs(transform.a * transform.e - transform.b * transform.d)

def volumes_against_plane(dem_path: str, ref_z: float) -> Dict[str, float]:
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype('float64')
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        diff = arr - ref_z
        area = _pixel_area(src.transform)
        fill = np.nansum(np.where(diff > 0, diff, 0.0)) * area
        cut  = np.nansum(np.where(diff < 0, -diff, 0.0)) * area
        return { 'cut_m3': float(cut), 'fill_m3': float(fill), 'net_m3': float(fill - cut) }

def volumes_against_dem(dem_path: str, ref_dem_path: str) -> Dict[str, float]:
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype('float64')
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        dst = np.full_like(arr, np.nan, dtype='float64')
        with rasterio.open(ref_dem_path) as ref:
            reproject(
                source=rasterio.band(ref, 1),
                destination=dst,
                src_transform=ref.transform,
                src_crs=ref.crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear
            )
        diff = arr - dst
        area = _pixel_area(src.transform)
        fill = np.nansum(np.where(diff > 0, diff, 0.0)) * area
        cut  = np.nansum(np.where(diff < 0, -diff, 0.0)) * area
        return { 'cut_m3': float(cut), 'fill_m3': float(fill), 'net_m3': float(fill - cut) }
