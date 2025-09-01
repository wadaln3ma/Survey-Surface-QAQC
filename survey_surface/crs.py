from typing import Optional
import geopandas as gpd
from pyproj import CRS

def to_crs(gdf: gpd.GeoDataFrame, target: str) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame to target CRS (EPSG:xxxx or proj string)."""
    if gdf.crs is None:
        raise ValueError('Input GeoDataFrame has no CRS; provide --in-crs.')
    return gdf.to_crs(CRS.from_user_input(target))

def ensure_crs(gdf: gpd.GeoDataFrame, crs: Optional[str]) -> gpd.GeoDataFrame:
    if crs:
        gdf = gdf.set_crs(CRS.from_user_input(crs), allow_override=True)
    if gdf.crs is None:
        raise ValueError('CRS is required. Use --in-crs to set the input CRS.')
    return gdf
