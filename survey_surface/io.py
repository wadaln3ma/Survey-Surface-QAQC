from typing import Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .crs import ensure_crs, to_crs

def read_points_csv(
    path: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    lon: Optional[str] = None,
    lat: Optional[str] = None,
    z: str = 'z',
    in_crs: Optional[str] = None,
    to: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load CSV and return GeoDataFrame of points with elevation column z.
    Provide either x/y or lon/lat. If `to` is provided, reproject to that CRS.
    """
    df = pd.read_csv(path)
    if x and y:
        xx, yy = df[x].astype(float).values, df[y].astype(float).values
    elif lon and lat:
        xx, yy = df[lon].astype(float).values, df[lat].astype(float).values
    else:
        raise ValueError('Provide either --x/--y or --lon/--lat')
    if z not in df.columns:
        raise ValueError(f'Missing elevation column: {z}')

    gdf = gpd.GeoDataFrame(df, geometry=[Point(a, b) for a, b in zip(xx, yy)], crs=None)
    gdf = ensure_crs(gdf, in_crs)
    if to:
        gdf = to_crs(gdf, to)
    return gdf
