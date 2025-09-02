from typing import Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .crs import ensure_crs, to_crs

# survey_surface/io.py  (replace the top of read_points_csv with this logic)
def read_points_csv(path, x=None, y=None, lon=None, lat=None, z="z", in_crs="EPSG:4326", to=None):
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point

    df = pd.read_csv(path)

    # --- Auto-detect if nothing was specified ---
    if (x is None or y is None) and (lon is None or lat is None):
        lower = {c.lower(): c for c in df.columns}
        # Try several common patterns
        if "x" in lower and "y" in lower:
            x, y = lower["x"], lower["y"]
        elif "easting" in lower and "northing" in lower:
            x, y = lower["easting"], lower["northing"]
        elif "lon" in lower and "lat" in lower:
            lon, lat = lower["lon"], lower["lat"]
        elif "longitude" in lower and "latitude" in lower:
            lon, lat = lower["longitude"], lower["latitude"]
        # If still nothing, we’ll error below as before.

    # --- Validate we have coords now ---
    if x and y:
        xx, yy = df[x].astype(float).values, df[y].astype(float).values
        gdf = gpd.GeoDataFrame(df, geometry=[Point(x_, y_) for x_, y_ in zip(xx, yy)], crs=in_crs)
    elif lon and lat:
        xx, yy = df[lon].astype(float).values, df[lat].astype(float).values
        gdf = gpd.GeoDataFrame(df, geometry=[Point(x_, y_) for x_, y_ in zip(xx, yy)], crs=in_crs)
    else:
        raise ValueError(
            f"Provide either --x/--y or --lon/--lat (columns found: {list(df.columns)})"
        )

    if to:
        gdf = gdf.to_crs(to)

    if z not in df.columns:
        raise ValueError(f"Missing elevation column: {z}")

    return gdf
