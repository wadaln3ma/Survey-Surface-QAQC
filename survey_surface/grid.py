from typing import Tuple
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Simple IDW for demos
def idw_grid(points_xy: np.ndarray, values: np.ndarray, cell: float, power: float = 2.0) -> Tuple[np.ndarray, rasterio.Affine]:
    x, y = points_xy[:, 0], points_xy[:, 1]
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()
    minx -= cell; miny -= cell; maxx += cell; maxy += cell

    width  = int(np.ceil((maxx - minx) / cell))
    height = int(np.ceil((maxy - miny) / cell))

    xs = minx + (np.arange(width) + 0.5) * cell
    ys = maxy - (np.arange(height) + 0.5) * cell  # top-to-bottom

    grid = np.full((height, width), np.nan, dtype=float)
    for j, yy in enumerate(ys):
        dy = yy - y
        for i, xx in enumerate(xs):
            dx = xx - x
            d2 = dx * dx + dy * dy
            near = d2 < 1e-9
            if np.any(near):
                grid[j, i] = float(values[near][0]); continue
            w = 1.0 / np.power(d2, power / 2.0)
            grid[j, i] = float(np.sum(w * values) / np.sum(w))

    transform = from_origin(minx, maxy, cell, cell)
    return grid, transform

def write_geotiff(path: str, arr: np.ndarray, transform: rasterio.Affine, crs: str):
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.astype('float32'), 1)
