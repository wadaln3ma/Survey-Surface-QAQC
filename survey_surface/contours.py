from typing import List
import json
import numpy as np
import rasterio
from skimage import measure

def contours_geojson_from_raster(raster_path: str, interval: float, out_path: str):
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        transform = src.transform
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        levels = _levels(vmin, vmax, interval)

        features = []
        for lvl in levels:
            cs = measure.find_contours(arr - lvl, 0.0)
            for poly in cs:
                coords = []
                for r, c in poly:
                    x = transform.c + c * transform.a + r * transform.b
                    y = transform.f + c * transform.d + r * transform.e
                    coords.append([x, y])
                if len(coords) < 2:
                    continue
                features.append({
                    'type': 'Feature',
                    'properties': { 'elev': float(lvl) },
                    'geometry': { 'type': 'LineString', 'coordinates': coords }
                })
        fc = { 'type': 'FeatureCollection', 'features': features }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(fc, f)

def _levels(vmin: float, vmax: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError('interval must be > 0')
    import numpy as np
    start = np.floor(vmin / step) * step
    end   = np.ceil(vmax / step) * step
    n = int(np.floor((end - start) / step)) + 1
    return [float(start + i * step) for i in range(n)]
