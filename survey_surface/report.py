import json
from typing import Optional
import numpy as np
import rasterio

def make_report_html(dem_path: str, contours_path: Optional[str] = None) -> str:
    with rasterio.open(dem_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        stats = {
            'min': float(np.nanmin(arr)),
            'max': float(np.nanmax(arr)),
            'mean': float(np.nanmean(arr)),
            'std': float(np.nanstd(arr)),
            'rows': int(arr.shape[0]),
            'cols': int(arr.shape[1])
        }

    contours_json = None
    if contours_path:
        with open(contours_path, 'r', encoding='utf-8') as f:
            contours_json = json.load(f)

    html = f"""
<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1'/>
<title>Survey Surface QA/QC Report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;line-height:1.4;margin:24px;color:#0f172a}}
.card{{border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:16px}}
.code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f8fafc;border-radius:8px;padding:8px;display:block;white-space:pre-wrap}}
.kv td{{padding:4px 8px;border-bottom:1px solid #f1f5f9}}
</style>
</head>
<body>
<h1>Survey Surface QA/QC Report</h1>
<div class='card'>
  <h3>Raster Stats</h3>
  <table class='kv'>
    <tr><td>Min</td><td>{stats['min']:.3f}</td></tr>
    <tr><td>Max</td><td>{stats['max']:.3f}</td></tr>
    <tr><td>Mean</td><td>{stats['mean']:.3f}</td></tr>
    <tr><td>Std</td><td>{stats['std']:.3f}</td></tr>
    <tr><td>Rows</td><td>{stats['rows']}</td></tr>
    <tr><td>Cols</td><td>{stats['cols']}</td></tr>
  </table>
</div>
<div class='card'>
  <h3>Inputs</h3>
  <div class='code'>DEM: {dem_path}</div>
  {f"<div class='code'>Contours: {contours_path}</div>" if contours_path else ''}
</div>
<div class='card'>
  <h3>Contours Preview (first 2 features)</h3>
  <div class='code'>"""
    if contours_json and contours_json.get('features'):
        sample = contours_json['features'][:2]
        html += json.dumps(sample, indent=2)
    else:
        html += 'No contours provided.'
    html += """</div>
</div>
</body>
</html>
"""
    return html
