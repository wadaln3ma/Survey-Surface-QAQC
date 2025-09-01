# Survey Surface QA/QC

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Geo](https://img.shields.io/badge/domain-geospatial-%23347d39)]()
[![Made for Civil 3D](https://img.shields.io/badge/Civil%203D-LandXML-orange)]()

Build **DEMs** from GNSS/drone points, extract **contours**, compute **cut/fill volumes**, export **LandXML** surfaces for **Civil 3D**, and preview results in a lightweight **Streamlit** app.

> **Why it matters:** This repo demonstrates end-to-end surveying & GIS skills: CRS handling, interpolation, raster & vector outputs, volume analysis, and CAD interoperability.

---

## Features

- **Input:** CSV points (`x,y,z` or `lon,lat,z`)
- **CRS-aware:** set `--in-crs`, reproject to a metric CRS via `--to-crs` (e.g., **UTM 38N `EPSG:32638`**)
- **DEM build (IDW):** portable inverse distance weighting → GeoTIFF
- **Contours:** marching squares → GeoJSON
- **Volumes:** cut/fill vs **reference plane** or **reference DEM**
- **LandXML export:** SciPy Delaunay → minimal LandXML 1.2 (Pnts/Faces) for **Civil 3D**
- **Viewer:** Streamlit + Folium for quick map previews and DEM stats

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Synthetic sample points (Riyadh-ish lon/lat)
python scripts/generate_synthetic_points.py --out data/sample_points.csv

# 2) DEM (grid in meters: UTM 38N)
python -m survey_surface.cli build-grid \
  --input data/sample_points.csv \
  --x x --y y --z z \
  --in-crs EPSG:4326 --to-crs EPSG:32638 \
  --cell 2.0 \
  --out outputs/dem.tif

# 3) Contours (1 m)
python -m survey_surface.cli contours \
  --raster outputs/dem.tif --interval 1.0 \
  --out outputs/contours.geojson

# 4a) Volumes vs plane (z = 600 m)
python -m survey_surface.cli volumes --dem outputs/dem.tif --ref-z 600

# 4b) Volumes vs reference DEM
# python -m survey_surface.cli volumes --dem outputs/dem.tif --ref-dem path/to/ref.tif

# 5) LandXML (import into Civil 3D)
python -m survey_surface.cli landxml \
  --input data/sample_points.csv \
  --in-crs EPSG:4326 --to-crs EPSG:32638 \
  --out outputs/surface_landxml.xml

# 6) QA/QC report (HTML)
python -m survey_surface.cli report \
  --dem outputs/dem.tif \
  --contours outputs/contours.geojson \
  --out outputs/report.html

# 7) Web viewer
streamlit run streamlit_app.py
