# streamlit_app.py
import os
import io
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image  # for PNG overlay

# Import your pipeline modules
from survey_surface.io import read_points_csv
from survey_surface.grid import idw_grid, write_geotiff
from survey_surface.contours import contours_geojson_from_raster
from survey_surface.report import make_report_html
from survey_surface.volumes import volumes_against_plane, volumes_against_dem
from survey_surface.landxml import landxml_from_points_csv


# ---------------------------- CONFIG ----------------------------
st.set_page_config(
    page_title="Survey Surface QA/QC",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_IN_CRS = "EPSG:4326"
DEFAULT_TO_CRS = "EPSG:32638"  # UTM 38N (Riyadh)
DEFAULT_CELL = 2.0
DEFAULT_INTERVAL = 1.0
DEFAULT_REF_Z = 600.0


# ---------------------------- HELPERS ----------------------------
def _save_uploaded_file(uploaded, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    return dest


@st.cache_data(show_spinner=False)
def _run_grid(csv_path: str, x: Optional[str], y: Optional[str],
              lon: Optional[str], lat: Optional[str], z_col: str,
              in_crs: str, to_crs: str, cell: float) -> Tuple[str, dict]:
    gdf = read_points_csv(csv_path, x=x, y=y, lon=lon, lat=lat, z=z_col, in_crs=in_crs, to=to_crs)
    pts = np.vstack([gdf.geometry.x.values, gdf.geometry.y.values]).T
    vals = gdf[z_col].astype(float).values
    arr, transform = idw_grid(pts, vals, cell=cell)
    dem_path = str(OUT_DIR / "dem.tif")
    write_geotiff(dem_path, arr, transform, to_crs)

    a = arr.astype(float)
    stats = {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
        "rows": int(a.shape[0]),
        "cols": int(a.shape[1]),
        "crs": to_crs,
    }
    return dem_path, stats


@st.cache_data(show_spinner=False)
def _run_contours(dem_path: str, interval: float) -> str:
    contours_path = str(OUT_DIR / "contours.geojson")
    contours_geojson_from_raster(dem_path, interval, contours_path)
    return contours_path


@st.cache_data(show_spinner=False)
def _run_report(dem_path: str, contours_path: Optional[str]) -> str:
    html = make_report_html(dem_path, contours_path)
    out = OUT_DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


@st.cache_data(show_spinner=False)
def _run_landxml_from_csv(csv_path: str, x: Optional[str], y: Optional[str],
                          lon: Optional[str], lat: Optional[str], z_col: str,
                          in_crs: str, to_crs: str, name: str = "SurveyTIN") -> str:
    xml = landxml_from_points_csv(csv_path, x=x, y=y, lon=lon, lat=lat, z=z_col,
                                  in_crs=in_crs, to_crs=to_crs, surface_name=name)
    out = OUT_DIR / "surface_landxml.xml"
    out.write_text(xml, encoding="utf-8")
    return str(out)


def _bounds_from_raster(dem_path: str):
    with rasterio.open(dem_path) as src:
        return [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]


def _build_map(contours_path: Optional[str], bounds: Optional[list],
               hillshade_png: Optional[str], hillshade_bounds_latlon: Optional[list],
               hs_opacity: float) -> folium.Map:
    # Center Riyadh-ish as fallback
    center = [24.7136, 46.6753]
    m = folium.Map(location=center, zoom_start=12, control_scale=True)

    # Basemaps
    folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)
    folium.TileLayer("Stamen Terrain", name="Terrain", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        control=True,
    ).add_to(m)

    # Hillshade overlay (PNG with alpha), georeferenced to EPSG:4326 bounds
    if hillshade_png and os.path.exists(hillshade_png) and hillshade_bounds_latlon:
        try:
            folium.raster_layers.ImageOverlay(
                name="Hillshade",
                image=hillshade_png,
                bounds=hillshade_bounds_latlon,
                opacity=hs_opacity,
                interactive=False,
                cross_origin=False,
                zindex=400,
            ).add_to(m)
        except Exception as e:
            st.warning(f"Could not add hillshade overlay: {e}")

    if contours_path and os.path.exists(contours_path):
        try:
            with open(contours_path, "r", encoding="utf-8") as f:
                gj = json.load(f)

            def _style(feature):
                elev = feature.get("properties", {}).get("elev", 0.0)
                return {"color": "#7c3aed" if elev else "#0ea5e9", "weight": 2, "opacity": 0.9}

            folium.GeoJson(
                gj,
                name="Contours",
                tooltip=folium.GeoJsonTooltip(fields=["elev"], aliases=["Elev (m)"]),
                style_function=_style,
                show=True,
            ).add_to(m)
        except Exception as e:
            st.warning(f"Could not render contours: {e}")

    if bounds:
        m.fit_bounds(bounds, padding=(10, 10))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _download_button(label: str, path: str, mime: str):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(label, f, file_name=os.path.basename(path), mime=mime)


def _auto_demo_if_empty():
    """If no outputs exist, suggest (and optionally run) an instant demo."""
    have_any = any((OUT_DIR / p).exists() for p in ["dem.tif", "contours.geojson", "report.html"])
    if not have_any:
        st.info("No outputs found yet. Run the **Quick Demo** or upload your CSV to try the pipeline.")
    return have_any


# ---------------------------- HILLSHADE ----------------------------
def _hillshade(arr: np.ndarray, transform, azimuth: float = 315.0, altitude: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """Compute hillshade (0..255) from DEM array using standard algorithm (radians)."""
    # cell sizes (assume north-up)
    dx = abs(transform.a)
    dy = abs(transform.e)

    # gradients (z units per meter); multiply by z_factor to exaggerate relief
    gy, gx = np.gradient(arr * z_factor, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)

    shade = (np.cos(alt) * np.cos(slope)) + (np.sin(alt) * np.sin(slope) * np.cos(az - aspect))
    shade = np.clip(shade, 0, 1)
    return (shade * 255).astype("uint8")


@st.cache_data(show_spinner=False)
def _make_hillshade_png(dem_path: str, azimuth: float, altitude: float, z_factor: float, out_png: str) -> Tuple[str, list]:
    """Compute hillshade in native CRS, then reproject to EPSG:4326 and save as RGBA PNG with alpha."""
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        # mask invalids for hillshade calc
        valid = np.isfinite(arr)
        arr_filled = np.where(valid, arr, np.nanmean(arr))

        hs = _hillshade(arr_filled, src.transform, azimuth=azimuth, altitude=altitude, z_factor=z_factor)
        hs = np.where(valid, hs, 0).astype("uint8")  # set nodata to 0 (black/transparent later)

        # Reproject hillshade to EPSG:4326 for Leaflet overlay
        dst_crs = "EPSG:4326"
        transform_out, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        dst = np.zeros((height, width), dtype="uint8")
        reproject(
            source=hs,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_out,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        # Build RGBA image with transparency where no data (value==0 AND original was invalid)
        # Reproject validity mask too (nearest) to keep transparency correct
        dst_valid = np.zeros((height, width), dtype="uint8")
        reproject(
            source=valid.astype("uint8"),
            destination=dst_valid,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_out,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

        rgba = np.stack([dst, dst, dst, (dst_valid * 255)], axis=-1)  # grey with alpha=255 on valid, 0 else
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(out_png, "PNG")

        # Bounds in lat/lon for overlay
        south, west, north, east = array_bounds(height, width, transform_out)
        bounds_latlon = [[south, west], [north, east]]

    return out_png, bounds_latlon
# ------------------------------------------------------------------


# ---------------------------- UI ----------------------------
st.markdown(
    """
    <div style="padding: 14px 18px; border-radius: 14px; background: linear-gradient(135deg,#0ea5e9, #6366f1); color: white;">
      <h1 style="margin:0; font-size: 28px;">Survey Surface QA/QC</h1>
      <div style="opacity:.95; font-size: 15px; margin-top: 6px;">
        Build DEMs from GNSS/drone points, extract contours, compute cut/fill volumes, export LandXML (Civil 3D), and preview on a map.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.sidebar.radio(
    "Mode",
    ["Quick Demo (auto-generate)", "Upload CSV & Run", "Browse Existing Outputs"],
    help="Choose how you want to try the app.",
)

with st.sidebar:
    st.markdown("### Parameters")
    in_crs = st.text_input("Input CRS", value=DEFAULT_IN_CRS, help="EPSG:4326 for lon/lat")
    to_crs = st.text_input("Target CRS (metric)", value=DEFAULT_TO_CRS, help="Use a metric CRS, e.g., UTM zone")
    cell = st.slider("Cell size (m)", 1.0, 10.0, DEFAULT_CELL, 0.5)
    interval = st.slider("Contour interval (m)", 0.5, 5.0, DEFAULT_INTERVAL, 0.5)
    ref_z = st.number_input("Reference plane Z (m) for volumes", value=DEFAULT_REF_Z, step=1.0)

    st.markdown("---")
    st.markdown("### Hillshade overlay")
    hs_on = st.checkbox("Enable hillshade overlay", value=True)
    hs_az = st.slider("Azimuth (°)", 0, 360, 315, 1)
    hs_alt = st.slider("Altitude (°)", 1, 90, 45, 1)
    hs_opacity = st.slider("Overlay opacity", 0.1, 1.0, 0.6, 0.05)
    hs_z = st.slider("Z factor", 0.2, 5.0, 1.0, 0.1)

    st.markdown("---")
    st.caption("💡 For Riyadh, UTM Zone 38N = EPSG:32638")


# ---------------------------- LOGIC PER MODE ----------------------------
dem_path, contours_path, landxml_path, report_path = None, None, None, None
stats = {}

if mode == "Quick Demo (auto-generate)":
    st.subheader("Quick Demo")
    st.write("We’ll generate synthetic points near Riyadh, build a DEM, draw contours, compute volumes, and export LandXML — all in a few seconds.")
    run = st.button("▶️ Run Demo Now", type="primary")
    if run:
        # Synthetic CSV
        csv_path = DATA_DIR / "sample_points.csv"
        import csv, math, random
        random.seed(42)
        cx, cy = 46.675, 24.715
        pts = []
        for _ in range(1200):
            dx = (random.random() - 0.5) * 0.02
            dy = (random.random() - 0.5) * 0.02
            x = cx + dx
            y = cy + dy
            r = math.sqrt((dx * 111_000) ** 2 + (dy * 111_000) ** 2)
            z = 600 + 30 * math.cos(r / 220) + random.random() * 0.6
            pts.append((x, y, z))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["x", "y", "z"]); w.writerows(pts)

        with st.spinner("Gridding to DEM..."):
            dem_path, stats = _run_grid(str(csv_path), x="x", y="y", lon=None, lat=None, z_col="z",
                                        in_crs=in_crs, to_crs=to_crs, cell=cell)
        with st.spinner("Generating contours..."):
            contours_path = _run_contours(dem_path, interval)
        with st.spinner("Exporting LandXML..."):
            landxml_path = _run_landxml_from_csv(str(csv_path), x="x", y="y", lon=None, lat=None,
                                                 z_col="z", in_crs=in_crs, to_crs=to_crs)
        with st.spinner("Assembling QA/QC report..."):
            report_path = _run_report(dem_path, contours_path)

elif mode == "Upload CSV & Run":
    st.subheader("Upload CSV & Run")
    up = st.file_uploader("CSV with columns (x,y,z) or (lon,lat,z)", type=["csv"])
    colx, coly, colz = st.columns(3)
    with colx:
        x_col = st.text_input("X/Easting column (or leave blank)", value="x")
    with coly:
        y_col = st.text_input("Y/Northing column (or leave blank)", value="y")
    with colz:
        z_col = st.text_input("Elevation column", value="z")
    use_lonlat = st.checkbox("My CSV uses lon/lat instead of x/y", value=False)

    if up:
        saved = _save_uploaded_file(up, DATA_DIR / "uploaded_points.csv")
        run = st.button("▶️ Run Pipeline", type="primary")
        if run:
            with st.spinner("Gridding to DEM..."):
                dem_path, stats = _run_grid(
                    str(saved),
                    x=None if use_lonlat else (x_col or None),
                    y=None if use_lonlat else (y_col or None),
                    lon="lon" if use_lonlat else None,
                    lat="lat" if use_lonlat else None,
                    z_col=z_col,
                    in_crs=in_crs,
                    to_crs=to_crs,
                    cell=cell,
                )
            with st.spinner("Generating contours..."):
                contours_path = _run_contours(dem_path, interval)
            with st.spinner("Exporting LandXML..."):
                landxml_path = _run_landxml_from_csv(
                    str(saved),
                    x=None if use_lonlat else (x_col or None),
                    y=None if use_lonlat else (y_col or None),
                    lon="lon" if use_lonlat else None,
                    lat="lat" if use_lonlat else None,
                    z_col=z_col,
                    in_crs=in_crs,
                    to_crs=to_crs,
                )
            with st.spinner("Assembling QA/QC report..."):
                report_path = _run_report(dem_path, contours_path)

else:  # Browse Existing Outputs
    st.subheader("Browse Existing Outputs")
    dem_path = st.text_input("DEM path", value=str(OUT_DIR / "dem.tif"))
    contours_path = st.text_input("Contours path", value=str(OUT_DIR / "contours.geojson"))
    report_path = st.text_input("Report path", value=str(OUT_DIR / "report.html"))
    landxml_path = st.text_input("LandXML path", value=str(OUT_DIR / "surface_landxml.xml"))
    st.caption("Tip: These are defaults created by the demo/pipeline.")


# If nothing exists yet, nudge user
_auto_demo_if_empty()


# ---------------------------- PRESENTATION TABS ----------------------------
tabs = st.tabs(["🗺️ Map", "📊 Stats & Volumes", "⬇️ Downloads", "ℹ️ How it works"])

with tabs[0]:
    dem_bounds = None
    if dem_path and os.path.exists(dem_path):
        try:
            dem_bounds = _bounds_from_raster(dem_path)
        except Exception as e:
            st.warning(f"Could not read DEM bounds: {e}")

    # Hillshade overlay PNG (computed lazily if DEM exists and toggle is on)
    hs_png, hs_bounds = None, None
    if hs_on and dem_path and os.path.exists(dem_path):
        try:
            hs_png_path = str(OUT_DIR / "hillshade.png")
            hs_png, hs_bounds = _make_hillshade_png(dem_path, hs_az, hs_alt, hs_z, hs_png_path)
        except Exception as e:
            st.warning(f"Could not compute hillshade: {e}")

    m = _build_map(contours_path, dem_bounds, hs_png, hs_bounds, hs_opacity)
    st_folium(m, height=560, use_container_width=True)

with tabs[1]:
    st.subheader("DEM Statistics")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min (m)", f"{stats['min']:.2f}")
        c2.metric("Max (m)", f"{stats['max']:.2f}")
        c3.metric("Mean (m)", f"{stats['mean']:.2f}")
        c4.metric("Std (m)", f"{stats['std']:.2f}")
        st.caption(f"Size: {stats['rows']} x {stats['cols']} • CRS: {stats['crs']}")
    elif dem_path and os.path.exists(dem_path):
        try:
            with rasterio.open(dem_path) as src:
                arr = src.read(1)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Min (m)", f"{float(np.nanmin(arr)):.2f}")
                c2.metric("Max (m)", f"{float(np.nanmax(arr)):.2f}")
                c3.metric("Mean (m)", f"{float(np.nanmean(arr)):.2f}")
                c4.metric("Std (m)", f"{float(np.nanstd(arr)):.2f}")
                st.caption(f"Size: {arr.shape[0]} x {arr.shape[1]} • CRS: {src.crs}")
        except Exception as e:
            st.warning(f"Could not compute stats: {e}")
    else:
        st.info("Run the demo or upload a CSV to see statistics.")

    st.markdown("### Volumes vs Plane")
    if dem_path and os.path.exists(dem_path):
        try:
            res = volumes_against_plane(dem_path, ref_z)
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Cut (m³)", f"{res['cut_m3']:.0f}")
            cc2.metric("Fill (m³)", f"{res['fill_m3']:.0f}")
            cc3.metric("Net (m³)", f"{res['net_m3']:.0f}")
        except Exception as e:
            st.warning(f"Volume calc failed: {e}")
    else:
        st.caption("Volumes are available after a DEM is generated.")

with tabs[2]:
    st.subheader("Downloads")
    _download_button("Download DEM (GeoTIFF)", dem_path or "", "image/tiff")
    _download_button("Download Contours (GeoJSON)", contours_path or "", "application/geo+json")
    _download_button("Download QA/QC Report (HTML)", report_path or "", "text/html")
    _download_button("Download LandXML (TIN)", landxml_path or "", "application/xml")

    if not ((dem_path and os.path.exists(dem_path)) or (contours_path and os.path.exists(contours_path))):
        st.info("Run the demo or upload your CSV to generate files first.")

with tabs[3]:
    st.subheader("How it works")
    st.markdown(
        """
        1. **CSV → GeoDataFrame** (CRS injected)  
        2. **IDW gridding** in a **metric CRS** → DEM (GeoTIFF)  
        3. **Contours** via marching squares → GeoJSON  
        4. **Volumes** vs a reference plane (or another DEM)  
        5. **LandXML** export (Delaunay TIN) for Civil 3D  
        6. **Report**: HTML summary of raster stats  
        
        For production-scale datasets, swap gridding to PDAL/GDAL and optionally enforce breaklines for TINs.
        """
    )
