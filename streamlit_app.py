# streamlit_app.py
import os
import io
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image, ImageDraw, ImageFont  # PNG overlays + drawing
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Import your pipeline modules
from survey_surface.io import read_points_csv
from survey_surface.grid import idw_grid, write_geotiff
from survey_surface.contours import contours_geojson_from_raster
from survey_surface.report import make_report_html
from survey_surface.volumes import volumes_against_plane
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

def _reset_outputs():
    """Clear outputs/ and Streamlit caches, then rerun fresh."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st.cache_data.clear()
    # clear any stored scale/bounds
    for k in ["cr_vmin", "cr_vmax", "dem_bounds", "map_key"]:
        if k in st.session_state:
            del st.session_state[k]
    st.success("Outputs and cache cleared.")
    st.rerun()

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

def _focus_map_on_dem(dem_path: str):
    """Compute DEM bounds (EPSG:4326) and ask the Map tab to refocus."""
    try:
        bounds = _bounds_from_raster(dem_path)  # [[south,west],[north,east]]
        st.session_state["dem_bounds"] = bounds
        st.session_state["map_key"] = st.session_state.get("map_key", 0) + 1  # force re-render
        st.toast("Map zoomed to AOI.", icon="🔍")
    except Exception as e:
        st.warning(f"Could not compute map bounds to focus: {e}")

# ---------------------------- HILLSHADE ----------------------------
def _hillshade(arr: np.ndarray, transform, azimuth: float = 315.0, altitude: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    dx = abs(transform.a)
    dy = abs(transform.e)
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
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        valid = np.isfinite(arr)
        arr_filled = np.where(valid, arr, np.nanmean(arr))
        hs = _hillshade(arr_filled, src.transform, azimuth=azimuth, altitude=altitude, z_factor=z_factor)
        hs = np.where(valid, hs, 0).astype("uint8")

        dst_crs = "EPSG:4326"
        transform_out, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        dst = np.zeros((height, width), dtype="uint8")
        reproject(source=hs, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform_out, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)

        dst_valid = np.zeros((height, width), dtype="uint8")
        reproject(source=valid.astype("uint8"), destination=dst_valid,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform_out, dst_crs=dst_crs,
                  resampling=Resampling.nearest)

        rgba = np.stack([dst, dst, dst, (dst_valid * 255)], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(out_png, "PNG")
        south, west, north, east = array_bounds(height, width, transform_out)
        bounds_latlon = [[south, west], [north, east]]
    return out_png, bounds_latlon

# ---------------------------- COLOR RELIEF + SCALE ----------------------------
def _compute_vmin_vmax_from_dem(dem_path: str, vmin_mode: str, vmax_mode: str,
                                custom_min: Optional[float], custom_max: Optional[float]) -> Tuple[float, float]:
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        if vmin_mode == "auto" or custom_min is None:
            vmin = float(np.nanpercentile(arr, 2.0))
        else:
            vmin = float(custom_min)
        if vmax_mode == "auto" or custom_max is None:
            vmax = float(np.nanpercentile(arr, 98.0))
        else:
            vmax = float(custom_max)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    return vmin, vmax

@st.cache_data(show_spinner=False)
def _make_colorrelief_png(
    dem_path: str,
    cmap_name: str,
    vmin_mode: str,
    vmax_mode: str,
    custom_min: Optional[float],
    custom_max: Optional[float],
    out_png: str
) -> Tuple[str, list, float, float]:
    """
    Reproject DEM to EPSG:4326, normalize via vmin/vmax, colorize, export RGBA PNG + bounds,
    and RETURN the vmin/vmax actually used (for legend).
    """
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        valid = np.isfinite(arr)

        dst_crs = "EPSG:4326"
        transform_out, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        dst = np.full((height, width), np.nan, dtype="float64")
        reproject(source=arr, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform_out, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)

        dst_valid = np.zeros((height, width), dtype="uint8")
        reproject(source=valid.astype("uint8"), destination=dst_valid,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform_out, dst_crs=dst_crs,
                  resampling=Resampling.nearest)

        data = dst.copy()
        data[dst_valid == 0] = np.nan

        # decide vmin/vmax on the reprojected grid (consistent with display)
        if vmin_mode == "auto" or custom_min is None:
            vmin = float(np.nanpercentile(data, 2.0))
        else:
            vmin = float(custom_min)
        if vmax_mode == "auto" or custom_max is None:
            vmax = float(np.nanpercentile(data, 98.0))
        else:
            vmax = float(custom_max)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = cm.get_cmap(cmap_name)
        rgba_f = mapper(norm(np.where(np.isfinite(data), data, np.nan)))
        rgba_f[~np.isfinite(data)] = [0, 0, 0, 0]
        rgba = (rgba_f * 255).astype("uint8")
        Image.fromarray(rgba, mode="RGBA").save(out_png, "PNG")

        south, west, north, east = array_bounds(height, width, transform_out)
        bounds_latlon = [[south, west], [north, east]]

    return out_png, bounds_latlon, vmin, vmax

def _make_colorramp_legend_png(cmap_name: str, vmin: float, vmax: float,
                               out_png: str, width: int = 420, height: int = 80) -> str:
    """
    Create a horizontal color ramp legend PNG with vmin/vmax labels.
    """
    # Gradient (0..1 across width)
    x = np.linspace(0, 1, width)
    grad = np.tile(x, (height, 1))
    mapper = cm.get_cmap(cmap_name)
    rgba_f = mapper(grad)  # (H, W, 4), floats 0..1
    rgba = (rgba_f * 255).astype("uint8")
    img = Image.fromarray(rgba, mode="RGBA")

    # Border + labels
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, width-1, height-1], outline=(255, 255, 255, 180), width=1)

    # Try to use a common font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_b = ImageFont.load_default()

    # Title
    title = f"Elevation (m) – {cmap_name}"
    tw, th = draw.textsize(title, font=font_b)
    draw.rectangle([10, 6, 10 + tw + 8, 6 + th + 4], fill=(0, 0, 0, 90))
    draw.text((14, 8), title, fill=(255, 255, 255, 230), font=font_b)

    # Labels (vmin, mid, vmax)
    labels = [(0, f"{vmin:.2f}"), (width//2, f"{(vmin+vmax)/2:.2f}"), (width-1, f"{vmax:.2f}")]
    for x_pos, txt in labels:
        tw, th = draw.textsize(txt, font=font)
        bx = max(2, min(width - tw - 2, x_pos - tw // 2))
        by = height - th - 6
        draw.rectangle([bx-2, by-2, bx+tw+2, by+th+2], fill=(0, 0, 0, 90))
        draw.text((bx, by), txt, fill=(255, 255, 255, 230), font=font)

    img.save(out_png, "PNG")
    return out_png

# ---------------------------- COMPOSITE EXPORT ----------------------------
def _read_png_rgba(path: str) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    return im

def _apply_opacity(img: Image.Image, alpha: float) -> Image.Image:
    r, g, b, a = img.split()
    a = a.point(lambda v: int(v * float(alpha)))
    return Image.merge("RGBA", (r, g, b, a))

def _draw_contours_on(img: Image.Image, contours_geojson_path: str, bounds_latlon: list,
                      color: tuple = (255, 255, 255, 220), width: int = 2) -> Image.Image:
    if not os.path.exists(contours_geojson_path):
        return img
    with open(contours_geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    (south, west), (north, east) = bounds_latlon
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    def to_px(lat, lon):
        x = (lon - west) / (east - west) * W
        y = (1.0 - (lat - south) / (north - south)) * H
        return (x, y)
    def draw_coords(coords):
        pts = [to_px(lat=lat, lon=lon) for lon, lat in coords]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        t = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue
        if t == "LineString":
            draw_coords(coords)
        elif t == "MultiLineString":
            for part in coords:
                draw_coords(part)
    return img

def _export_composite_png(
    out_path: str,
    colorrelief_png: Optional[str],
    hillshade_png: Optional[str],
    contours_geojson: Optional[str],
    bounds_latlon: Optional[list],
    cr_opacity: float,
    hs_opacity: float
) -> Optional[str]:
    layers = []
    if colorrelief_png and os.path.exists(colorrelief_png):
        layers.append(("cr", colorrelief_png, cr_opacity))
    if hillshade_png and os.path.exists(hillshade_png):
        layers.append(("hs", hillshade_png, hs_opacity))
    if not layers:
        return None
    base_tag, base_path, base_opacity = layers[0]
    base_img = _apply_opacity(_read_png_rgba(base_path), base_opacity)
    for tag, pth, op in layers[1:]:
        img = _apply_opacity(_read_png_rgba(pth), op)
        if img.size != base_img.size:
            img = img.resize(base_img.size, Image.BILINEAR)
        base_img = Image.alpha_composite(base_img, img)
    if contours_geojson and bounds_latlon:
        base_img = _draw_contours_on(base_img, contours_geojson, bounds_latlon,
                                     color=(255, 255, 255, 235), width=2)
    base_img.save(out_path, "PNG")
    return out_path

# ---------------------------- MAP BUILDER ----------------------------
def _build_map(
    contours_path: Optional[str],
    bounds: Optional[list],
    basemap_choice: str,
    hybrid_labels: bool,
    colorrelief_png: Optional[str],
    colorrelief_bounds: Optional[list],
    cr_opacity: float,
    hillshade_png: Optional[str],
    hillshade_bounds: Optional[list],
    hs_opacity: float,
) -> folium.Map:
    center = [24.7136, 46.6753]
    m = folium.Map(location=center, zoom_start=12, control_scale=True)
    base_defs = {
        "OpenStreetMap": dict(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="© OpenStreetMap contributors",
        ),
        "Topographic": dict(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr=("Tiles © Esri — Sources: Esri, Garmin, USGS, NOAA, FAO, NPS, NRCan, GeoBase, "
                  "IGN, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), "
                  "OpenStreetMap contributors, and the GIS User Community"),
        ),
        "Satellite": dict(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        ),
    }
    for name, cfg in base_defs.items():
        folium.TileLayer(
            tiles=cfg["tiles"],
            attr=cfg["attr"],
            name=name,
            control=False,  # use sidebar picker instead
            show=(name == basemap_choice),
        ).add_to(m)

    # Optional labels overlay for 'Hybrid' with Satellite
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Labels © Esri",
        name="Hybrid Labels",
        control=True,
        overlay=True,
        show=(hybrid_labels and basemap_choice == "Satellite"),
    ).add_to(m)

    # Overlays
    if colorrelief_png and os.path.exists(colorrelief_png) and colorrelief_bounds:
        folium.raster_layers.ImageOverlay(
            name="Color Relief",
            image=colorrelief_png,
            bounds=colorrelief_bounds,
            opacity=cr_opacity,
            interactive=False,
            cross_origin=False,
            zindex=300,
        ).add_to(m)

    if hillshade_png and os.path.exists(hillshade_png) and hillshade_bounds:
        folium.raster_layers.ImageOverlay(
            name="Hillshade",
            image=hillshade_png,
            bounds=hillshade_bounds,
            opacity=hs_opacity,
            interactive=False,
            cross_origin=False,
            zindex=400,
        ).add_to(m)

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
    have_any = any((OUT_DIR / p).exists() for p in ["dem.tif", "contours.geojson", "report.html"])
    if not have_any:
        st.info("No outputs found yet. Run the **Quick Demo** or upload your CSV to try the pipeline.")
    return have_any

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
    st.markdown("### Overlays")
    cr_on = st.checkbox("Enable Color Relief", value=True)
    cr_cmap = st.selectbox("Colormap", ["viridis", "terrain", "plasma", "magma", "cividis"], index=0)
    cr_opacity = st.slider("Color Relief opacity", 0.1, 1.0, 0.85, 0.05)
    col_min, col_max = st.columns(2)
    with col_min:
        cr_min_mode = st.selectbox("Min scale", ["auto", "custom"], index=0)
        cr_min_val = st.number_input("Custom min (m)", value=0.0, step=1.0, disabled=(cr_min_mode=="auto"))
    with col_max:
        cr_max_mode = st.selectbox("Max scale", ["auto", "custom"], index=0)
        cr_max_val = st.number_input("Custom max (m)", value=1000.0, step=1.0, disabled=(cr_max_mode=="auto"))

    st.markdown("—")
    hs_on = st.checkbox("Enable Hillshade", value=True)
    hs_az = st.slider("Hillshade azimuth (°)", 0, 360, 315, 1, disabled=not hs_on)
    hs_alt = st.slider("Hillshade altitude (°)", 1, 90, 45, 1, disabled=not hs_on)
    hs_opacity = st.slider("Hillshade opacity", 0.1, 1.0, 0.6, 0.05, disabled=not hs_on)
    hs_z = st.slider("Hillshade Z factor", 0.2, 5.0, 1.0, disabled=not hs_on)

    st.markdown("---")
    st.markdown("### Basemap")
    basemap_choice = st.radio("Base", ["OpenStreetMap", "Topographic", "Satellite"], index=2, horizontal=True)
    hybrid_labels = st.checkbox("Hybrid labels (with Satellite)", value=True)

    st.markdown("---")
    st.markdown("### Maintenance")
    if st.button("♻️ Reset outputs & cache", help="Clears outputs/ and cache, then reloads the app."):
        _reset_outputs()

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

        # Auto-focus the map to the DEM AOI
        _focus_map_on_dem(dem_path)

elif mode == "Upload CSV & Run":
    st.subheader("Upload CSV & Run")
    up = st.file_uploader("CSV with columns (x,y,z) or (lon,lat,z)", type=["csv"])
    colx, coly, colz = st.columns(3)
    with colx: x_col = st.text_input("X/Easting column (or leave blank)", value="x")
    with coly: y_col = st.text_input("Y/Northing column (or leave blank)", value="y")
    with colz: z_col = st.text_input("Elevation column", value="z")
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

            # Auto-focus the map to the DEM AOI
            _focus_map_on_dem(dem_path)

else:
    st.subheader("Browse Existing Outputs")
    dem_path = st.text_input("DEM path", value=str(OUT_DIR / "dem.tif"))
    contours_path = st.text_input("Contours path", value=str(OUT_DIR / "contours.geojson"))
    report_path = st.text_input("Report path", value=str(OUT_DIR / "report.html"))
    landxml_path = st.text_input("LandXML path", value=str(OUT_DIR / "surface_landxml.xml"))
    st.caption("Tip: These are defaults created by the demo/pipeline.")

# If nothing exists yet, nudge user
_ = _auto_demo_if_empty()

# ---------------------------- PRESENTATION TABS ----------------------------
tabs = st.tabs(["🗺️ Map", "📊 Stats & Volumes", "⬇️ Downloads", "ℹ️ How it works"])

with tabs[0]:
    # Prefer bounds saved by pipeline; fallback to reading from file
    dem_bounds = st.session_state.get("dem_bounds")
    if (dem_bounds is None) and dem_path and os.path.exists(dem_path):
        try:
            dem_bounds = _bounds_from_raster(dem_path)
        except Exception as e:
            st.warning(f"Could not read DEM bounds: {e}")

    # Build overlays
    cr_png, cr_bounds = None, None
    cr_vmin, cr_vmax = None, None
    if "cr_on" in st.session_state:
        pass  # no-op; quiet linters
    if (st.session_state.get("cr_on", True) or True) and st.session_state.get("cr_on", True) and dem_path and os.path.exists(dem_path):
        # respect the control state (cr_on) from the sidebar
        if st.session_state.get("cr_on", True):
            try:
                cr_png_path = str(OUT_DIR / "color_relief.png")
                cr_png, cr_bounds, cr_vmin, cr_vmax = _make_colorrelief_png(
                    dem_path,
                    cmap_name=st.session_state.get("cr_cmap", "viridis") if "cr_cmap" in st.session_state else "viridis",
                    vmin_mode=st.session_state.get("cr_min_mode", "auto"),
                    vmax_mode=st.session_state.get("cr_max_mode", "auto"),
                    custom_min=st.session_state.get("cr_min_val") if st.session_state.get("cr_min_mode") == "custom" else None,
                    custom_max=st.session_state.get("cr_max_val") if st.session_state.get("cr_max_mode") == "custom" else None,
                    out_png=cr_png_path,
                )
                st.session_state["cr_vmin"] = cr_vmin
                st.session_state["cr_vmax"] = cr_vmax
            except Exception as e:
                st.warning(f"Could not compute color relief: {e}")

    # Hillshade
    hs_png, hs_bounds = None, None
    if st.session_state.get("hs_on", True) and dem_path and os.path.exists(dem_path):
        try:
            hs_png_path = str(OUT_DIR / "hillshade.png")
            hs_png, hs_bounds = _make_hillshade_png(
                dem_path,
                st.session_state.get("hs_az", 315),
                st.session_state.get("hs_alt", 45),
                st.session_state.get("hs_z", 1.0),
                hs_png_path,
            )
        except Exception as e:
            st.warning(f"Could not compute hillshade: {e}")

    # Build map
    m = _build_map(
        contours_path=contours_path,
        bounds=dem_bounds,
        basemap_choice=st.session_state.get("basemap_choice", "Satellite") if "basemap_choice" in st.session_state else "Satellite",
        hybrid_labels=st.session_state.get("hybrid_labels", True) if "hybrid_labels" in st.session_state else True,
        colorrelief_png=cr_png,
        colorrelief_bounds=cr_bounds,
        cr_opacity=st.session_state.get("cr_opacity", 0.85) if "cr_opacity" in st.session_state else 0.85,
        hillshade_png=hs_png,
        hillshade_bounds=hs_bounds,
        hs_opacity=st.session_state.get("hs_opacity", 0.6) if "hs_opacity" in st.session_state else 0.6,
    )

    # Manual zoom button
    colz1, colz2 = st.columns([1, 6])
    with colz1:
        if st.button("🔍 Zoom to AOI", disabled=(st.session_state.get("dem_bounds") is None)):
            st.session_state["map_key"] = st.session_state.get("map_key", 0) + 1
            st.rerun()

    st_folium(m, height=560, use_container_width=True, key=f"map-{st.session_state.get('map_key',0)}")

with tabs[1]:
    st.subheader("DEM Statistics")
    if 'stats' in locals() and stats:
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

    st.markdown("—")
    st.markdown("**Map & Legend export (overlays only)**")
    if st.button("🖼️ Compose overlays to PNG + legend"):
        # Ensure overlays exist with current params; reuse earlier if created
        cr_png_path = str(OUT_DIR / "color_relief.png") if os.path.exists(OUT_DIR / "color_relief.png") else None
        hs_png_path = str(OUT_DIR / "hillshade.png") if os.path.exists(OUT_DIR / "hillshade.png") else None

        # If missing, try to create from DEM
        if (cr_png_path is None) and dem_path and os.path.exists(dem_path):
            try:
                cr_png_path, cr_bounds_tmp, vmin_used, vmax_used = _make_colorrelief_png(
                    dem_path,
                    cmap_name=st.session_state.get("cr_cmap", "viridis"),
                    vmin_mode=st.session_state.get("cr_min_mode", "auto"),
                    vmax_mode=st.session_state.get("cr_max_mode", "auto"),
                    custom_min=st.session_state.get("cr_min_val") if st.session_state.get("cr_min_mode") == "custom" else None,
                    custom_max=st.session_state.get("cr_max_val") if st.session_state.get("cr_max_mode") == "custom" else None,
                    out_png=str(OUT_DIR / "color_relief.png"),
                )
                cr_bounds = cr_bounds_tmp
                st.session_state["cr_vmin"] = vmin_used
                st.session_state["cr_vmax"] = vmax_used
            except Exception as e:
                st.error(f"Failed to compute color relief: {e}")

        if (hs_png_path is None) and dem_path and os.path.exists(dem_path):
            try:
                hs_png_path, hs_bounds_tmp = _make_hillshade_png(
                    dem_path,
                    st.session_state.get("hs_az", 315),
                    st.session_state.get("hs_alt", 45),
                    st.session_state.get("hs_z", 1.0),
                    str(OUT_DIR / "hillshade.png"),
                )
                hs_bounds = hs_bounds_tmp
            except Exception as e:
                st.error(f"Failed to compute hillshade: {e}")

        # Choose bounds (prefer color relief bounds; else hillshade)
        bounds_latlon = st.session_state.get("dem_bounds") or cr_bounds or hs_bounds or None

        out_png = str(OUT_DIR / "map_overlays_composite.png")
        saved = _export_composite_png(
            out_path=out_png,
            colorrelief_png=cr_png_path,
            hillshade_png=hs_png_path,
            contours_geojson=contours_path if contours_path and os.path.exists(contours_path) else None,
            bounds_latlon=bounds_latlon,
            cr_opacity=st.session_state.get("cr_opacity", 0.85),
            hs_opacity=st.session_state.get("hs_opacity", 0.6),
        )
        if saved:
            st.success("Composite created.")
            _download_button("Download Map (PNG, overlays only)", saved, "image/png")
        else:
            st.info("Nothing to compose yet. Generate overlays first (run demo or upload CSV).")

        # Legend PNG
        legend_path = str(OUT_DIR / "color_relief_legend.png")
        vmin = st.session_state.get("cr_vmin")
        vmax = st.session_state.get("cr_vmax")
        if (vmin is None) or (vmax is None):
            vmin, vmax = _compute_vmin_vmax_from_dem(
                dem_path,
                st.session_state.get("cr_min_mode", "auto"),
                st.session_state.get("cr_max_mode", "auto"),
                st.session_state.get("cr_min_val") if st.session_state.get("cr_min_mode") == "custom" else None,
                st.session_state.get("cr_max_val") if st.session_state.get("cr_max_mode") == "custom" else None,
            )
        try:
            legend_saved = _make_colorramp_legend_png(st.session_state.get("cr_cmap", "viridis"), vmin, vmax, legend_path)
            _download_button("Download Legend (PNG)", legend_saved, "image/png")
        except Exception as e:
            st.warning(f"Legend generation failed: {e}")

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
        
        **PNG export** composes **Color Relief + Hillshade + Contours** (no basemap) for a clean, shareable image.  
        A **Legend PNG** is generated using the same colormap and scale as the overlay.
        """
    )

# ---------------------------- FOOTER ATTRIBUTION ----------------------------
st.markdown(
    """
    <div style="margin-top:8px; font-size:12px; opacity:.75;">
      Basemaps © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &nbsp;•&nbsp;
      Esri World Topo/Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community
    </div>
    """,
    unsafe_allow_html=True,
)
