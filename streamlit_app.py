import json
import numpy as np
import rasterio
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title='Survey Surface Viewer', layout='wide')

st.title('Survey Surface Viewer')
st.caption('Contours + DEM summary. Drop in your own files if you like.')

col1, col2 = st.columns([1,1])

with col1:
    dem_path = st.text_input('DEM GeoTIFF path', 'outputs/dem.tif')
    contours_path = st.text_input('Contours GeoJSON path', 'outputs/contours.geojson')

    stats = {}
    center = [24.7136, 46.6753]
    zoom = 12
    bounds = None

    if dem_path and dem_path.strip():
        try:
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
                    'cols': int(arr.shape[1]),
                    'crs': str(src.crs),
                }
                b = src.bounds
                center = [(b.top + b.bottom)/2, (b.left + b.right)/2]
                bounds = [[b.bottom, b.left], [b.top, b.right]]
                zoom = 13
        except Exception as e:
            st.warning(f'Could not read DEM: {e}')

    with st.expander('DEM Statistics', expanded=True if stats else False):
        if stats:
            st.json(stats)
        else:
            st.write('No DEM stats yet.')

with col2:
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True, tiles='OpenStreetMap')
    if contours_path and contours_path.strip():
        try:
            with open(contours_path, 'r', encoding='utf-8') as f:
                gj = json.load(f)
            folium.GeoJson(gj, name='Contours', show=True, tooltip=folium.GeoJsonTooltip(fields=['elev'], aliases=['Elev']))                  .add_to(m)
        except Exception as e:
            st.warning(f'Could not read contours: {e}')
    if bounds:
        m.fit_bounds(bounds, padding=(10,10))
    st_folium(m, height=520, use_container_width=True)

st.info('Tip: Generate outputs with the CLI first (DEM + contours), then load them here.')
