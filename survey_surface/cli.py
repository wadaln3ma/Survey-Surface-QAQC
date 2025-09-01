import json
from pathlib import Path
import numpy as np
import typer
from rich import print
from .io import read_points_csv
from .grid import idw_grid, write_geotiff
from .contours import contours_geojson_from_raster
from .report import make_report_html
from .volumes import volumes_against_plane, volumes_against_dem
from .landxml import landxml_from_points_csv

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def build_grid(
    input: Path = typer.Option(..., exists=True, help='Input CSV'),
    x: str = typer.Option(None, help='X/Easting column'),
    y: str = typer.Option(None, help='Y/Northing column'),
    lon: str = typer.Option(None, help='Longitude column'),
    lat: str = typer.Option(None, help='Latitude column'),
    z: str = typer.Option('z', help='Elevation column name'),
    in_crs: str = typer.Option(None, help='Input CRS, e.g., EPSG:4326'),
    to_crs: str = typer.Option('EPSG:32638', help='Target metric CRS'),
    cell: float = typer.Option(2.0, help='Cell size (m)'),
    out: Path = typer.Option(..., help='Output GeoTIFF path'),
):
    """Interpolate points to a DEM GeoTIFF via IDW."""
    gdf = read_points_csv(str(input), x=x, y=y, lon=lon, lat=lat, z=z, in_crs=in_crs, to=to_crs)
    pts = np.vstack([gdf.geometry.x.values, gdf.geometry.y.values]).T
    vals = gdf[z].astype(float).values
    arr, transform = idw_grid(pts, vals, cell=cell)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(str(out), arr, transform, to_crs)
    print(f'[green]Wrote DEM[/green]: {out}')

@app.command()
def contours(
    raster: Path = typer.Option(..., exists=True, help='Input DEM GeoTIFF'),
    interval: float = typer.Option(1.0, help='Contour interval (same units as DEM)'),
    out: Path = typer.Option(..., help='Output GeoJSON path'),
):
    """Generate GeoJSON contours from a DEM."""
    out.parent.mkdir(parents=True, exist_ok=True)
    contours_geojson_from_raster(str(raster), interval, str(out))
    print(f'[green]Wrote contours[/green]: {out}')

@app.command()
def report(
    dem: Path = typer.Option(..., exists=True, help='DEM GeoTIFF'),
    contours: Path = typer.Option(None, help='Contours GeoJSON (optional)'),
    out: Path = typer.Option(..., help='Output HTML report'),
):
    """Build a minimal HTML QA/QC report."""
    out.parent.mkdir(parents=True, exist_ok=True)
    html = make_report_html(str(dem), str(contours) if contours else None)
    out.write_text(html, encoding='utf-8')
    print(f'[green]Wrote report[/green]: {out}')

@app.command()
def volumes(
    dem: Path = typer.Option(..., exists=True, help='DEM GeoTIFF'),
    ref_z: float = typer.Option(None, help='Reference elevation (meters)'),
    ref_dem: Path = typer.Option(None, help='Reference DEM GeoTIFF (resampled to DEM)'),
):
    """Compute cut/fill volumes (m3) vs reference plane or DEM."""
    if ref_dem and ref_z is not None:
        print('[yellow]Both --ref-dem and --ref-z provided; using --ref-dem[/yellow]')
    if ref_dem:
        res = volumes_against_dem(str(dem), str(ref_dem))
    elif ref_z is not None:
        res = volumes_against_plane(str(dem), float(ref_z))
    else:
        raise typer.BadParameter('Provide either --ref-z or --ref-dem')
    print(json.dumps(res, indent=2))

@app.command()
def landxml(
    input: Path = typer.Option(..., exists=True, help='Input CSV of points'),
    x: str = typer.Option(None, help='X/Easting column'),
    y: str = typer.Option(None, help='Y/Northing column'),
    lon: str = typer.Option(None, help='Longitude column'),
    lat: str = typer.Option(None, help='Latitude column'),
    z: str = typer.Option('z', help='Elevation column name'),
    in_crs: str = typer.Option(None, help='Input CRS, e.g., EPSG:4326'),
    to_crs: str = typer.Option('EPSG:32638', help='Target metric CRS'),
    out: Path = typer.Option(..., help='Output LandXML path'),
    name: str = typer.Option('SurveyTIN', help='Surface name'),
):
    """Export a minimal LandXML surface (TIN) from points via Delaunay triangulation."""
    xml = landxml_from_points_csv(
        str(input), x=x, y=y, lon=lon, lat=lat, z=z, in_crs=in_crs, to_crs=to_crs, surface_name=name
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding='utf-8')
    print(f'[green]Wrote LandXML[/green]: {out}')

if __name__ == '__main__':
    app()
