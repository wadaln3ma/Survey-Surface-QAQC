import xml.etree.ElementTree as ET
from typing import Optional
import numpy as np
from scipy.spatial import Delaunay
from .io import read_points_csv
import datetime

def landxml_from_points_csv(
    input_csv: str,
    x: str = None,
    y: str = None,
    lon: str = None,
    lat: str = None,
    z: str = 'z',
    in_crs: Optional[str] = None,
    to_crs: Optional[str] = None,
    surface_name: str = 'SurveyTIN'
) -> str:
    gdf = read_points_csv(input_csv, x=x, y=y, lon=lon, lat=lat, z=z, in_crs=in_crs, to=to_crs)
    X = gdf.geometry.x.values.astype(float)
    Y = gdf.geometry.y.values.astype(float)
    Z = gdf[z].values.astype(float)
    pts = np.vstack([X, Y]).T
    if len(pts) < 3:
        raise ValueError('At least 3 points are required for triangulation.')
    tri = Delaunay(pts)
    ns = {None: "http://www.landxml.org/schema/LandXML-1.2"}
    ET.register_namespace('', ns[None])
    root = ET.Element('LandXML', attrib={'version':'1.2','date':f"{datetime.datetime.utcnow().isoformat()}Z"})
    surfaces = ET.SubElement(root, 'Surfaces')
    surf = ET.SubElement(surfaces, 'Surface', attrib={'name': surface_name})
    defn = ET.SubElement(surf, 'Definition', attrib={'surfType':'TIN'})
    pnts = ET.SubElement(defn, 'Pnts')
    for i,(xv,yv,zv) in enumerate(zip(X,Y,Z), start=1):
        p = ET.SubElement(pnts, 'P', attrib={'id': str(i)})
        p.text = f"{xv} {yv} {zv}"
    faces = ET.SubElement(defn, 'Faces')
    for tri_idx in tri.simplices:
        i,j,k = (tri_idx[0]+1, tri_idx[1]+1, tri_idx[2]+1)
        f = ET.SubElement(faces, 'F')
        f.text = f"{i} {j} {k}"
    return ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')
