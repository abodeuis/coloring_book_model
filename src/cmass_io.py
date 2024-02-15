import os
import json
import logging
import numpy as np
import rasterio
import geopandas as gpd
import multiprocessing
from typing import List

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.cmass_types import CMASS_Map, CMASS_Legend, CMASS_Feature

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

### Uncharted Transform Attempt 1
# Uncharted is using python 3.10 for pickling
# import sklearn #1.4.0
# import pickle
# from typing import Tuple, List

# def pixel_to_lonlat_pts(
#     xy_pts: List[Tuple[float, float]], geop_transform_tuple: Tuple
# ) -> List[Tuple[float, float]]:
#     polyregx_model, polyregx, polyregy_model, polyregy = geop_transform_tuple
#     lons = polyregx_model.predict(polyregx.fit_transform(np.array(xy_pts)))
#     lats = polyregy_model.predict(polyregy.fit_transform(np.array(xy_pts)))
#     lonslats = [(x_w, y_w) for x_w, y_w in zip(lons, lats)]
#     return lonslats

# def loadTransform(filepath : str): # -> ???
#     with open(filepath, 'rb') as fh:
#         transform = pickle.load(fh)
#     return transform

# def applyTransform(transform, image):
#     return transform.predict_xy_pts(image)

# from functools import partial
# filepath = 'uncharted_something.pkl'
# trans_data = loadTransform(filepath)
# partial_func = partial(pixel_to_lonlat_pts, geop_transform_tuple=trans_data)
### End Uncharted Transform Attempt 1

### Uncharted Transform Attempt 2
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
import pandas as pd

def build_geo_ref(map_name, gcp_csv_path):
    # Default to WGS84
    crs = CRS.from_epsg(4326)
    # Possibly switch to NAD83? epsg num : 4269
    df = pd.read_csv(gcp_csv_path)
    map_gcps = df[df['raster_id'] == map_name]
    rasterio_gcps = []
    for _, row in map_gcps.iterrows():
        rasterio_gcps.append(GroundControlPoint(row['row'], row['col'], row['NAD83_x'], row['NAD83_y']))
        
    transform = from_gcps(rasterio_gcps)
    return crs, transform

### End Uncharted Transform Attempt 2

def loadCMASSMap(image_path : Path, legend_path : Path=None, layout_path : Path=None) -> CMASS_Map:
    """Load a the data for a CMASS map. If legend_path or layout_path are provided they will set their respective
       attributes in the CMASS_Map class. Returns a CMASS_Map object."""
    map_name = os.path.basename(os.path.splitext(image_path)[0])
    
    # Start Threads
    with ThreadPoolExecutor() as executor:
        img_future = executor.submit(loadGeoTiff, image_path)
        if legend_path is not None:
            lgd_future = executor.submit(loadCMASSLegend, legend_path)
        if layout_path is not None:
            lay_future = executor.submit(loadLayoutJson, layout_path)
        
        image, crs, transform = img_future.result()
        if legend_path is not None:
            legend = lgd_future.result()
        if layout_path is not None:
            layout = lay_future.result()

    map_data = CMASS_Map(map_name, image, crs, transform)
    if legend_path is not None:
        map_data.legend = legend
    if layout_path is not None:
        if 'map' in layout:
            map_data.map_contour = layout['map']['bounds']
        if 'legend_polygons' in layout:
            map_data.legend_contour = layout['legend_polygons']['bounds']

    return map_data

def parallelLoadCMASSMaps(map_files, legend_path=None, layout_path=None, processes : int=multiprocessing.cpu_count()):
    """Load a list of maps in parallel with N processes. Returns a list of CMASS_Map objects"""
    map_args = []
    for file in map_files:
        map_name = os.path.basename(os.path.splitext(file)[0])
        lgd_file = None
        if legend_path is not None:
            lgd_file = os.path.join(legend_path, f'{map_name}.json')
            if not os.path.exists(lgd_file):
                lgd_file = None
        lay_file = None
        if layout_path is not None:
            lay_file = os.path.join(layout_path, f'{map_name}.json')
            if not os.path.exists(lay_file):
                lay_file = None
        map_args.append((file, lgd_file, lay_file))

    with multiprocessing.Pool(processes) as p:
        results = p.starmap(loadCMASSMap, map_args)

    return results

def loadCMASSLegend(filepath : Path, feature_type : str='all') -> CMASS_Legend:
    """Load a legend json file. Json is expected to be in USGS format. Converts shape point data to int. Supports
       filtering by feature type. Returns a dictionary"""
    # Check that feature_type is valid
    valid_ftype = ['point','line','polygon','all']
    if feature_type not in valid_ftype:
        msg = f'Invalid feature type "{feature_type}" specified.\nAvailable feature types are : {valid_ftype}'
        raise TypeError(msg)
    
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    # Set feature type filter
    if feature_type == 'point':
        feature_type = ['pt']
    if feature_type == 'polygon':
        feature_type = ['poly']
    if feature_type == 'line':
        feature_type = ['line']
    if feature_type == 'all':
        feature_type = ['pt','poly','line']

    # Convert to CMASS_Legend struct
    features = {}
    for f in json_data['shapes']:
        f_type = f['label'].split('_')[-1]
        # Filter unwanted feature types
        if f_type not in feature_type:
            continue
        # Convert pix coords to int
        f['points'] = np.array(f['points']).astype(int)

        features[f['label']] = CMASS_Feature(f['label'], type=f_type, contour=f['points'], contour_type=f['shape_type'])

    legend_data = CMASS_Legend(features, origin='USGS Json')

    return legend_data

def parallelLoadLegends(legend_files, feature_type : str='all', threads : int=32):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        legends = {}
        for file in legend_files:
            map_name = os.path.basename(os.path.splitext(file)[0])
            legends[map_name] = executor.submit(loadCMASSLegend, file, feature_type).result()
    return legends

#     // Q: What order does tensorflow expect a batch to be in?
#     // A: (batch, height, width, channels)
#     # BHWC

#     // q: What order does pytorch expect a batch to be in?
#     // A: (batch, channels, height, width) 
#     # BCHW

#     // q: What order does rasterio expect a batch to be in?
#     // A: (bands, height, width)
#     # CHW

def loadGeoTiff(filepath : Path):
    """Load a GeoTiff file. Image is in CHW format. Raises exception if image is not loaded properly. Returns a tuple of the image, crs and transform """
    with rasterio.open(filepath) as fh:
        image = fh.read()
        crs = fh.crs
        transform = fh.transform
    if image is None:
        msg = f'Unknown issue caused "{filepath}" to fail while loading'
        raise Exception(msg)
    
    return image, crs, transform

def parallelLoadGeoTiffs(files : List, processes : int=multiprocessing.cpu_count()): # -> list[tuple(image, crs, transfrom)]:
    """Load a list of filenames in parallel with N processes. Returns a list of images"""
    p=multiprocessing.Pool(processes=processes)
    with multiprocessing.Pool(processes) as p:
        images = p.map(loadGeoTiff, files)

    return images

def loadLegendJson(filepath : str, feature_type : str='all') -> dict:
    """Load a legend json file. Json is expected to be in USGS format. Converts shape point data to int. Supports
       filtering by feature type. Returns a dictionary"""
    # Check that feature_type is valid
    valid_ftype = ['point','polygon','all']
    if feature_type not in valid_ftype:
        msg = f'Invalid feature type "{feature_type}" specified.\nAvailable feature types are : {valid_ftype}'
        raise TypeError(msg)
    
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    # Filter by feature type
    if feature_type == 'point':
        json_data['shapes'] = [s for s in json_data['shapes'] if s['label'].split('_')[-1] == 'pt']
    if feature_type == 'polygon':
        json_data['shapes'] = [s for s in json_data['shapes'] if s['label'].split('_')[-1] == 'poly']

    # Convert pix coords to int
    for feature in json_data['shapes']:
        feature['points'] = np.array(feature['points']).astype(int)

    return json_data


def loadLayoutJson(filepath : str) -> dict:
    """Loads a layout json file. Json is expected to be in uncharted format. Converts bounding point data to int.
       Returns a dictionary"""
    with open(filepath, 'r') as fh:
        formatted_json = {}
        for line in fh:
            raw_json = json.loads(line)
            formatted_json[raw_json['model']['field']] = {'bounds' : np.array(raw_json['bounds']).astype(int)}
    return formatted_json

def old_loadLayoutJson(filepath : str) -> dict:
    """Loads a layout json file. Json is expected to be in uncharted format. Converts bounding point data to int.
       Returns a dictionary"""
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    formated_json = {}
    for section in json_data:
        # Convert pix coords to correct format
        section['bounds'] = np.array(section['bounds']).astype(int)
        formated_json[section['name']] = section
        
    return formated_json

def parallelLoadLayouts(layout_files, threads : int=32):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        layouts = {}
        for file in layout_files:
            map_name = os.path.basename(os.path.splitext(file)[0])
            layouts[map_name] = executor.submit(loadLayoutJson, file).result()
    return layouts

def saveGeoTiff(filename, prediction, crs, transform):
    """
    Save the prediction results to a specified filename.

    Parameters:
    - prediction: The prediction result (should be a 2D or 3D numpy array).
    - crs: The projection of the prediction.
    - transform: The transform of the prediction.
    - filename: The name of the file to save the prediction to.
    """
    
    image = np.array(prediction[...], ndmin=3)
    with rasterio.open(filename, 'w', driver='GTiff', compress='lzw', height=image.shape[1], width=image.shape[2],
                       count=image.shape[0], dtype=image.dtype, crs=crs, transform=transform) as fh:
        fh.write(image)

def saveCMASSLegend(filepath : str, legend : CMASS_Legend) -> None:
    json_data = {'shapes' : []}
    for label, f in legend.features.items():
        json_data['shapes'].append({'label' : f.name, 'points' : f.contour.tolist()})
    with open(filepath, 'w') as fh:
        fh.write(json.dumps(json_data))

def saveLegendJson(filepath : str, features : dict) -> None:
    """Save legend data to a json file. Features is expected to conform to the USGS format."""
    for s in features['shapes']:
        s['points'] = s['points'].tolist()
    with open(filepath, 'w') as fh:
        fh.write(json.dumps(features))

def saveGeopackage(geoDataFrame, filename, layer=None, filetype='geopackage'):
    SUPPORTED_FILETYPES = ['json', 'geojson','geopackage']

    if filetype not in SUPPORTED_FILETYPES:
        log.error(f'ERROR : Cannot export data to unsupported filetype "{filetype}". Supported formats are {SUPPORTED_FILETYPES}')
        return # Could raise exception but just skipping for now.
    
    # GeoJson
    if filetype in ['json', 'geojson']:
        if os.path.splitext(filename)[1] not in ['.json','.geojson']:
            filename += '.geojson'
        geoDataFrame.to_crs('EPSG:4326')
        geoDataFrame.to_file(filename, driver='GeoJSON')

    # GeoPackage
    elif filetype == 'geopackage':
        if os.path.splitext(filename)[1] != '.gpkg':
            filename += '.gpkg'
        geoDataFrame.to_file(filename, layer=layer, driver="GPKG")