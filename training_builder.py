import warnings # For pandas deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os
import cv2
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as fx
from tqdm import tqdm
from shapely.geometry import Polygon, LinearRing
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor

import src.cmass_io as io
import src.utils as utils

MASK_COLOR = 1
BORDER_THICKNESS = 1

def parse_command_line():
    from typing import List
    def parse_directory(path : str) -> str:
        """Command line argument parser for directory path arguments. Raises argument error if the path does not exist
           or if it is not a valid directory. Returns directory path"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist\n'
            raise argparse.ArgumentTypeError(msg)
        # Check if its a directory
        if not os.path.isdir(path):
            msg = f'Invalid path "{path}" specified : Path is not a directory\n'
            raise argparse.ArgumentTypeError(msg)
        return path
    
    def parse_data(path: str) -> List[str]:
        """Command line argument parser for --data. --data should accept a list of file and/or directory paths as an
           input. This function is run called on each individual element of that list and checks if the path is valid
           and if the path is a directory expands it to all the valid files paths inside the dir. Returns a list of 
           valid files. This is intended to be used in conjunction with the post_parse_data function"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            #log.warning(msg)
            return None
            #raise argparse.ArgumentTypeError(msg+'\n')
        # Check if its a directory
        if os.path.isdir(path):
            data_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')]
            #if len(data_files) == 0:
                #log.warning(f'Invalid path "{path}" specified : Directory does not contain any .tif files')
        if os.path.isfile(path):
            data_files = [path]
        return data_files
    
    def post_parse_data(data : List[List[str]]) -> List[str]:
        """Cleans up the output of parse data from a list of lists to a single list and does validity checks for the 
           data as a whole. Returns a list of valid files. Raises an argument exception if no valid files were given"""
        # Check that there is at least 1 valid map to run on
        data_files = [file for sublist in data if sublist is not None for file in sublist]
        if len(data_files) == 0:
            msg = f'No valid files where given to --data argument. --data should be given a path or paths to file(s) \
                    and/or directory(s) containing the data to perform inference on. program will only run on .tif files'
            raise argparse.ArgumentTypeError(msg)
        return data_files
    
    def parse_data_type(data_type: str) -> str:
        """Command line argument parser for --type. Checks if the data type is valid. Returns the data type"""
        if data_type not in ['raster', 'vector']:
            msg = f'Invalid data type "{data_type}" specified : Data type must be one of "raster" or "vector"'
            raise argparse.ArgumentTypeError(msg)
        return data_type

    def parse_export_type(export_type: str) -> str:
        """Command line argument parser for --export_type. Checks if the export type is valid. Returns the export type"""
        if export_type not in ['json', 'geojson', 'geopackage']:
            msg = f'Invalid export type "{export_type}" specified : Export type must be one of "json", "geojson", or "geopackage"'
            raise argparse.ArgumentTypeError(msg)
        return export_type
    
    parser = argparse.ArgumentParser(description='', add_help=False)
    # Required Arguments
    required_args = parser.add_argument_group('required arguments', 'These are the arguments the script requires to \
                                               run.')
    required_args.add_argument('--data', 
                        type=parse_data,
                        nargs='+',
                        help='Path to file(s) and/or directory(s) containing the data to perform inference on. The \
                              program will run on any *_poly.tif files.')           
    
    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('-o', '--output',
                        default='training_data',
                        help='Directory to write the training data to. Defaults to "training_data"')
    optional_args.add_argument('-l', '--log',
                        default='logs/Latest.log',
                        help='Option to set where the file logging will output to. Defaults to "logs/Latest.log"')
    optional_args.add_argument('-t', '--type',
                        default='vector',
                        type=parse_data_type,
                        help='Option to set the type of training data to build. Defaults to "vector"')
    optional_args.add_argument('-e', '--export_type',
                        default='geopackage',
                        type=parse_export_type,
                        help='Option to set the type of training data to export. Defaults to "geopackage"')

    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                            action='help', 
                            help='show this message and exit')
    flag_group.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Flag to change the logging level from INFO to DEBUG')
    
    args = parser.parse_args()
    args.data = post_parse_data(args.data)
    return args

def draw_bounding_mask(image, map_image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(map_image, contours, -1, MASK_COLOR, BORDER_THICKNESS)

    return map_image

def build_raster_training_data(data, output):
    training_data = {}
    for file in tqdm(data):
        file_name = os.path.basename(os.path.splitext(file)[0])
        map_name = '_'.join(file_name.split('_')[:2])
        feat_type = file_name.split('_')[-1]
        if feat_type != 'poly':
            continue

        # Load the poly image
        image, crs, transform = io.loadGeoTiff(file)
        image = image.transpose(1,2,0) # Convert to opencv format
        log.debug(f'Loaded {file_name} with shape {image.shape}')

        # Initalize an image for the map if it does not exist
        if training_data.get(map_name) is None:
            training_data[map_name] = (np.zeros_like(image), crs, transform)
        
        # add boundary mask to the map image
        map_image = training_data[map_name][0]
        map_image = draw_bounding_mask(image, map_image)
        log.debug('    Drew boundary mask on map image')

    # Save boundary training data
    log.info(f'Saving boundary training data to {output}')
    for label, (map_image, crs, transform) in tqdm(training_data.items()):
        map_image = map_image.transpose(2,0,1) # Convert back to rasterio format
        save_path = os.path.join(output, f'{label}_boundary_line.tif')
        io.saveGeoTiff(save_path, map_image, crs, transform)
        log.debug(f'    Saved {save_path}')

# Got to be a better way to do this
def apply_transfrom(contours, transfrom):
    geo_contours = []
    for contour in contours:
        geo_contour = []
        if len(contour) <= 2:
            continue
        for point in contour.squeeze():
            geo_contour.append(np.array(point.tolist() * transfrom))
        geo_contours.append(np.array(geo_contour))
    return geo_contours

def build_vector_training_data(data, output, filetype):
    training_data = {}
    for file in tqdm(data):
        file_name = os.path.basename(os.path.splitext(file)[0])
        map_name = '_'.join(file_name.split('_')[:1])
        feat_type = file_name.split('_')[-1]
        if feat_type != 'poly':
            continue

        # Load the poly image
        image, crs, transform = io.loadGeoTiff(file)
        image = image.transpose(1,2,0) # Convert to opencv format

         # Initalize an array for the map if it does not exist
        if training_data.get(map_name) is None:
            training_data[map_name] = ([], crs, transform)

        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = apply_transfrom(contours, transform)
        lines = [Polygon(contour.squeeze()) for contour in contours if len(contour) > 2]
        training_data[map_name][0].extend(lines)

    # Save boundary training data
    log.info(f'Saving boundary training data to {output}')
    for label, (lines, crs, transform) in tqdm(training_data.items()):
        # Save boundary training data
        label = f'{label}_boundary_line'
        if filetype == 'geopackage':
            save_path = os.path.join(output, f'{label}.gpkg')
        else:
            save_path = os.path.join(output, f'{label}.geojson')
        gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
        io.saveGeopackage(gdf, save_path, layer=label, filetype=filetype)
        log.debug(f'    Saved {save_path}')

def main():
    args = parse_command_line()

    # Start logger
    FILE_LOG_LEVEL = logging.DEBUG
    STREAM_LOG_LEVEL = logging.INFO
    if args.verbose:
        STREAM_LOG_LEVEL = logging.DEBUG

    global log
    log = utils.start_logger('CB_Training_builder', args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL)

    # Create output directories if needed
    if args.output is not None and not os.path.exists(args.output):
        os.makedirs(args.output)

    # Log info statement to start
    log.info(f'Building training data from {args.data}.')
    if args.type == 'raster':
        build_raster_training_data(args.data, args.output)
    if args.type == 'vector':
        build_vector_training_data(args.data, args.output, args.export_type)
    
if __name__ == '__main__':
    main()

# file = '../../../datasets/validation/tmp/AK_Dillingham_ak_poly.tif'