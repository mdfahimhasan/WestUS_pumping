# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import ee
import sys
import requests
import numpy as np
import rasterio as rio
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Polygon
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, clip_resample_reproject_raster, mosaic_rasters_from_directory

# ***************************** gcloud and earth engine authentication *****************************

# # first, we need to install and authenticate gcloud. use the link https://cloud.google.com/sdk/docs/install to install
# # gcloud for windows/linux
# # Once, installed gcloud need to be authenticated. For windows, I used the instructions fro the following link to authenticate
# # and set gcloud project - https://www.youtube.com/watch?v=k-8qFh8EfFA
# # For linux, I used the instructions from the following link to download and install gcloud - https://cloud.google.com/sdk/docs/install#linux
# # After installation in linux, I used the following commands to authenticate and set project
# #   >> gcloud auth login --no-launch-browser  (it will give a link > copy the link in windows browser > paste the generated code in linux command line > authentication will be completed)
# #   >> gcloud config set project project_name  (this will set the project, but first a project has to be created in gcloud following the instructions in this link - https://www.youtube.com/watch?v=k-8qFh8EfFA)


# # if ee.Authenticate() shows gcloud error even after gcloud has been installed or
# # it shows local host error when trying to authenticate from linux commandline,
# # use the following command
# # earthengine authenticate --auth_mode=notebook
# # source: https://gis.stackexchange.com/questions/445457/gcloud-command-not-found-when-authenticating-google-earth-engine
# # This command works for both windows in linux. In linux, once the environment is activated, no need to start python.
# # instead, use the command in the command line > **copy the link in windows browser** > paste the verification code in linux command line again to complete verification.
# # The errors are stemming from earthengine-api version which is associated with python version.
# # previously, in linux, earthengine authenticate --quiet command used to work.

# # # Once gcloud and earth engine have been authenticated, no need to run the authentication process again.
# # # Just start from ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

# ee.Authenticate()

# ***************************************************************************************

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/ref_rasters/GEE_merging_refraster_larger_grids.tif'
gee_grid_shape_large = '../../Data_main/ref_shapes/WestUS_gee_grid_large.shp'


def get_data_GEE_saveTopath(url_and_file_path):
    """
    Uses data url to get data from GEE and save it to provided local file paths.

    :param url_and_file_path: A list of tuples where each tuple has the data url (1st member) and local file path
                             (2nd member).
    :return: None
    """
    # unpacking tuple
    data_url, file_path = url_and_file_path

    # get data from GEE
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(data_url, allow_redirects=True)
            print('Downloading', file_path, '.....')
            r.raise_for_status()  # Raise an exception for bad HTTP status codes

            # save data to local file path
            with open(file_path, 'wb') as f:
                f.write(r.content)

            # This is a check block to see if downloaded datasets are OK
            # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
            # Re-downloading the same data might not have that error
            if '.tif' in file_path:  # only for data downloaded in geotiff format
                src = rio.open(file_path)
                data = src.read(1)
                src.close()

            break      # exit loop if download and data reading succeed; it data can't be read break will not be implemented
                       # and code will go to the 'except' block

        except Exception as e:
            print(f'attempt {attempt + 1} failed for {file_path}. Error: {e}. Trying again')
            if attempt == MAX_RETRIES - 1:
                print(f'failed to download {file_path} after {MAX_RETRIES} attempts.')


def download_data_from_GEE_by_multiprocess(download_urls_fp_list, use_cpu=2):
    """
    Use python multiprocessing library to download data from GEE in a multi-thread approach. This function is a
    wrapper over get_data_GEE_saveTopath() function providing muti-threading support.

    :param download_urls_fp_list: A list of tuples where each tuple has the data url (1st member) and local file path
                                  (2nd member).
    :param use_cpu: Number of CPU/core (Int) to use for downloading. Default set to 2.

    :return: None.
    """
    # Using ThreadPool() instead of pool() as this is an I/O bound job not CPU bound
    # Using imap() as it completes assigning one task at a time to the ThreadPool()
    # and blocks until each task is complete
    print('######')
    print('Downloading data from GEE..')
    print(f'{cpu_count()} CPUs on this machine. Engaging {use_cpu} CPUs for downloading')
    print('######')

    pool = ThreadPool(use_cpu)
    results = pool.imap(get_data_GEE_saveTopath, download_urls_fp_list)
    pool.close()
    pool.join()


def create_fishnet(input_gdf, fishnet_crs, fishnet_size, fishnet_file):
    """
    Create fishnet polygons of given size from input gdf.

    :param input_gdf: A geodataframe whose bounds are used to create the fishnet polygons.
                      The CRS of the input gdf must be in projected coordinate system.
    :param fishnet_crs: CRS of the output fishnet.
    :param fishnet_size: Fishnet dimension (one side) in unit of meters.
    :param fishnet_file: File path of output fishnet polygon.

    :return: Fishnet geodataframe.
    """
    output_dir = os.path.dirname(fishnet_file)
    makedirs([output_dir])

    # min-max bounds of the input gdf file
    xmin, ymin, xmax, ymax = input_gdf.total_bounds

    # making list of column and row coordinates to make the fishnets
    length = fishnet_size
    width = fishnet_size
    cols = list(np.arange(xmin, xmax + width, width))
    rows = list(np.arange(ymin, ymax + length, length))

    polygons = []  # an empty list to add all the polygons

    # appending the bounding coordinates of each polygon of the fishnet in the polygons columns
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x, y), (x + width, y), (x + width, y + length), (x, y + length)]))

    # creating geodataframe from the fishnet polygon coordinates and changing crs to given crs
    fishnet = gpd.GeoDataFrame({'geometry': polygons}, crs=input_gdf.crs).clip(input_gdf).reset_index(drop=True)
    fishnet['FID'] = fishnet.index + 1
    fishnet = fishnet.to_crs(fishnet_crs)
    fishnet.to_file(fishnet_file)

    return fishnet


def get_gee_dict(data_name):
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    gee_data_dict = {
        'Landsat5_NDVI': 'LANDSAT/LT05/C02/T1_L2',
        'Landsat8_NDVI': 'LANDSAT/LC08/C02/T1_L2',
        'Landsat5_OSAVI': 'LANDSAT/LT05/C02/T1_L2',
        'Landsat8_OSAVI': 'LANDSAT/LC08/C02/T1_L2',
        'Landsat5_NDMI': 'LANDSAT/LT05/C02/T1_L2',
        'Landsat8_NDMI': 'LANDSAT/LC08/C02/T1_L2',
        'Landsat5_GCVI': 'LANDSAT/LT05/C02/T1_L2',
        'Landsat8_GCVI': 'LANDSAT/LC08/C02/T1_L2',
        'MODIS_Day_LST': 'MODIS/006/MOD11A2',  # check for cloudcover
        'MODIS_Terra_NDVI': 'MODIS/061/MOD13Q1',  # cloudcover mask added later
        'MODIS_Terra_EVI': 'MODIS/061/MOD13Q1',  # cloudcover mask added later
        'MODIS_NDMI': 'MODIS/061/MOD09A1',  # cloudcover mask added later
        'MODIS_NDVI': 'MODIS/061/MOD09A1',  # cloudcover mask added later
        'MODIS_LAI': 'MODIS/061/MOD15A2H',
        'GRIDMET_Precip': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_RET': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_Tmax': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_maxRH': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_minRH': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_windVel': 'IDAHO_EPSCOR/GRIDMET',  # at 10m
        'GRIDMET_shortRad': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_vpd': 'IDAHO_EPSCOR/GRIDMET',
        'DAYMET_sunHr': 'NASA/ORNL/DAYMET_V4',
        'USDA_CDL': 'USDA/NASS/CDL',
        'Field_capacity': 'OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01',
        'Bulk_density': 'OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02',
        'Sand_content': 'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02',
        'Clay_content': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
        'DEM': 'USGS/SRTMGL1_003',
        'Tree_cover': 'NASA/MEASURES/GFCC/TC/v3',
        'spi': 'GRIDMET/DROUGHT',       # Standardized Precipitation Index (precipitation anomalies)
        'spei': 'GRIDMET/DROUGHT',      # Standardized Precipitation Evapotranspiration Index (temperature-driven drought-water balance)
        'eddi': 'GRIDMET/DROUGHT'       # Evaporative Drought Demand Index (atmospheric drying demand)
    }

    gee_band_dict = {
        'Landsat5_NDVI': ['SR_B4', 'SR_B3'],    # bands for NIR and Red, respectively
        'Landsat8_NDVI': ['SR_B5', 'SR_B4'],    # bands for NIR and Red, respectively
        'Landsat5_OSAVI': ['SR_B4', 'SR_B3'],   # bands for NIR and Red, respectively
        'Landsat8_OSAVI': ['SR_B5', 'SR_B4'],   # bands for NIR and Red, respectively
        'Landsat5_NDMI': ['SR_B4', 'SR_B5'],    # bands for NIR and SWIR1, respectively
        'Landsat8_NDMI': ['SR_B5', 'SR_B6'],    # bands for NIR and SWIR1, respectively
        'Landsat5_GCVI': ['SR_B4', 'SR_B2'],    # bands for NIR and Green, respectively
        'Landsat8_GCVI': ['SR_B4', 'SR_B2'],    # bands for NIR and Green, respectively,
        'MODIS_Day_LST': 'LST_Day_1km',
        'MODIS_Terra_NDVI': 'NDVI',
        'MODIS_Terra_EVI': 'EVI',
        'MODIS_NDMI': ['sur_refl_b02', 'sur_refl_b06'],  # bands for NIR and SWIR, respectively
        'MODIS_NDVI': ['sur_refl_b02', 'sur_refl_b01'],  # bands for NIR and Red, respectively
        'MODIS_LAI': 'Lai_500m',
        'GRIDMET_Precip': 'pr',  # daily total, unit in mm
        'GRIDMET_RET': 'etr',
        'GRIDMET_Tmax': 'tmmx',  # unit in K
        'GRIDMET_maxRH': 'rmax',
        'GRIDMET_minRH': 'rmin',
        'GRIDMET_windVel': 'vs',
        'GRIDMET_shortRad': 'srad',
        'GRIDMET_vpd': 'vpd',
        'DAYMET_sunHr': 'dayl',
        'USDA_CDL': 'cropland',
        'Field_capacity': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'Bulk_density': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'Sand_content': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'Clay_content': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'DEM': 'elevation',
        'Tree_cover': 'tree_canopy_cover',
        'spi': 'spi1y',
        'spei': 'spi1y',
        'eddi': 'eddi1y'
    }

    gee_scale_dict = {
        'Landsat5_NDVI': [0.0000275, -0.2],     # the first factor is multiplied, the second factor is an offset
        'Landsat8_NDVI': [0.0000275, -0.2],     # the first factor is multiplied, the second factor is an offset
        'Landsat5_OSAVI': [0.0000275, -0.2],    # the first factor is multiplied, the second factor is an offset
        'Landsat8_OSAVI': [0.0000275, -0.2],    # the first factor is multiplied, the second factor is an offset
        'Landsat5_NDMI': [0.0000275, -0.2],     # the first factor is multiplied, the second factor is an offset
        'Landsat8_NDMI': [0.0000275, -0.2],     # the first factor is multiplied, the second factor is an offset
        'Landsat5_GCVI': [0.0000275, -0.2],     # the first factor is multiplied, the second factor is an offset
        'Landsat8_GCVI': [0.0000275, -0.2],     # the first factor is multiplied, the second factor is an offset
        'MODIS_Day_LST': 0.02,
        'MODIS_Terra_NDVI': 0.0001,
        'MODIS_Terra_EVI': 0.0001,
        'MODIS_NDMI': 0.0001,
        'MODIS_NDVI': 0.0001,
        'MODIS_LAI': 0.1,
        'MODIS_ET': 0.1,
        'GRIDMET_Precip':1,
        'GRIDMET_RET': 1,
        'GRIDMET_Tmax':1,
        'GRIDMET_maxRH': 1,
        'GRIDMET_minRH': 1,
        'GRIDMET_windVel': 1,
        'GRIDMET_shortRad': 1,
        'GRIDMET_vpd': 1,
        'DAYMET_sunHr': 1,
        'USDA_CDL': 1,
        'Field_capacity': 1,
        'Bulk_density': 1,
        'Organic_carbon_content': 1,
        'Sand_content': 1,
        'Clay_content': 1,
        'DEM': 1,
        'Tree_cover': 1,
        'spi': 1,
        'spei': 1,
        'eddi': 1
    }

    aggregation_dict = {
        'Landsat5_NDVI': ee.Reducer.median(),
        'Landsat8_NDVI': ee.Reducer.median(),
        'Landsat5_OSAVI': ee.Reducer.median(),
        'Landsat8_OSAVI': ee.Reducer.median(),
        'Landsat5_NDMI': ee.Reducer.median(),
        'Landsat8_NDMI': ee.Reducer.median(),
        'Landsat5_GCVI': ee.Reducer.median(),
        'Landsat8_GCVI': ee.Reducer.median(),
        'MODIS_Day_LST': ee.Reducer.mean(),
        'MODIS_Terra_NDVI': ee.Reducer.median(),
        'MODIS_Terra_EVI': ee.Reducer.median(),
        'MODIS_NDMI': ee.Reducer.median(),
        'MODIS_NDVI': ee.Reducer.median(),
        'MODIS_LAI': ee.Reducer.mean(),
        'GRIDMET_Precip': ee.Reducer.sum(),
        'GRIDMET_RET': ee.Reducer.sum(),
        'GRIDMET_Tmax': ee.Reducer.mean(),
        'GRIDMET_maxRH': ee.Reducer.mean(),
        'GRIDMET_minRH': ee.Reducer.mean(),
        'GRIDMET_windVel': ee.Reducer.mean(),
        'GRIDMET_shortRad': ee.Reducer.mean(),
        'GRIDMET_vpd': ee.Reducer.mean(),
        'DAYMET_sunHr': ee.Reducer.mean(),
        'USDA_CDL': ee.Reducer.first(),
        'Field_capacity': ee.Reducer.mean(),
        'Bulk_density': ee.Reducer.mean(),
        'Sand_content': ee.Reducer.mean(),
        'Clay_content': ee.Reducer.mean(),
        'DEM': None,
        'Tree_cover': ee.Reducer.mean(),
        'spi': ee.Reducer.mean(),
        'spei': ee.Reducer.mean(),
        'eddi': ee.Reducer.mean()
    }

    # # Note on start date and end date dictionaries
    # The start and end dates have been set based on what duration of data can be downloaded.
    # They may not exactly match with the data availability in GEE
    # In most cases the end date is shifted a month later to cover the end month's data

    month_start_date_dict = {
        'Landsat5_NDVI': datetime(1984, 3, 1),
        'Landsat8_NDVI': datetime(2013, 3, 1),
        'Landsat5_OSAVI': datetime(1984, 3, 1),
        'Landsat8_OSAVI': datetime(2013, 3, 1),
        'Landsat5_NDMI': datetime(1984, 3, 1),
        'Landsat8_NDMI': datetime(2013, 3, 1),
        'Landsat5_GCVI': datetime(1984, 3, 1),
        'Landsat8_GCVI': datetime(2013, 3, 1),
        'MODIS_Day_LST': datetime(2000, 2, 1),
        'MODIS_Terra_NDVI': datetime(2000, 2, 1),
        'MODIS_Terra_EVI': datetime(2000, 2, 1),
        'MODIS_NDMI': datetime(2000, 2, 1),
        'MODIS_NDVI': datetime(2000, 2, 1),
        'MODIS_LAI': datetime(2000, 2, 1),
        'MODIS_ET': datetime(2001, 1, 1),
        'GRIDMET_Precip': datetime(1979, 1, 1),
        'GRIDMET_RET': datetime(1979, 1, 1),
        'GRIDMET_Tmax': datetime(1979, 1, 1),
        'GRIDMET_maxRH': datetime(1979, 1, 1),
        'GRIDMET_minRH': datetime(1979, 1, 1),
        'GRIDMET_windVel': datetime(1979, 1, 1),
        'GRIDMET_shortRad': datetime(1979, 1, 1),
        'GRIDMET_vpd': datetime(1979, 1, 1),
        'DAYMET_sunHr': datetime(1980, 1, 1),
        'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
        'Field_capacity': None,
        'Bulk_density': None,
        'Sand_content': None,
        'Clay_content': None,
        'DEM': None,
        'Tree_cover': datetime(2000, 1, 1),
        'spi': datetime(1980, 1, 5),
        'spei': datetime(1980, 1, 5),
        'eddi': datetime(1980, 1, 5)
    }

    month_end_date_dict = {
        'Landsat5_NDVI': datetime(2012, 5, 1),
        'Landsat8_NDVI': datetime(2024, 1, 1),
        'Landsat5_OSAVI': datetime(2012, 5, 1),
        'Landsat8_OSAVI': datetime(2024, 1, 1),
        'Landsat5_NDMI': datetime(2012, 5, 1),
        'Landsat8_NDMI': datetime(2024, 1, 1),
        'Landsat5_GCVI': datetime(2012, 5, 1),
        'Landsat8_GCVI': datetime(2024, 1, 1),
        'MODIS_Day_LST': datetime(2023, 8, 29),
        'MODIS_Terra_NDVI': datetime(2023, 8, 13),
        'MODIS_Terra_EVI': datetime(2023, 8, 13),
        'MODIS_NDMI': datetime(2023, 8, 29),
        'MODIS_NDVI': datetime(2023, 8, 29),
        'MODIS_LAI': datetime(2023, 11, 9),
        'GRIDMET_Precip': datetime(2023, 9, 15),
        'GRIDMET_RET': datetime(2022, 12, 1),
        'GRIDMET_Tmax': datetime(2022, 12, 1),
        'GRIDMET_maxRH': datetime(2022, 12, 1),
        'GRIDMET_minRH': datetime(2022, 12, 1),
        'GRIDMET_windVel': datetime(2022, 12, 1),
        'GRIDMET_shortRad': datetime(2022, 12, 1),
        'GRIDMET_vpd': datetime(2022, 12, 1),
        'DAYMET_sunHr': datetime(2022, 12, 31),
        'USDA_CDL': datetime(2022, 1, 1),
        'Field_capacity': None,
        'Bulk_density': None,
        'Sand_content': None,
        'Clay_content': None,
        'DEM': None,
        'Tree_cover': datetime(2015, 1, 1),
        'spi': datetime(2024, 12, 31),
        'spei': datetime(2024, 12, 31),
        'eddi': datetime(2024, 12, 31)
    }

    year_start_date_dict = {
        'Landsat5_NDVI': datetime(1984, 1, 1),
        'Landsat8_NDVI': datetime(2013, 1, 1),
        'Landsat5_OSAVI': datetime(1984, 1, 1),
        'Landsat8_OSAVI': datetime(2013, 1, 1),
        'Landsat5_NDMI': datetime(1984, 1, 1),
        'Landsat8_NDMI': datetime(2013, 1, 1),
        'Landsat5_GCVI': datetime(1984, 1, 1),
        'Landsat8_GCVI': datetime(2013, 1, 1),
        'MODIS_Day_LST': datetime(2000, 1, 1),
        'MODIS_Terra_NDVI': datetime(2000, 1, 1),
        'MODIS_Terra_EVI': datetime(2000, 1, 1),
        'MODIS_NDMI': datetime(2000, 1, 1),
        'MODIS_NDVI': datetime(2000, 1, 1),
        'MODIS_LAI': datetime(2000, 1, 1),
        'GRIDMET_Precip': datetime(1979, 1, 1),
        'GRIDMET_RET': datetime(1979, 1, 1),
        'GRIDMET_Tmax': datetime(1979, 1, 1),
        'GRIDMET_maxRH': datetime(1979, 1, 1),
        'GRIDMET_minRH': datetime(1979, 1, 1),
        'GRIDMET_windVel': datetime(1979, 1, 1),
        'GRIDMET_shortRad': datetime(1979, 1, 1),
        'GRIDMET_vpd': datetime(1979, 1, 1),
        'DAYMET_sunHr': datetime(1980, 1, 1),
        'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
        'Field_capacity': None,
        'Bulk_density': None,
        'Sand_content': None,
        'Clay_content': None,
        'DEM': None,
        'Tree_cover': datetime(2000, 1, 1),
        'spi': datetime(1980, 1, 5),
        'spei': datetime(1980, 1, 5),
        'eddi': datetime(1980, 1, 5)
    }

    year_end_date_dict = {
        'Landsat5_NDVI': datetime(2012, 1, 1),
        'Landsat8_NDVI': datetime(2024, 1, 1),
        'Landsat5_OSAVI': datetime(2012, 1, 1),
        'Landsat8_OSAVI': datetime(2024, 1, 1),
        'Landsat5_NDMI': datetime(2012, 1, 1),
        'Landsat8_NDMI': datetime(2024, 1, 1),
        'Landsat5_GCVI': datetime(2012, 1, 1),
        'Landsat8_GCVI': datetime(2024, 1, 1),
        'MODIS_Day_LST': datetime(2024, 1, 1),
        'MODIS_Terra_NDVI': datetime(2024, 1, 1),
        'MODIS_Terra_EVI': datetime(2024, 1, 1),
        'MODIS_NDMI': datetime(2024, 1, 1),
        'MODIS_NDVI': datetime(2024, 1, 1),
        'MODIS_LAI': datetime(2024, 1, 1),
        'GRIDMET_Precip': datetime(2024, 1, 1),
        'GRIDMET_RET': datetime(2024, 12, 1),
        'GRIDMET_Tmax': datetime(2024, 12, 1),
        'GRIDMET_maxRH': datetime(2024, 1, 1),
        'GRIDMET_minRH': datetime(2024, 1, 1),
        'GRIDMET_windVel': datetime(2024, 1, 1),
        'GRIDMET_shortRad': datetime(2024, 1, 1),
        'GRIDMET_vpd': datetime(2024, 12, 1),
        'DAYMET_sunHr': datetime(2023, 1, 1),
        'USDA_CDL': datetime(2022, 1, 1),
        'Field_capacity': None,
        'Bulk_density': None,
        'Sand_content': None,
        'Clay_content': None,
        'DEM': None,
        'Tree_cover': datetime(2015, 1, 1),
        'spi': datetime(2024, 12, 31),
        'spei': datetime(2024, 12, 31),
        'eddi': datetime(2024, 12, 31)
    }

    return gee_data_dict[data_name], gee_band_dict[data_name], gee_scale_dict[data_name], aggregation_dict[data_name], \
           month_start_date_dict[data_name], month_end_date_dict[data_name], year_start_date_dict[data_name], \
           year_end_date_dict[data_name]


def cloudMask_MODIS(data_name, start_date, end_date, from_bit, to_bit, geometry_bounds):
    """
    Applies cloud cover mask on GEE MODIS data.

    :param data_name: Data Name.
           Valid dataset include- ['MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDMI', 'MODIS_NDVI'].
    :param start_date: Start date of data to download. Generated from download_gee_data() func.
    :param end_date: End date of data to download. Generated from download_gee_data() func.
    :param from_bit: Start bit to consider for masking.
    :param to_bit: End bit to consider for masking.
    :param geometry_bounds: GEE geometry object.

    :return: Cloud filtered imagecollection.
    """

    def bitwise_extract(img):
        """
        Applies cloudmask on image.

        code source: https://spatialthoughts.com/2021/08/19/qa-bands-bitmasks-gee/

        :param img: The image.

        :return Cloud-masked image.
        """
        ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

        global qc_img

        if data_name in ['MODIS_Terra_NDVI', 'MODIS_Terra_EVI']:
            qc_img = img.select('DetailedQA')

        elif data_name in ['MODIS_NDMI', 'MODIS_NDVI']:
            qc_img = img.select('StateQA')

        masksize = ee.Number(1).add(to_bit).subtract(from_bit)
        mask = ee.Number(1).leftShift(masksize).subtract(1)
        apply_mask = qc_img.rightShift(from_bit).bitwiseAnd(mask).lte(1)
        return img.updateMask(apply_mask)

    if data_name in ['MODIS_Terra_NDVI', 'MODIS_Terra_EVI']:
        # filterBounds are not necessary, added it to reduce processing extent
        image = ee.ImageCollection(data_name).filterDate(start_date, end_date).filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked

    elif data_name in ['MODIS_NDMI', 'MODIS_NDVI']:
        image = ee.ImageCollection('MODIS/061/MOD09A1').filterDate(start_date, end_date).filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked


def cloudMask_landsat(data_name, imcol, start_date, end_date, geometry_bounds):
    """
    Applies cloud cover mask on GEE landsat data.

    :param data_name: Data Name.
           Valid dataset include-
           ['Landsat5_NDVI', 'Landsat8_NDVI', 'Landsat5_NDMI', 'Landsat8_NDMI',
            'Landsat5_GCVI', 'Landsat8_GCVI'].
    :param imcol: GEE imagecollection name.
    :param start_date: Start date of data to download. Generated from download_gee_data() func.
    :param end_date: End date of data to download. Generated from download_gee_data() func.
    :param geometry_bounds: GEE geometry object.

    :return: Cloud filtered imagecollection.
    """

    def bitwise_extract(img):
        """
        Applies cloudmask on image.
        :param img: The image.

        :return Cloud-masked image.
        """
        ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

        qa = img.select('QA_PIXEL')

        if data_name in ['Landsat5_NDVI', 'Landsat8_NDVI', 'Landsat5_NDMI',
                         'Landsat8_NDMI', 'Landsat5_GCVI', 'Landsat8_GCVI']:
            # Bits 3 and 4 are cloud shadow and cloud, respectively
            cloud = qa.bitwiseAnd(1 << 3).eq(0)  # Cloud bit(Bit 3)
            cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)  # Cloud shadow bit(Bit 4)

            mask = cloud.And(cloudShadow)
            img = img.updateMask(mask)

        return img

    imcol_selected = ee.ImageCollection(imcol).filterDate(start_date, end_date).filterBounds(geometry_bounds)
    cloud_masked_imcol = imcol_selected.map(bitwise_extract)

    return cloud_masked_imcol


def download_soil_datasets(data_name, download_dir, merge_keyword,
                           gee_grid_shape=gee_grid_shape_large,
                           refraster_westUS=WestUS_raster,
                           refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=WestUS_shape):
    """
    Download soil datasets from GEE.

    :param data_name: Data name.
                      Current valid data names are -
                        ['Field_capacity', 'Bulk_density', 'Sand_content','Clay_content']
    :param download_dir: File path of download directory.
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param gee_grid_shape: File path of the gee grids that will be used to download the data.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    if data_name in ['Field_capacity', 'Bulk_density', 'Sand_content', 'Clay_content']:
        data, all_bands, multiply_scale, reducer, _, _, _, _ = get_gee_dict(data_name)

        # selecting datasets with all bands ['b0', 'b10', 'b30', 'b60', 'b100', 'b200']
        data_all_bands = ee.Image(data).select(all_bands)

        # calculating band average
        dataset_mean = data_all_bands.reduce(reducer)

        # loading grids that will be used to download the data
        grids = gpd.read_file(gee_grid_shape)
        grids = grids.sort_values(by='FID', ascending=True)
        grid_geometry = grids['geometry']
        grid_no = grids['FID']

        for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
            roi = geometry.bounds
            gee_extent = ee.Geometry.Rectangle(roi)

            # making data url
            data_url = dataset_mean.getDownloadURL({'name': data_name,
                                                    'crs': 'EPSG:4269',  # NAD83
                                                    'scale': 2200,  # in meter. equal to ~0.02 deg
                                                    'region': gee_extent,
                                                    'format': 'GEO_TIFF'})
            key_word = data_name
            local_file_name = os.path.join(download_dir, f'{key_word}_{str(grid_sr)}.tif')
            print('Downloading', local_file_name, '.....')
            r = requests.get(data_url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)

        mosaic_name = f'{data_name}.tif'
        mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
        clip_dir = os.path.join(download_dir, f'{merge_keyword}')

        makedirs([clip_dir, mosaic_dir])
        merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                                  raster_name=mosaic_name,
                                                                  ref_raster=refraster_gee_merge,
                                                                  search_by=f'*.tif', nodata=no_data_value)

        clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                       output_raster_dir=clip_dir, clip_and_resample=True,
                                       use_ref_width_height=False, resolution=model_res,
                                       ref_raster=refraster_westUS)

        print(f'{data_name} data downloaded and merged')

    else:
        pass


def download_tree_cover_data(data_name, download_dir, merge_keyword,
                             gee_grid_shape='../../Data_main/ref_shapes/WestUS_gee_extent.shp',
                             refraster_westUS=WestUS_raster,
                             refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=WestUS_shape):
    """
    Download Tree Cover data from GEE.

    :param data_name: Data name. Use 'Tree_cover'.
    :param download_dir: File path of download directory.
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param gee_grid_shape: File path of the gee grids that will be used to download the data.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # getting dataset information from the data dictionary
    data, band, multiply_scale, reducer, _, _, _, _ = get_gee_dict(data_name)

    # filtering data
    dataset = ee.ImageCollection(data).filter(ee.Filter.date('2000-01-01', '2015-01-01')).select(band) \
        .reduce(reducer).toFloat()

    # loading grids that will be used to download the data
    grids = gpd.read_file(gee_grid_shape)
    grids = grids.sort_values(by='FID', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['FID']

    for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
        roi = geometry.bounds
        gee_extent = ee.Geometry.Rectangle(roi)

        # making data url
        data_url = dataset.getDownloadURL({'name': data_name,
                                           'crs': 'EPSG:4269',  # NAD83
                                           'scale': 2200,  # in meter. equal to ~0.02 deg
                                           'region': gee_extent,
                                           'format': 'GEO_TIFF'})
        key_word = data_name
        local_file_name = os.path.join(download_dir, f'{key_word}_{str(grid_sr)}.tif')
        print('Downloading', local_file_name, '.....')
        r = requests.get(data_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)

    mosaic_name = f'{data_name}.tif'
    mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
    clip_dir = os.path.join(download_dir, f'{merge_keyword}')

    makedirs([clip_dir, mosaic_dir])
    merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                              raster_name=mosaic_name,
                                                              ref_raster=refraster_gee_merge,
                                                              search_by=f'*.tif', nodata=no_data_value)

    clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                   output_raster_dir=clip_dir, clip_and_resample=True,
                                   use_ref_width_height=False, resolution=model_res,
                                   ref_raster=refraster_westUS)

    print(f'{data_name} data downloaded and merged')


def download_DEM_Slope_data(data_name, download_dir, merge_keyword,
                            gee_grid_shape=gee_grid_shape_large,
                            refraster_westUS=WestUS_raster,
                            refraster_gee_merge=GEE_merging_refraster_large_grids,
                            westUS_shape=WestUS_shape,
                            terrain_slope=False):
    """
    Download DEM/Slope data from GEE.

    :param data_name: Data name. Use keyword 'DEM' for downloading both DEM and Slope data. SLope is downloaded
                      in degrees.
    :param download_dir: File path of download directory.
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param gee_grid_shape: File path of the gee grids that will be used to download the data.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.
    :param terrain_slope : If slope data download is needed in degrees from GEE directly. Defaults to False to download
                           DEM data only. The DEM data will be later processed to 'percent' slope data using gdal.

    :return: None.
    """
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # getting dataset information from the data dictionary
    data, band, multiply_scale, reducer, _, _, _, _ = get_gee_dict(data_name)

    # filtering data
    dataset = ee.Image(data).select(band).multiply(multiply_scale).toFloat()

    if terrain_slope:
        dataset = ee.Terrain.slope(dataset)

    # loading grids that will be used to download the data
    grids = gpd.read_file(gee_grid_shape)
    grids = grids.sort_values(by='FID', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['FID']

    for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
        roi = geometry.bounds
        gee_extent = ee.Geometry.Rectangle(roi)

        # making data url
        data_url = dataset.getDownloadURL({'name': data_name,
                                           'crs': 'EPSG:4269',  # NAD83
                                           'scale': 2200,  # in meter. equal to ~0.02 deg
                                           'region': gee_extent,
                                           'format': 'GEO_TIFF'})
        key_word = data_name
        local_file_name = os.path.join(download_dir, f'{key_word}_{str(grid_sr)}.tif')
        print('Downloading', local_file_name, '.....')
        r = requests.get(data_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)

    mosaic_name = f'{data_name}.tif'
    mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
    clip_dir = os.path.join(download_dir, f'{merge_keyword}')

    makedirs([clip_dir, mosaic_dir])
    merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                              raster_name=mosaic_name,
                                                              ref_raster=refraster_gee_merge,
                                                              search_by=f'*.tif', nodata=no_data_value)

    clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                   output_raster_dir=clip_dir, clip_and_resample=True,
                                   use_ref_width_height=False, resolution=model_res,
                                   ref_raster=refraster_westUS)

    print(f'{data_name} data downloaded and merged')


def download_gee_data_monthly(data_name, download_dir, year_list, month_range, merge_keyword,
                              gee_grid_shape='../../Data_main/ref_shapes/WestUS_gee_grid_large.shp',
                              refraster_westUS=WestUS_raster, refraster_gee_merge=GEE_merging_refraster_large_grids,
                              use_cpu_while_multidownloading=15, westUS_shape=WestUS_shape):
    """
    Download data (at yearly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['MODIS_Day_LST', 'Landsat5_NDVI', 'Landsat8_NDVI', 'Landsat5_OSAVI', 'Landsat8_OSAVI',
        'Landsat5_NDMI', 'Landsat8_NDMI', 'Landsat5_GCVI', 'Landsat8_GCVI', 'MODIS_Terra_NDVI',
        'MODIS_Terra_EVI', 'MODIS_NDMI', 'MODIS_NDVI', 'GRIDMET_Precip', 'GRIDMET_Tmax',
        'GRIDMET_RET', 'GRIDMET_maxRH', 'GRIDMET_minRH', 'GRIDMET_windVel',
        'GRIDMET_shortRad', 'GRIDMET_vpd', 'DAYMET_sunHr']
    :param download_dir: File path of download directory.
    :param year_list: List of year_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param gee_grid_shape: File path of gee grids that will be used to download the data.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    global key_word

    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    if any('Landsat' in i for i in ['Landsat5_NDVI', 'Landsat8_NDVI',
                                    'Landsat5_OSAVI', 'Landsat8_OSAVI',
                                    'Landsat5_NDMI', 'Landsat8_NDMI',
                                    'Landsat5_GCVI', 'Landsat8_GCVI']):
        download_dir = os.path.join(download_dir, data_name.split('_')[-1])
    else:
        download_dir = os.path.join(download_dir, data_name)

    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, scale_factor, reducer, month_start_range, month_end_range, \
    year_start_range, year_end_range = get_gee_dict(data_name)

    # loading grids that will be used to download the data
    grids = gpd.read_file(gee_grid_shape)
    grids = grids.sort_values(by='FID', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['FID']

    month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

    for year in year_list:  # first loop for year_list
        for month in month_list:
            print('********')
            print(f'Getting data urls for year={year}, month={month}.....')

            # Setting date ranges
            start_date = ee.Date.fromYMD(year, month, 1)
            start_date_dt = datetime(year, month, 1)

            if month < 12:
                end_date = ee.Date.fromYMD(year, month + 1, 1)
                end_date_dt = datetime(year, month + 1, 1)
            else:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
                end_date_dt = datetime(year + 1, 1, 1)

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= year_start_range) & (end_date_dt <= year_end_range):
                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
                    roi = geometry.bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    if data_name in ['Landsat5_NDVI', 'Landsat8_NDVI']:
                        nir = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[0]). \
                            reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                        red = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[1]). \
                            reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                        download_data = nir.subtract(red).divide(nir.add(red))

                    elif data_name in ['Landsat5_OSAVI', 'Landsat8_OSAVI']:
                        nir = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[0]). \
                            reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                        red = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[1]). \
                            reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                        download_data = nir.subtract(red).divide(nir.add(red).add(0.16))

                    elif data_name in ['Landsat5_NDMI', 'Landsat8_NDMI']:
                        nir = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[0]). \
                            reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                        swir1 = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[1]). \
                            reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                        download_data = nir.subtract(swir1).divide(nir.add(swir1))

                    elif data_name in ['Landsat5_GCVI', 'Landsat8_GCVI']:
                        nir = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[0]). \
                            reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                        green = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[1]). \
                            reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                        download_data = nir.divide(green).subtract(1)

                    elif data_name == 'GRIDMET_RET':
                        # multiplying by 0.85 to applying bias correction in GRIDMET RET. GRIDMET RET is overestimated
                        # by 12-31% across CONUS (Blankenau et al. (2020). Senay et al. (2022) applied 0.85 as constant
                        # bias correction factor.
                        download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            filterBounds(gee_extent).reduce(reducer).multiply(0.85).multiply(scale_factor).toFloat()

                    elif data_name == 'DAYMET_sun_hr':
                        # dividing by 3600 to convert from second to hr
                        download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            filterBounds(gee_extent).reduce(reducer).divide(3600).multiply(scale_factor).toFloat()

                    else:
                        download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            filterBounds(gee_extent).reduce(reducer).multiply(scale_factor).toFloat()

                    data_url = download_data.getDownloadURL({'name': data_name,
                                                             'crs': 'EPSG:4269',  # NAD83
                                                             'scale': 2200,  # in meter. equal to ~0.02 deg
                                                             'region': gee_extent,
                                                             'format': 'GEO_TIFF'})

                    if any('Landsat' in i for i in ['Landsat5_NDVI', 'Landsat8_NDVI',
                                                    'Landsat5_OSAVI', 'Landsat8_OSAVI',
                                                    'Landsat5_NDMI', 'Landsat8_NDMI',
                                                    'Landsat5_GCVI', 'Landsat8_GCVI']):
                        key_word = data_name.split('_')[-1]
                    else:
                        key_word = data_name

                    local_file_path = os.path.join(download_dir, f'{key_word}_{str(year)}_{month}_{str(grid_sr)}.tif')

                    # Appending data url and local file path (to save data) to a central list
                    data_url_list.append(data_url)
                    local_file_paths_list.append(local_file_path)

                    # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                    # there is enough data url gathered for download.
                    if (len(data_url_list) == 120) | (grid_sr == len(grid_no)):  # downloads data when one of the conditions are met

                        # Combining url and file paths together to pass in multiprocessing
                        urls_to_file_paths_compile = []

                        for j, k in zip(data_url_list, local_file_paths_list):
                            urls_to_file_paths_compile.append([j, k])

                        # Download data by multi-processing/multi-threading
                        download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                               use_cpu=use_cpu_while_multidownloading)

                        # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                        # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                        data_url_list = []
                        local_file_paths_list = []

                # merging downloaded datasets
                mosaic_name = f'{key_word}_{year}_{month}.tif'
                mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
                clip_dir = os.path.join(download_dir, f'{merge_keyword}')

                makedirs([clip_dir, mosaic_dir])
                merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                                          raster_name=mosaic_name,
                                                                          ref_raster=refraster_gee_merge,
                                                                          search_by=f'*{year}_{month}*.tif',
                                                                          nodata=no_data_value)

                clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                               output_raster_dir=clip_dir, clip_and_resample=True,
                                               use_ref_width_height=False, resolution=model_res,
                                               ref_raster=refraster_westUS)

                print(f'{data_name} yearly data downloaded and merged')

            else:
                print(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_gee_data_yearly(data_name, download_dir, year_list, month_range, merge_keyword,
                             gee_grid_shape='../../Data_main/ref_shapes/WestUS_gee_grid_large.shp',
                             refraster_westUS=WestUS_raster, refraster_gee_merge=GEE_merging_refraster_large_grids,
                             use_cpu_while_multidownloading=15, westUS_shape=WestUS_shape):
    """
    Download data (at yearly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['MODIS_Day_LST', 'Landsat5_NDVI', 'Landsat8_NDVI', 'Landsat5_OSAVI', 'Landsat8_OSAVI',
        'Landsat5_NDMI', 'Landsat8_NDMI', 'Landsat5_GCVI', 'Landsat8_GCVI', 'MODIS_Terra_NDVI',
        'MODIS_Terra_EVI', 'MODIS_NDMI', 'MODIS_NDVI', 'GRIDMET_Precip', 'GRIDMET_Tmax',
        'GRIDMET_RET', 'GRIDMET_maxRH', 'GRIDMET_minRH', 'GRIDMET_windVel',
        'GRIDMET_shortRad', 'GRIDMET_vpd', 'DAYMET_sunHr']
    :param download_dir: File path of download directory.
    :param year_list: List of year_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 4-12 use (4, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param gee_grid_shape: File path of gee grids that will be used to download the data.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    global key_word

    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    if any('Landsat' in i for i in ['Landsat5_NDVI', 'Landsat8_NDVI',
                                    'Landsat5_OSAVI', 'Landsat8_OSAVI',
                                    'Landsat5_NDMI', 'Landsat8_NDMI',
                                    'Landsat5_GCVI', 'Landsat8_GCVI']):
        download_dir = os.path.join(download_dir, data_name.split('_')[-1])
    else:
        download_dir = os.path.join(download_dir, data_name)

    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, scale_factor, reducer, month_start_range, month_end_range, \
    year_start_range, year_end_range = get_gee_dict(data_name)

    # loading grids that will be used to download the data
    grids = gpd.read_file(gee_grid_shape)
    grids = grids.sort_values(by='FID', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['FID']

    for year in year_list:  # first loop for year_list
        start_date = ee.Date.fromYMD(year, month_range[0], 1)
        start_date_dt = datetime(year, month_range[0], 1)
        if month_range[1] < 12:
            end_date = ee.Date.fromYMD(year, month_range[1] + 1, 1)
            end_date_dt = datetime(year, month_range[1] + 1, 1)
        else:
            end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
            end_date_dt = datetime(year + 1, 1, 1)

        # will collect url and file name in url list and local_file_paths_list
        data_url_list = []
        local_file_paths_list = []

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        if (start_date_dt >= year_start_range) & (end_date_dt <= year_end_range):

            for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
                roi = geometry.bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                if data_name in ['Landsat5_NDVI', 'Landsat8_NDVI']:
                    nir = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[0]). \
                        reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                    red = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[1]). \
                        reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                    download_data = nir.subtract(red).divide(nir.add(red))

                elif data_name in ['Landsat5_OSAVI', 'Landsat8_OSAVI']:
                    nir = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[0]). \
                        reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                    red = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[1]). \
                        reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                    download_data = nir.subtract(red).divide(nir.add(red).add(0.16))

                elif data_name in ['Landsat5_NDMI', 'Landsat8_NDMI']:
                    nir = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[0]). \
                        reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                    swir1 = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[1]). \
                        reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                    download_data = nir.subtract(swir1).divide(nir.add(swir1))

                elif data_name in ['Landsat5_GCVI', 'Landsat8_GCVI']:
                    nir = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[0]). \
                        reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                    green = cloudMask_landsat(data_name, data, start_date, end_date, gee_extent).select(band[1]). \
                        reduce(reducer).multiply(scale_factor[0]).add(scale_factor[1]).toFloat()
                    download_data = nir.divide(green).subtract(1)

                elif data_name == 'USDA_CDL':
                    cdl_dataset = ee.ImageCollection(data).filter((ee.Filter.calendarRange(year, year, 'year'))) \
                        .select(band).reduce(reducer).multiply(scale_factor).toFloat()

                    # List of non-crop pixels
                    noncrop_list = ee.List([60, 61, 63, 64, 65, 81, 82, 83, 87, 88, 111, 112, 121, 122, 123,
                                            124, 131, 141, 142, 143, 152, 190,
                                            195])  # 176 (pasture) kept in downloaded data

                    # Filtering out non-crop pixels. In non-crop pixels, assigning 0 and in crop pixels assigning 1
                    cdl_mask = cdl_dataset.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

                    # Masking with cdl mask to assign nodata value on non crop pixels
                    cdl_cropland = cdl_dataset.updateMask(cdl_mask)
                    download_data = cdl_cropland

                elif data_name == 'GRIDMET_RET':
                    # multiplying by 0.85 to applying bias correction in GRIDMET RET. GRIDMET RET is overestimated
                    # by 12-31% across CONUS (Blankenau et al. (2020). Senay et al. (2022) applied 0.85 as constant
                    # bias correction factor.
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).multiply(0.85).multiply(scale_factor).toFloat()

                elif data_name == 'DAYMET_sun_hr':
                    # dividing by 3600 to convert from second to hr
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).divide(3600).multiply(scale_factor).toFloat()

                else:
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).multiply(scale_factor).toFloat()

                data_url = download_data.getDownloadURL({'name': data_name,
                                                         'crs': 'EPSG:4269',  # NAD83
                                                         'scale': 2200,  # in meter. equal to ~0.02 deg
                                                         'region': gee_extent,
                                                         'format': 'GEO_TIFF'})

                if any('Landsat' in i for i in ['Landsat5_NDVI', 'Landsat8_NDVI',
                                                'Landsat5_OSAVI', 'Landsat8_OSAVI',
                                                'Landsat5_NDMI', 'Landsat8_NDMI',
                                                'Landsat5_GCVI', 'Landsat8_GCVI']):
                    key_word = data_name.split('_')[-1]
                else:
                    key_word = data_name

                local_file_path = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(grid_sr)}.tif')

                # Appending data url and local file path (to save data) to a central list
                data_url_list.append(data_url)
                local_file_paths_list.append(local_file_path)

                # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                # there is enough data url gathered for download.
                if (len(data_url_list) == 120) | (
                        grid_sr == len(grid_no)):  # downloads data when one of the conditions are met
                    # Combining url and file paths together to pass in multiprocessing
                    urls_to_file_paths_compile = []
                    for j, k in zip(data_url_list, local_file_paths_list):
                        urls_to_file_paths_compile.append([j, k])

                    # Download data by multi-processing/multi-threading
                    download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                           use_cpu=use_cpu_while_multidownloading)

                    # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                    # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                    data_url_list = []
                    local_file_paths_list = []

            # merging downloaded datasets
            mosaic_name = f'{key_word}_{year}.tif'
            mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
            clip_dir = os.path.join(download_dir, f'{merge_keyword}')

            makedirs([clip_dir, mosaic_dir])
            merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                                      raster_name=mosaic_name,
                                                                      ref_raster=refraster_gee_merge,
                                                                      search_by=f'*{year}*.tif', nodata=no_data_value)

            clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                           output_raster_dir=clip_dir, clip_and_resample=True,
                                           use_ref_width_height=False, resolution=model_res,
                                           ref_raster=refraster_westUS)

            print(f'{data_name} yearly data downloaded and merged')

        else:
            print(f'Data for year {year} is out of range. Skipping query')
            pass


def download_drought_indices_water_year(data_name, download_dir, year_list, merge_keyword,
                                        gee_grid_shape='../../Data_main/ref_shapes/WestUS_gee_grid_large.shp',
                                        refraster_westUS=WestUS_raster, refraster_gee_merge=GEE_merging_refraster_large_grids,
                                        use_cpu_while_multidownloading=15, westUS_shape=WestUS_shape):
    """
    Download drought indices (spi, spei, eddi)  data (at water yearly scale) from GEE.

    :param data_name: Data name. Have to be either of - 'spi', 'spei', 'eddi'.
    :param download_dir: File path of download directory.
    :param year_list: List of year_list to download data for.
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param gee_grid_shape: File path of gee grids that will be used to download the data.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    global key_word

    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, scale_factor, reducer, _, _, \
       year_start_range, year_end_range = get_gee_dict(data_name)

    # loading grids that will be used to download the data
    grids = gpd.read_file(gee_grid_shape)
    grids = grids.sort_values(by='FID', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['FID']

    for year in year_list:  # first loop for year_list

        # We are downloading drought indices in the '1y' bands, meaning the index is calculated based on
        # climate conditions (precipitation and PET) averaged over the previous 12 months.
        #
        # To align with the water year, we set the date to the **end of the water year** (~September 30).
        # But the data isn't available at September 30 each year due to a 5 day cadence. So, we set up a range
        # of 6 days, search the data within that range. If there is single data found in that range, we download it,
        # otherwise, we average it and download (if 2 datasets are found).
        #
        # Note: The **water year** starts in October of the **previous calendar year** and ends in September
        # of the **current calendar year**. For example, Water Year 2000 runs from **October 1999 to September 2000**.

        start_date = ee.Date.fromYMD(year, 9, 24)
        start_date_dt = datetime(year, 9, 24)

        end_date = ee.Date.fromYMD(year, 9, 30)
        end_date_dt = datetime(year, 9, 30)

        # will collect url and file name in url list and local_file_paths_list
        data_url_list = []
        local_file_paths_list = []

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        if (start_date_dt >= year_start_range) & (end_date_dt <= year_end_range):

            for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
                roi = geometry.bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                    filterBounds(gee_extent).reduce(reducer).multiply(scale_factor).toFloat()

                data_url = download_data.getDownloadURL({'name': data_name,
                                                         'crs': 'EPSG:4269',  # NAD83
                                                         'scale': 2200,  # in meter. equal to ~0.02 deg
                                                         'region': gee_extent,
                                                         'format': 'GEO_TIFF'})


                key_word = data_name
                local_file_path = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(grid_sr)}.tif')

                # Appending data url and local file path (to save data) to a central list
                data_url_list.append(data_url)
                local_file_paths_list.append(local_file_path)

                # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                # there is enough data url gathered for download.
                if (len(data_url_list) == 120) | (
                        grid_sr == len(grid_no)):  # downloads data when one of the conditions are met
                    # Combining url and file paths together to pass in multiprocessing
                    urls_to_file_paths_compile = []
                    for j, k in zip(data_url_list, local_file_paths_list):
                        urls_to_file_paths_compile.append([j, k])

                    # Download data by multi-processing/multi-threading
                    download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                           use_cpu=use_cpu_while_multidownloading)

                    # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                    # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                    data_url_list = []
                    local_file_paths_list = []

            # merging downloaded datasets
            mosaic_name = f'{key_word}_{year}.tif'
            mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
            clip_dir = os.path.join(download_dir, f'{merge_keyword}')

            makedirs([clip_dir, mosaic_dir])
            merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                                      raster_name=mosaic_name,
                                                                      ref_raster=refraster_gee_merge,
                                                                      search_by=f'*{year}*.tif', nodata=no_data_value)

            clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                           output_raster_dir=clip_dir, clip_and_resample=True,
                                           use_ref_width_height=False, resolution=model_res,
                                           ref_raster=refraster_westUS)

            print(f'{data_name} yearly data downloaded and merged')

        else:
            print(f'Data for year {year} is out of range. Skipping query')
            pass


def download_all_gee_data(data_list, download_dir, year_list, month_range,
                          skip_download=False):
    """
    Used to download all gee data together.

    :param data_list: List of valid data names to download.
    Current valid data names are -
        ['MODIS_Day_LST',
        'Landsat5_NDVI', 'Landsat8_NDVI',
        'Landsat5_OSAVI', 'Landsat8_OSAVI',
        'Landsat5_NDMI', 'Landsat8_NDMI',
        'Landsat5_GCVI', 'Landsat8_GCVI',
        'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDMI', 'MODIS_NDVI',
        'MODIS_LAI', 'GRIDMET_Precip', 'GRIDMET_Tmax', 'GRIDMET_RET', 'GRIDMET_maxRH',
        'GRIDMET_minRH', 'GRIDMET_windVel', 'GRIDMET_shortRad', 'GRIDMET_vpd',
        'DAYMET_sunHr', Field_capacity', 'Bulk_density',
        'Sand_content', 'Clay_content', 'DEM', 'spi', 'spei', 'eddi']

    :param download_dir: File path of main download directory. It will consist directory of individual dataset.
    :param year_list: List of year_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param skip_download: Set to True to skip download.

    :return: None
    """
    if not skip_download:
        for data_name in data_list:

            if data_name in ['MODIS_LAI', 'GRIDMET_Precip',
                             'GRIDMET_RET', 'GRIDMET_Tmax',
                             'GRIDMET_maxRH', 'GRIDMET_minRH',
                             'GRIDMET_windVel', 'GRIDMET_shortRad',
                             'GRIDMET_vpd', 'DAYMET_sunHr']:
                download_gee_data_monthly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                          month_range=month_range, merge_keyword='WestUS_monthly')

            elif data_name in ['Landsat5_NDVI', 'Landsat8_NDVI',
                               'Landsat5_OSAVI', 'Landsat8_OSAVI',
                               'Landsat5_NDMI', 'Landsat8_NDMI',
                               'Landsat5_GCVI', 'Landsat8_GCVI',
                               'MODIS_Terra_NDVI', 'MODIS_Terra_EVI',
                               'MODIS_NDMI', 'MODIS_NDVI', 'MODIS_Day_LST']:
                download_gee_data_yearly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                          month_range=(2, 10), merge_keyword='WestUS_yearly')  # months 2-10 chosen to keep a good overlap between regions with different growing season

            elif data_name == 'USDA_CDL':
                download_gee_data_yearly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                         month_range=month_range, merge_keyword='WestUS_yearly')

            elif data_name in ['Field_capacity', 'Bulk_density', 'Sand_content', 'Clay_content']:
                download_soil_datasets(data_name=data_name, download_dir=download_dir, merge_keyword='WestUS')

            elif data_name == 'DEM':
                download_DEM_Slope_data(data_name=data_name, download_dir=download_dir,
                                        merge_keyword='WestUS',
                                        terrain_slope=False)

            elif data_name == 'Tree_cover':
                download_tree_cover_data(data_name='Tree_cover', download_dir=download_dir,
                                         merge_keyword='WestUS')

            elif data_name in ['spi', 'spei', 'eddi']:
                download_drought_indices_water_year(data_name=data_name, download_dir=download_dir,
                                                    year_list=year_list, merge_keyword='WestUS_WaterYear')
    else:
        pass
