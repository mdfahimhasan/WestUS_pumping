# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
from glob import glob
from download import download_all_gee_data
from download_openET import download_all_openET_datasets

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.download_preprocess.preprocess import run_all_preprocessing
from Codes.download_preprocess.tiles_utils import make_multiband_datasets, make_training_tiles

# ----------------------------------------------------------------------------------------------------------------------
# 1. Data download
# ----------------------------------------------------------------------------------------------------------------------

# directories and variables
data_download_dir = '../../Data_main/rasters'
gee_grid_shape_large = '../../Data_main/ref_shapes/WestUS_gee_grid.shp'
gee_grid_shape_for30m_IrrMapper = '../../Data_main/ref_shapes/WestUS_gee_grid_for30m_IrrMapper.shp'
gee_grid_shape_for30m_LANID = '../../Data_main/ref_shapes/WestUS_gee_grid_for30m_LANID.shp'

gee_data_list = [
    'Landsat5_NDVI',
    'Landsat8_NDVI',
    'Landsat5_OSAVI',
    'Landsat8_OSAVI',
    'Landsat5_NDMI',
    'Landsat8_NDMI',
    'Landsat5_GCVI',
    'Landsat8_GCVI',
    'GRIDMET_Precip',
    'GRIDMET_RET',
    'GRIDMET_Tmax'
    'MODIS_Day_LST',
    'Field_capacity',
    'Sand_content',
    'Clay_content'
]

openET_data_list = ['Irrig_crop_OpenET_IrrMapper',
                    'Irrig_crop_OpenET_LANID',
                    'Irrigation_Frac_IrrMapper',
                    'Irrigation_Frac_LANID']

years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
         2016, 2017, 2018, 2019, 2020]
months = (1, 12)

# flags
skip_download_gee_data = True  #########################################################################################
skip_download_OpenET_data = True  ######################################################################################

download_all_gee_data(data_list=gee_data_list, download_dir=data_download_dir,
                      year_list=years, month_range=months,
                      skip_download=skip_download_gee_data)

download_all_openET_datasets(year_list=years, month_range=months,
                             openET_data_list=openET_data_list,
                             data_download_dir=data_download_dir,
                             grid_shape_for_2km_ensemble=None,
                             grid_shape_for30m_irrmapper=gee_grid_shape_for30m_IrrMapper,
                             grid_shape_for30m_lanid=gee_grid_shape_for30m_LANID,
                             skip_download_OpenET_data=skip_download_OpenET_data)

# ----------------------------------------------------------------------------------------------------------------------
# 2. Data preprocessing
# ----------------------------------------------------------------------------------------------------------------------

# directories and variables
years = list(range(2000, 2019 + 1))  # collecting data from 2000-2019 as growing season Peff is available upto 2019 only

# flags
skip_process_GS_data = True  ###########################################################################################
skip_prism_precip_processing = True  ###################################################################################
skip_prism_tmax_processing = True  #####################################################################################
skip_GRIDMET_RET_processing = True  ####################################################################################
skip_GRIDMET_precip_processing = True  #################################################################################
skip_GRIDMET_tmax_processing = True  ###################################################################################

run_all_preprocessing(skip_process_GrowSeason_data=skip_process_GS_data,
                      skip_prism_precip_processing=skip_prism_precip_processing,
                      skip_prism_tmax_processing=skip_prism_tmax_processing,
                      skip_gridmet_RET_processing=skip_GRIDMET_RET_processing,
                      skip_gridmet_precip_processing=skip_GRIDMET_precip_processing,
                      skip_gridmet_tmax_processing=skip_GRIDMET_tmax_processing)

# ----------------------------------------------------------------------------------------------------------------------
# 3. Multi-band raster creation for model training (includes pumping data in 1st band)
# ----------------------------------------------------------------------------------------------------------------------

# directories and variables
datasets_dict = {'../../Data_main/pumping/rasters/Colorado/pumping_mm': 'pumping_mm',
                 # the pumping data only has data from Colorado for now
                 '../../Data_main/rasters/NetGW_irrigation/WesternUS': 'netGWIrr',
                 '../../Data_main/rasters/Effective_precip_prediction_WestUS/v19_grow_season_scaled': 'peff',
                 '../../Data_main/rasters/RET/WestUS_growing_season': 'ret',
                 '../../Data_main/rasters/Precip/WestUS_growing_season': 'precip',
                 '../../Data_main/rasters/Tmax/WestUS_growing_season': 'tmax',
                 '../../Data_main/rasters/Irrigated_cropET/WestUS_grow_season': 'irr_cropET',
                 '../../Data_main/rasters/Irrigated_cropland/Irrigated_Frac': 'irr_crop_frac',
                 '../../Data_main/rasters/Irrigated_cropland': 'irr_cropland',
                 '../../Data_main/rasters/GCVI/WestUS_yearly': 'gcvi',
                 '../../Data_main/rasters/OSAVI/WestUS_yearly': 'osavi',
                 '../../Data_main/rasters/NDVI/WestUS_yearly': 'ndvi',
                 '../../Data_main/rasters/NDMI/WestUS_yearly': 'ndmi',
                 '../../Data_main/rasters/Clay_content/WestUS': 'clay',
                 '../../Data_main/rasters/Sand_content/WestUS': 'sand',
                 '../../Data_main/rasters/Field_capacity/WestUS': 'fc'}

static_vars_dir = [i for i in datasets_dict.keys() if
                   any(var in i for var in ('Clay_content', 'Sand_content', 'Field_capacity'))]
temporal_vars_dir = [i for i in datasets_dict.keys() if i not in static_vars_dir]

band_key_list = list(datasets_dict.values())  # make sure to not include the training data's name as that band will be removed

westUS_multiband_dir = '../../Data_main/rasters/multibands/training/westUS'

training_years = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                  # 2012,   # 2012 skipped as no GCVI/NDVI/NDVI/NDMi data available due to gap in Landsat data; will think an alternative
                  2013, 2014, 2015, 2016, 2017, 2018, 2019)

# flags
skip_create_multiband_raster = True  ####################################################################################

# multi-band raster creation
make_multiband_datasets(list_of_temporal_var_dirs=temporal_vars_dir, list_of_static_var_dirs=static_vars_dir,
                        band_key_list=band_key_list,
                        output_dir=westUS_multiband_dir,
                        years_list=training_years, skip_processing=skip_create_multiband_raster)

# ----------------------------------------------------------------------------------------------------------------------
# 4. Multi-band tile creation for model training (includes pumping data in 1st band)
# ----------------------------------------------------------------------------------------------------------------------

# directories
multiband_rasters = glob(os.path.join(westUS_multiband_dir, '*.tif'))
multiband_tile_dir = '../../Data_main/rasters/multibands/training/tiles'
target_csv = '../../Data_main/rasters/multibands/training/tiles/target.csv'

band_key_list = band_key_list[1:]

# flags
skip_create_tile = False  ###############################################################################################

# external tile counter
current_tile_no = 1  # Start tile numbering

for multi_ras in multiband_rasters:
    tile_maker = make_training_tiles(tiff_path=multi_ras, band_key_list=band_key_list,
                                     tile_output_dir=multiband_tile_dir,
                                     target_data_output_csv=target_csv,
                                     start_tile_no=current_tile_no,
                                     skip_processing=skip_create_tile)

    current_tile_no = tile_maker.tile_no





