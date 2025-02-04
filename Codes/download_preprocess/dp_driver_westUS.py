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

from Codes.utils.system_ops import assign_cpu_nodes
from Codes.download_preprocess.preprocess import run_all_preprocessing
from Codes.download_preprocess.tiles_utils import make_multiband_datasets, make_training_tiles, \
        train_val_test_split, calc_scaling_statistics, standardize_train_val_test


# The `if __name__ == "__main__":` guard is required when using Python's multiprocessing module
# (used in make_training_tiles() class), especially on Windows and macOS. It ensures that the code inside this block
# is only executed when the script is run directly. This prevents recursive imports and ensures that worker processes
# are correctly spawned without re-running the entire script in each process.

# Includes train_val_test split and standardization

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Data download
    # ------------------------------------------------------------------------------------------------------------------

    # directories and variables
    data_download_dir = '../../Data_main/rasters'
    gee_grid_shape_large = '../../Data_main/ref_shapes/WestUS_gee_grid_large.shp'
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
        'GRIDMET_Tmax',
        'GRIDMET_maxRH',
        'GRIDMET_minRH',
        'GRIDMET_windVel',
        'GRIDMET_shortRad',
        'GRIDMET_vpd',
        'DAYMET_sunHr',
        'MODIS_Day_LST',
        'Field_capacity',
        'Sand_content',
        'Clay_content'
    ]

    openET_data_list = [
        'OpenET_ensemble',
        # 'Irrig_crop_OpenET_IrrMapper',
        # 'Irrig_crop_OpenET_LANID',
        'Irrigation_Frac_IrrMapper',
        'Irrigation_Frac_LANID'
    ]

    years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
             2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
             2016, 2017, 2018, 2019, 2020]
    months = (1, 12)

    # flags
    skip_download_gee_data = True  ######################################################################################
    skip_download_OpenET_data = True  ###################################################################################

    download_all_gee_data(data_list=gee_data_list, download_dir=data_download_dir,
                          year_list=years, month_range=months,
                          skip_download=skip_download_gee_data)

    download_all_openET_datasets(year_list=years, month_range=months,
                                 openET_data_list=openET_data_list,
                                 data_download_dir=data_download_dir,
                                 grid_shape_for_2km_ensemble=gee_grid_shape_large,
                                 grid_shape_for30m_irrmapper=gee_grid_shape_for30m_IrrMapper,
                                 grid_shape_for30m_lanid=gee_grid_shape_for30m_LANID,
                                 skip_download_OpenET_data=skip_download_OpenET_data)

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Data preprocessing
    # ------------------------------------------------------------------------------------------------------------------

    # directories and variables
    years = list(range(2000, 2019 + 1))  # collecting data from 2000-2019 as growing season Peff is available upto 2019 only

    # flags
    skip_stateID_raster_creation = True         ############################ this won't run on linux
    skip_process_GS_data = True                 ########################################################################
    skip_ET_processing = True                   ########################################################################
    skip_prism_precip_processing = True         ########################################################################
    skip_prism_tmax_processing = True           ########################################################################
    skip_GRIDMET_RET_processing = True          ########################################################################
    skip_GRIDMET_precip_processing = True       ########################################################################
    skip_GRIDMET_tmax_processing = True         ########################################################################
    skip_gridmet_maxRH_processing = True        ########################################################################
    skip_gridmet_minRH_processing = True        ########################################################################
    skip_gridmet_windVel_processing = True      ########################################################################
    skip_gridmet_shortRad_processing = True     ########################################################################
    skip_gridmet_vpd_processing = True          ########################################################################
    skip_daymet_sunHR_processing = True         ########################################################################

    run_all_preprocessing(skip_stateID_raster_creation=skip_stateID_raster_creation,
                          skip_process_GrowSeason_data=skip_process_GS_data,
                          skip_ET_processing=skip_ET_processing,
                          skip_prism_precip_processing=skip_prism_precip_processing,
                          skip_prism_tmax_processing=skip_prism_tmax_processing,
                          skip_gridmet_RET_processing=skip_GRIDMET_RET_processing,
                          skip_gridmet_precip_processing=skip_GRIDMET_precip_processing,
                          skip_gridmet_tmax_processing=skip_GRIDMET_tmax_processing,
                          skip_gridmet_maxRH_processing=skip_gridmet_maxRH_processing,
                          skip_gridmet_minRH_processing=skip_gridmet_minRH_processing,
                          skip_gridmet_windVel_processing=skip_gridmet_windVel_processing,
                          skip_gridmet_shortRad_processing=skip_gridmet_shortRad_processing,
                          skip_gridmet_vpd_processing=skip_gridmet_vpd_processing,
                          skip_daymet_sunHr_processing=skip_daymet_sunHR_processing
                          )


    # ------------------------------------------------------------------------------------------------------------------
    # 3. Multi-band raster creation for model training (includes pumping data in 1st band)
    # ------------------------------------------------------------------------------------------------------------------

    # directories and variables
    datasets_dict = {'../../Data_main/pumping/rasters/WestUS_pumping': 'pumping_mm',
                     '../../Data_main/rasters/NetGW_irrigation/WesternUS': 'netGWIrr',
                     '../../Data_main/rasters/Effective_precip_prediction_WestUS/v19_grow_season_scaled': 'peff',
                     '../../Data_main/rasters/RET/WestUS_growing_season': 'ret',
                     '../../Data_main/rasters/Precip/WestUS_growing_season': 'precip',
                     '../../Data_main/rasters/Tmax/WestUS_growing_season': 'tmax',
                     '../../Data_main/rasters/OpenET_ensemble/WestUS_growing_season': 'ET',
                     '../../Data_main/rasters/Irrigated_cropland/Irrigated_Frac': 'irr_crop_frac',
                     '../../Data_main/rasters/Irrigated_cropland': 'irr_cropland',
                     '../../Data_main/rasters/maxRH/WestUS_growing_season': 'maxRH',
                     '../../Data_main/rasters/minRH/WestUS_growing_season': 'minRH',
                     '../../Data_main/rasters/shortRad/WestUS_growing_season': 'shortRad',
                     '../../Data_main/rasters/vpd/WestUS_growing_season': 'vpd',
                     '../../Data_main/rasters/windVel/WestUS_growing_season': 'windVel',
                     '../../Data_main/rasters/sunHr/WestUS_growing_season': 'sunHr',
                     '../../Data_main/rasters/GCVI/WestUS_yearly': 'gcvi',
                     '../../Data_main/rasters/OSAVI/WestUS_yearly': 'osavi',
                     '../../Data_main/rasters/NDVI/WestUS_yearly': 'ndvi',
                     '../../Data_main/rasters/NDMI/WestUS_yearly': 'ndmi',
                     '../../Data_main/ref_rasters/stateID': 'stateID',
                     '../../Data_main/rasters/Clay_content/WestUS': 'clay',
                     '../../Data_main/rasters/Sand_content/WestUS': 'sand',
                     '../../Data_main/rasters/Field_capacity/WestUS': 'fc'}

    static_vars_dir = [i for i in datasets_dict.keys() if
                       any(var in i for var in ('stateID', 'Clay_content', 'Sand_content', 'Field_capacity'))]
    temporal_vars_dir = [i for i in datasets_dict.keys() if i not in static_vars_dir]

    multiband_key_list = list(datasets_dict.values())  # 'pumping_mm' and 'stateID' included here

    westUS_multiband_dir = '../../Data_main/rasters/multibands_westUS/training/westUS'

    training_years = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                      # 2012,   # 2012 skipped as no GCVI/NDVI/NDVI/NDMi data available due to gap in Landsat data; will think an alternative
                      2013, 2014, 2015, 2016, 2017, 2018, 2019)

    # flags
    skip_create_multiband_raster = True  ###############################################################################

    # multi-band raster creation
    make_multiband_datasets(list_of_temporal_var_dirs=temporal_vars_dir,
                            list_of_static_var_dirs=static_vars_dir,
                            band_key_list=multiband_key_list,
                            output_dir=westUS_multiband_dir,
                            years_list=training_years, skip_processing=skip_create_multiband_raster)


    # ------------------------------------------------------------------------------------------------------------------
    # 4. Multi-band tile creation for model training (includes pumping data in 1st band)
    # ------------------------------------------------------------------------------------------------------------------

    # directories and variables
    multiband_rasters = glob(os.path.join(westUS_multiband_dir, '*.tif'))
    interim_multiband_tile_dir = '../../Data_main/rasters/multibands_westUS/training/tiles/interim'
    interim_target_csv = '../../Data_main/rasters/multibands_westUS/training/tiles/interim/target.csv'
    final_multiband_tile_dir = '../../Data_main/rasters/multibands_westUS/training/tiles'
    final_target_csv = '../../Data_main/rasters/multibands_westUS/training/tiles/target.csv'

    tile_band_list = [i for i in list(datasets_dict.values())
                      if i not in ['pumping_mm', 'stateID']]  # making sure to not include 'pumping_mm' and 'stateID' here

    # flags
    skip_create_tile = True  ###########################################################################################
    use_cpu_nodes = assign_cpu_nodes([skip_create_tile])

    make_training_tiles(tiff_path_list=multiband_rasters, band_key_list=tile_band_list,
                        interim_tile_output_dir=interim_multiband_tile_dir,
                        interim_target_data_output_csv=interim_target_csv,
                        final_tile_output_dir=final_multiband_tile_dir,
                        final_target_data_output_csv=final_target_csv,
                        start_tile_no=1,
                        num_workers=use_cpu_nodes,
                        skip_processing=skip_create_tile)

    # ------------------------------------------------------------------------------------------------------------------
    # 5. Train-validation-test split
    # ------------------------------------------------------------------------------------------------------------------

    # directories
    multiband_tile_dir = final_multiband_tile_dir
    target_csv = final_target_csv
    train_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/train'
    val_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/val'
    test_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/test'

    # flags
    skip_split_train_val_test = False  ##################################################################################
    use_cpu_nodes = assign_cpu_nodes([skip_split_train_val_test])

    train_val_test_split(target_data_csv=target_csv, input_tile_dir=multiband_tile_dir,
                         train_dir=train_dir, val_dir=val_dir, test_dir=test_dir,
                         train_size=0.7, val_size=0.2, test_size=0.1,
                         random_state=42, num_workers=use_cpu_nodes,
                         stratify=True,                                # stratified split based on 'stateID'
                         skip_processing=skip_split_train_val_test)

    # ------------------------------------------------------------------------------------------------------------------
    # 6. Calculate standardization statistics
    # ------------------------------------------------------------------------------------------------------------------

    # directories
    statistics_dir = '../../Data_main/rasters/multibands_westUS/scaling_stats'

    # flags
    skip_calc_stats = False       ########################################################################################
    use_cpu_nodes = assign_cpu_nodes([skip_calc_stats])

    mean_dict, std_dict, _, _ = \
        calc_scaling_statistics(train_dir=train_dir, output_dir=statistics_dir,
                                num_workers=use_cpu_nodes,
                                skip_processing=skip_calc_stats)

    # ------------------------------------------------------------------------------------------------------------------
    # 7. Standardize
    # ------------------------------------------------------------------------------------------------------------------

    # directories
    train_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/train'
    standardized_train_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/train'

    val_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/val'
    standardized_val_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/val'

    test_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/test'
    standardized_test_dir = r'../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/test'

    # flags
    skip_standardize_train = False  #####################################################################################
    skip_standardize_val = False    #####################################################################################
    skip_standardize_test = False   #####################################################################################

    use_cpu_nodes = assign_cpu_nodes([skip_standardize_train, skip_standardize_val, skip_standardize_test])

    standardize_train_val_test(input_tile_dir=train_dir,
                               mean_dict=mean_dict, std_dict=std_dict,
                               split_type='train', output_dir=standardized_train_dir,
                               num_workers=use_cpu_nodes,
                               skip_processing=skip_standardize_train)

    standardize_train_val_test(input_tile_dir=val_dir,
                               mean_dict=mean_dict, std_dict=std_dict,
                               split_type='val', output_dir=standardized_val_dir,
                               num_workers=use_cpu_nodes,
                               skip_processing=skip_standardize_val)

    standardize_train_val_test(input_tile_dir=test_dir,
                               mean_dict=mean_dict, std_dict=std_dict,
                               split_type='test', output_dir=standardized_test_dir,
                               num_workers=use_cpu_nodes,
                               skip_processing=skip_standardize_test)
