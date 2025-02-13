# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
from download import download_all_gee_data
from download_openET import download_all_openET_datasets

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.download_preprocess.preprocess import run_all_preprocessing


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
        # 'Landsat5_NDVI',
        # 'Landsat8_NDVI',
        # 'Landsat5_OSAVI',
        # 'Landsat8_OSAVI',
        # 'Landsat5_NDMI',
        # 'Landsat8_NDMI',
        # 'Landsat5_GCVI',
        # 'Landsat8_GCVI',
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
        # 'Field_capacity',
        # 'Sand_content',
        # 'Clay_content'
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
    skip_HUC12_SW_processing = True             ########################################################################
    skip_HUC12_GW_perc_processing = True        ########################################################################
    skip_koppen_geiger_processing = True        ########################################################################

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
                          skip_daymet_sunHr_processing=skip_daymet_sunHR_processing,
                          skip_HUC12_SW_processing=skip_HUC12_SW_processing,
                          skip_HUC12_GW_perc_processing=skip_HUC12_GW_perc_processing,
                          skip_koppen_geiger_processing=skip_koppen_geiger_processing
                          )
