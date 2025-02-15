# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
from glob import glob

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import assign_cpu_nodes
from utils_tiles import make_multiband_datasets, make_training_tiles, \
        train_val_test_split_tiles, calc_scaling_statistics, standardize_train_val_test


if __name__ == '__main__':
    # flags
    skip_create_multiband_raster = False    #############################################################################
    skip_create_tile = False                #############################################################################
    skip_split_train_val_test = False       #############################################################################
    skip_calc_stats = False                 #############################################################################
    skip_standardize_train = False          #############################################################################
    skip_standardize_val = False            #############################################################################
    skip_standardize_test = False           #############################################################################


    # ------------------------------------------------------------------------------------------------------------------
    # 1. Multi-band raster creation for model training (includes pumping data in 1st band)
    # ------------------------------------------------------------------------------------------------------------------
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
                     '../../Data_main/rasters/sunHr/WestUS_growing_season': 'sunHr',
                     '../../Data_main/rasters/HUC12_SW': 'sw_huc12',
                     '../../Data_main/rasters/HUC12_GW_perc': 'gw_perc_huc12',
                     '../../Data_main/ref_rasters/stateID': 'stateID'}

    static_vars_dir = [i for i in datasets_dict.keys() if 'stateID' in i]
    temporal_vars_dir = [i for i in datasets_dict.keys() if i not in static_vars_dir]

    multiband_key_list = list(datasets_dict.values())  # 'pumping_mm' and 'stateID' included here

    westUS_multiband_dir = '../../Data_main/rasters/multibands_westUS/training/westUS'

    training_years = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                      2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019)

    # multi-band raster creation
    make_multiband_datasets(list_of_temporal_var_dirs=temporal_vars_dir,
                            list_of_static_var_dirs=static_vars_dir,
                            band_key_list=multiband_key_list,
                            output_dir=westUS_multiband_dir,
                            years_list=training_years,
                            skip_processing=skip_create_multiband_raster)


    # ------------------------------------------------------------------------------------------------------------------
    # 2. Multi-band tile creation for model training (includes pumping data in 1st band)
    # ------------------------------------------------------------------------------------------------------------------
    multiband_rasters = glob(os.path.join(westUS_multiband_dir, '*.tif'))
    interim_multiband_tile_dir = '../../Data_main/rasters/multibands_westUS/training/tiles/interim'
    interim_target_csv = '../../Data_main/rasters/multibands_westUS/training/tiles/interim/target.csv'
    final_multiband_tile_dir = '../../Data_main/rasters/multibands_westUS/training/tiles'
    final_target_csv = '../../Data_main/rasters/multibands_westUS/training/tiles/target.csv'

    tile_band_list = [i for i in list(datasets_dict.values())
                      if i not in ['pumping_mm', 'stateID']]  # making sure to not include 'pumping_mm' and 'stateID' here

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
    # 3. Train-validation-test split
    # ------------------------------------------------------------------------------------------------------------------
    multiband_tile_dir = final_multiband_tile_dir
    target_csv = final_target_csv
    train_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/train'
    val_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/val'
    test_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/test'

    use_cpu_nodes = assign_cpu_nodes([skip_split_train_val_test])

    train_val_test_split_tiles(target_data_csv=target_csv, input_tile_dir=multiband_tile_dir,
                               train_dir=train_dir, val_dir=val_dir, test_dir=test_dir,
                               train_size=0.7, val_size=0.2, test_size=0.1,
                               random_state=42, num_workers=use_cpu_nodes,
                               stratify=True,  # stratified split based on 'stateID'
                               skip_processing=skip_split_train_val_test)

    # ------------------------------------------------------------------------------------------------------------------
    # 4. Calculate standardization statistics
    # ------------------------------------------------------------------------------------------------------------------
    statistics_dir = '../../Data_main/rasters/multibands_westUS/scaling_stats'

    use_cpu_nodes = assign_cpu_nodes([skip_calc_stats])

    mean_dict, std_dict, _, _ = \
        calc_scaling_statistics(train_dir=train_dir, output_dir=statistics_dir,
                                num_workers=use_cpu_nodes,
                                skip_processing=skip_calc_stats)

    # ------------------------------------------------------------------------------------------------------------------
    # 6. Standardize
    # ------------------------------------------------------------------------------------------------------------------
    train_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/train'
    standardized_train_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/train'

    val_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/val'
    standardized_val_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/val'

    test_dir = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/test'
    standardized_test_dir = r'../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/test'


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
