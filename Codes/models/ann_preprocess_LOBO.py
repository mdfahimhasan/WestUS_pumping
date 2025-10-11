# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
from glob import glob

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.raster_ops import mask_raster_by_shape, set_nodata_inside_shapefile
from Codes.models.ann_df import create_train_test_dataframe, split_train_val_test_set_v2, \
    calc_scaling_statistics, standardize_train_val_test


def process_dataset_for_LOBO(years_list, years_no_pumping_data_dict,
                             model_version, basin_code, basin_shape, state_code,
                             annual_data_path_dict, static_data_path_dict,
                             exclude_columns_in_scaling, skip_processing=False):
    if not skip_processing:
        print('-----------------------------------------------------------')
        print(f'Processing datasets for Leave-One-Basin-Out test for {basin_code}...')
        print('-----------------------------------------------------------')

        # --------------------------------------------------------------------------------------------------------------
        # Boolean flags
        # --------------------------------------------------------------------------------------------------------------
        skip_process_pumping_data = False  ######
        skip_df_creation = False  ######
        skip_split = False  ######
        skip_calc_stats = False  ######
        skip_standardizing = False  ######

        # --------------------------------------------------------------------------------------------------------------
        # Pumping data adjustment
        # during training, the pumping data will consist no pixel from inside the respective basin
        # during testing, the holdout will only have pumping data from inside the basin
        # --------------------------------------------------------------------------------------------------------------
        if not skip_process_pumping_data:
            westUS_pumping_dir = f'../../Data_main/pumping/rasters/WestUS_pumping'  # this is the WestUS-scale pumping data
            westUS_pumping_rasters = glob(os.path.join(westUS_pumping_dir, '*.tif'))

            # pumping data extraction for fitting (training) and holdout
            fitting_output_dir = f'../../Data_main/rasters/ANN_LOBO/{basin_code}/fitting/pumping'
            holdout_dir = f'../../Data_main/rasters/ANN_LOBO/{basin_code}/holdout/pumping'

            for ras in westUS_pumping_rasters:
                raster_name = os.path.basename(ras)
                year_of_ras = int(raster_name.split('_')[-1].split('.')[0])

                # if pumping data not available for the year, skip
                # else, create train and holdout pumping data
                if year_of_ras in years_no_pumping_data_dict[state_code]:
                    continue

                else:
                    set_nodata_inside_shapefile(input_raster=ras, input_shape=basin_shape,
                                                output_dir=fitting_output_dir, raster_name=raster_name)

                    mask_raster_by_shape(input_raster=ras, input_shape=basin_shape, crop=False,
                                         output_dir=holdout_dir, raster_name=raster_name)
        else:
            fitting_output_dir = f'../../Data_main/rasters/ANN_LOBO/{basin_code}/fitting/pumping'
            holdout_dir = f'../../Data_main/rasters/ANN_LOBO/{basin_code}/holdout/pumping'

        # --------------------------------------------------------------------------------------------------------------
        # Dataframe creation + train-val split + standardization (for Train and validation dataset)
        # --------------------------------------------------------------------------------------------------------------
        # updating years_list to exclude years when no pumping data is available
        years_list_updated = [i for i in years_list if i not in years_no_pumping_data_dict[state_code]]

        # adding target (pumping_mm) to the annual dataset
        annual_training_data_path_dict = {**annual_data_path_dict,
                                          'target': fitting_output_dir}  # final annual data paths

        # datasets to include in the dataframe (not all will go into the final model)
        datasets_to_include = list(static_data_path_dict.keys()) + list(annual_training_data_path_dict.keys())
        # target (pumping_mm) will be excluded in train_val_test_split_v2

        # create dataframe
        train_test_parquet_path = f'../../Model_run/ANN_model/LOBO/{model_version}/{basin_code}/train_val.parquet'

        create_train_test_dataframe(years_list=years_list_updated,
                                    yearly_data_path_dict=annual_training_data_path_dict,
                                    static_data_path_dict=static_data_path_dict,
                                    datasets_to_include=datasets_to_include,
                                    output_parquet=train_test_parquet_path,
                                    n_partitions=5,
                                    skip_processing=skip_df_creation)

        # train-test split
        output_dir = f'../../Model_run/ANN_model/LOBO/{model_version}/{basin_code}'

        split_train_val_test_set_v2(data_parquet=train_test_parquet_path, output_dir=output_dir,
                                    train_size=0.7, val_size=0.25, test_size=0.05,
                                    random_state=42, skip_processing=skip_split)

        # calculate standardization statistics
        standardized_output_dir = os.path.join(output_dir, 'standardized')
        mean_dict, std_dict = \
            calc_scaling_statistics(train_csv=os.path.join(output_dir, 'train.csv'),
                                    features_to_exclude=exclude_columns_in_scaling,
                                    output_dir=standardized_output_dir,
                                    skip_processing=skip_calc_stats)

        # standardization
        standardize_train_val_test(split_csv=os.path.join(output_dir, 'train.csv'),
                                   mean_dict=mean_dict, std_dict=std_dict,
                                   exclude_features_from_standardizing=exclude_columns_in_scaling,
                                   output_dir=standardized_output_dir, split_type='train',
                                   skip_processing=skip_standardizing)

        standardize_train_val_test(split_csv=os.path.join(output_dir, 'val.csv'),
                                   mean_dict=mean_dict, std_dict=std_dict,
                                   exclude_features_from_standardizing=exclude_columns_in_scaling,
                                   output_dir=standardized_output_dir, split_type='val',
                                   skip_processing=skip_standardizing)

        # --------------------------------------------------------------------------------------------------------------
        # Dataframe creation + train-val split + standardization (for Test (holdout) dataset)
        # --------------------------------------------------------------------------------------------------------------

        # adding target (pumping_mm) to the annual dataset
        annual_training_data_path_dict = {**annual_data_path_dict,
                                          'target': holdout_dir}  # final annual data paths

        # datasets to include in the dataframe (not all will go into the final model)
        datasets_to_include = list(static_data_path_dict.keys()) + list(annual_training_data_path_dict.keys())
        # target (pumping_mm) will be excluded in train_val_test_split_v2

        # create dataframe
        holdout_csv_path = f'../../Model_run/ANN_model/LOBO/{model_version}/{basin_code}/holdout.csv'

        holdout_csv = create_train_test_dataframe(years_list=years_list_updated,
                                                  yearly_data_path_dict=annual_training_data_path_dict,
                                                  static_data_path_dict=static_data_path_dict,
                                                  datasets_to_include=datasets_to_include,
                                                  output_parquet=holdout_csv_path,
                                                  n_partitions=5,
                                                  skip_processing=skip_df_creation)

        # standardization
        standardize_train_val_test(split_csv=holdout_csv,
                                   mean_dict=mean_dict, std_dict=std_dict,
                                   exclude_features_from_standardizing=exclude_columns_in_scaling,
                                   output_dir=standardized_output_dir, split_type='holdout',
                                   skip_processing=skip_standardizing)


# dataset
data_path_dict = {
    # 'netGW_Irr': '../../Data_main/rasters/NetGW_irrigation/WesternUS',
    'peff': '../../Data_main/rasters/Effective_precip_prediction_WestUS/v19_grow_season_scaled',
    # 'SW_Irr': '../../Data_main/rasters/SW_irrigation',
    'ret': '../../Data_main/rasters/RET/WestUS_growing_season',
    'precip': '../../Data_main/rasters/Precip/WestUS_growing_season',
    'tmax': '../../Data_main/rasters/Tmax/WestUS_growing_season',
    'ET': '../../Data_main/rasters/OpenET_ensemble/WestUS_growing_season',
    'irr_crop_frac': '../../Data_main/rasters/Irrigated_cropland/Irrigated_Frac',
    'maxRH': '../../Data_main/rasters/maxRH/WestUS_growing_season',
    'minRH': '../../Data_main/rasters/minRH/WestUS_growing_season',
    'shortRad': '../../Data_main/rasters/shortRad/WestUS_growing_season',
    'vpd': '../../Data_main/rasters/vpd/WestUS_growing_season',
    'sunHr': '../../Data_main/rasters/sunHr/WestUS_growing_season',
    'FC': '../../Data_main/rasters/Field_capacity/WestUS',
    'Canal_density': '../../Data_main/rasters/Canal_density',
    'Canal_distance': '../../Data_main/rasters/Canal_distance',
    'pixelID': '../../Data_main/ref_rasters/pixelID',
    'stateID': '../../Data_main/ref_rasters/stateID'
}

datasets_to_include = data_path_dict.keys()  # datasets to include in the main dataframe
static_vars = {'FC', 'Canal_density', 'Canal_distance', 'stateID', 'pixelID'}  # static vars
annual_data_path_dict = {i: j for i, j in data_path_dict.items() if i not in static_vars}  # annual data paths
static_data_path_dict = {i: j for i, j in data_path_dict.items() if i in static_vars}  # static data paths

# exclude columns during scaling
exclude_columns_in_scaling = ['stateID', 'pixelID', 'year', 'target']

if __name__ == '__main__':
    # flags
    model_version = 'v7'
    skip_LOBO_GMD3 = False      ##### GMD3, KS
    skip_LOBO_GMD4 = False      ##### GMD4, KS
    skip_LOBO_RPB = False       ##### Republican Basin, CO
    skip_LOBO_SPB = False       ##### South Platte River Basin, CO
    skip_LOBO_AR = False        ##### Arkansas River Basin, CO
    skip_LOBO_SLV = False       ##### San Luis Valley, CO
    skip_LOBO_HQR = False       ##### Harquahala INA, AZ
    skip_LOBO_DOUG = False      ##### Douglas AMA, AZ
    skip_LOBO_PHX = False       ##### Phoenix AMA, AZ
    skip_LOBO_PNL = False       ##### Pinal AMA, AZ
    skip_LOBO_SCRUZ = False     ##### Santa Cruz AMA, AZ

    # time period
    years = list(range(2000, 2024))

    years_no_data_dict = {
        'KS': list(range(2021, 2024)),  # applies to GMD3, GMD4
        'CO': list(range(2000, 2011)),  # applies to RPB, SPB, AR, SLV
        'AZ': []  # applies to HQR, DOUG, PHX, PNL, SCRUZ
    }

    # # GMD3, KS
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='GMD3', state_code='KS',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/GMD3.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_GMD3)

    # # GMD4, KS
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='GMD4', state_code='KS',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/GMD4.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_GMD4)

    # # Republican River Basin, CO
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='RPB', state_code='CO',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_RPB)

    # # South Platte River Basin, CO
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='SPB', state_code='CO',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/South_Platte_Basin.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_SPB)
    # # Arkansas River Basin, CO
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='AR', state_code='CO',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/Arkansas_Basin.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_AR)

    # # San Luis Valley, CO
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='SLV', state_code='CO',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/Rio_Grande_Basin.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_SLV)

    # # Douglas AMA, AZ
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='DOUG', state_code='AZ',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_DOUG)

    # # Harquahala INA, AZ
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='HQR', state_code='AZ',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_HQR)

    # # Phoenix AMA, AZ
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='PHX', state_code='AZ',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/Phoenix_AMA.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_PHX)

    # # Pinal AMA, AZ
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='PNL', state_code='AZ',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/Pinal_AMA.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_PNL)

    # # Santa Cruz AMA, AZ
    process_dataset_for_LOBO(years_list=years, years_no_pumping_data_dict=years_no_data_dict,
                             model_version=model_version, basin_code='SCRUZ', state_code='AZ',
                             basin_shape='../../Data_main/shapefiles/Basins_of_interest/SantaCruz_AMA.shp',
                             annual_data_path_dict=annual_data_path_dict, static_data_path_dict=static_data_path_dict,
                             exclude_columns_in_scaling=exclude_columns_in_scaling,
                             skip_processing=skip_LOBO_SCRUZ)
