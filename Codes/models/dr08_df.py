# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import numpy as np
import pandas as pd
from glob import glob
import dask.dataframe as ddf
from sklearn.model_selection import train_test_split

import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.ml_ops import create_annual_dataframes_for_pumping_prediction
from Codes.utils.raster_ops import read_raster_arr_object, make_lat_lon_array_from_raster


def create_train_test_dataframe(years_list, yearly_data_path_dict,
                                static_data_path_dict, datasets_to_include, output_parquet,
                                n_partitions=20, skip_processing=False):
    """
    Compile yearly/static datasets into a dataframe. This function-generated dataframe will be used as
    train-test data for ML model at annual scale.

    *** if there is no static data, set static_data_path_dict to None.

    :param years_list: A list of years_list for which data to include in the dataframe.
    :param yearly_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
                                  Can't be None.
    :param static_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
                                  Set to None if there is static dataset.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param output_parquet: Output filepath of the parquet file to save. Using parquet as it requires lesser memory.
                            Can also save smaller dataframe as csv file if name has '.csv' extension.
    :param n_partitions: Number of partitions to save the parquet file in using dask dataframe.
    :param skip_processing: Set to True to skip this dataframe creation process.

    :return: The filepath of the output parquet file.
    """
    if not skip_processing:
        print('\ncreating train-test dataframe for annual model...')

        makedirs([os.path.dirname(output_parquet)])

        variable_dict = {}

        # annual data compilation
        for var in yearly_data_path_dict.keys():
            if var in datasets_to_include:
                print(f'processing data for {var}..')

                for year_count, year in enumerate(years_list):
                    yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]

                    data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                    # extracting longitude and latitude information
                    lon_arr, lat_arr = make_lat_lon_array_from_raster(input_raster=yearly_data, nodata=-9999)

                    lon_arr = lon_arr.flatten()
                    lat_arr = lat_arr.flatten()

                    if (year_count == 0) & (var not in variable_dict.keys()):
                        variable_dict[var] = list(data_arr)
                        variable_dict['year'] = list([year] * len(list(data_arr)))

                        variable_dict['lon'] = list(lon_arr)
                        variable_dict['lat'] = list(lat_arr)

                    else:
                        variable_dict[var].extend(list(data_arr))
                        variable_dict['year'].extend(list([year] * len(list(data_arr))))

                        variable_dict['lon'].extend(list(lon_arr))
                        variable_dict['lat'].extend(list(lat_arr))

        # static data compilation
        if static_data_path_dict is not None:
            for var in static_data_path_dict.keys():
                if var in datasets_to_include:
                    print(f'processing data for {var}..')

                    static_data = glob(os.path.join(static_data_path_dict[var], '*.tif'))[0]
                    data_arr = read_raster_arr_object(static_data, get_file=False).flatten()

                    data_duplicated_for_total_years = list(data_arr) * len(years_list)
                    variable_dict[var] = data_duplicated_for_total_years

        train_test_ddf = ddf.from_dict(variable_dict, npartitions=n_partitions)
        train_test_ddf = train_test_ddf.dropna()

        if '.parquet' in output_parquet:
            train_test_ddf.to_parquet(output_parquet, write_index=False)

        elif '.csv' in output_parquet:
            train_test_df = train_test_ddf.compute()
            train_test_df.to_csv(output_parquet, index=False)

        return output_parquet

    else:
        return output_parquet


def split_train_val_test_set_v2(data_parquet, output_dir,
                                train_size=0.7, val_size=0.15, test_size=0.15,
                                random_state=42, skip_processing=False):
    """
    Splits dataset into train, validation, and test datasets, along with their target values.

    :return: None.
    """
    if not skip_processing:
        print(f'\nmaking train-validation-test ({train_size * 100}-{val_size * 100}-{test_size * 100} %) splits....\n')

        # loading data CSV
        df = pd.read_parquet(data_parquet)

        # replacing samples (very few) with stateID 3 - Nebraska and 1 -  Oklahoma. They belong to kansas
        # but came in Nebraska and Oklahoma during stateID raster creation (along state border)
        # otherwise train_val_test split function throws error
        df.loc[df['stateID'] == 3, 'stateID'] = 12
        df.loc[df['stateID'] == 1, 'stateID'] = 12

        # getting unique pixelID and stateID
        unique_pixels = df[['pixelID', 'stateID']].drop_duplicates()

        # Splitting at the pixelID level
        train_pixels, temp_pixels = train_test_split(unique_pixels, train_size=train_size,
                                                     stratify=unique_pixels['stateID'],
                                                     random_state=random_state)
        val_pixels, test_pixels = train_test_split(temp_pixels, test_size=test_size / (val_size + test_size),
                                                   stratify=temp_pixels['stateID'],
                                                   random_state=random_state)

        # assigning dataset labels to full dataset
        df['split'] = 'test'    # Default to test
        df.loc[df['pixelID'].isin(train_pixels['pixelID']), 'split'] = 'train'
        df.loc[df['pixelID'].isin(val_pixels['pixelID']), 'split'] = 'val'

        # splitting into separate DataFrames
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']

        # drop column 'split'
        train_df = train_df.drop(columns=['split'])
        val_df = val_df.drop(columns=['split'])
        test_df = test_df.drop(columns=['split'])

        # saving
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    else:
        pass


def save_statistics_to_csv(statistics_dicts, output_dir):
    """Saves multiple statistics dictionaries to CSV."""
    for dict_name, dictionary in zip(['mean', 'std'], statistics_dicts):
        df = pd.DataFrame(dictionary.items(), columns=['variable', 'value'])
        df.to_csv(os.path.join(output_dir, f'{dict_name}.csv'), index=False)


def load_statistics_from_csv(output_dir):
    """Loads mean, std statistics from CSV files into dictionaries."""
    mean_csv = pd.read_csv(os.path.join(output_dir, 'mean.csv'))
    std_csv = pd.read_csv(os.path.join(output_dir, 'std.csv'))

    return (
        dict(zip(mean_csv['variable'], mean_csv['value'])),
        dict(zip(std_csv['variable'], std_csv['value']))
    )


def calc_scaling_statistics(train_csv, features_to_exclude,
                            output_dir, skip_processing=False):
    """
    Calculates the mean and standard deviation for each features and the target value.


    :return: Four dictionaries: mean_csv, std_csv.
    """
    if not skip_processing:  # calculating the statistics

        makedirs([output_dir])

        # loading data csv to dataframe and selecting valid features to calculate statistics
        train_df = pd.read_csv(train_csv)

        features_to_exclude = set(features_to_exclude or [])  # convert to set to ensure uniqueness and handle None case
        features_to_exclude.add('pixelID')  # exclude 'pixelID'
        features_to_exclude.add('year')     # exclude 'year'
        features_to_exclude.add('lon')      # exclude 'lon'
        features_to_exclude.add('lat')      # exclude 'lat'

        valid_features = [i for i in train_df.columns if i not in features_to_exclude]

        print(f'Features processed for statistics: {valid_features} \n')

        # calculating statistics
        mean_dict = {col: np.nanmean(train_df[col]) for col in valid_features}
        std_dict = {col: np.nanstd(train_df[col]) for col in valid_features}

        # saving dictionaries as csv
        save_statistics_to_csv(statistics_dicts=[mean_dict, std_dict], output_dir=output_dir)

        return mean_dict, std_dict

    else:  # loading the saved statistics
        mean_dict, std_dict = load_statistics_from_csv(output_dir)

        return mean_dict, std_dict


def standardize_train_val_test(split_csv, mean_dict, std_dict,
                               exclude_features_from_standardizing, output_dir,
                               split_type='train', skip_processing=False):
    """
    Standardizing features and target with calculated statistics.
    """
    if not skip_processing:
        print(f"standardizing '{split_type}' dataset... \n")

        makedirs([output_dir])

        # loading original tran/val/test csv
        df = pd.read_csv(split_csv)

        # standardizing
        standardized_df = pd.DataFrame()

        for col in df.columns:
            if col not in exclude_features_from_standardizing:
                standardized_df[col] = (df[col] - mean_dict[col]) / std_dict[col]

            else:
                standardized_df[col] = df[col]

        # saving standardized data as csv
        standardized_df.to_csv(os.path.join(output_dir, f'{split_type}.csv'), index=False)

    else:
        pass


def standardize_annual_df(annual_csv, mean_dict, std_dict,
                          exclude_features_from_standardizing,
                          output_dir, skip_processing=False):
    """
    Standardizing features of annual dataframes with calculated statistics.
    """
    if not skip_processing:
        print(f"standardizing annual dataset... \n")

        makedirs([output_dir])

        # loading original tran/val/test csv
        df = pd.read_csv(annual_csv)

        # standardizing
        standardized_df = pd.DataFrame()

        for col in df.columns:
            if col not in exclude_features_from_standardizing:
                standardized_df[col] = (df[col] - mean_dict[col]) / std_dict[col]


        # saving standardized data as csv
        output_csv = os.path.join(output_dir, os.path.basename(annual_csv))
        standardized_df.to_csv(output_csv, index=False)

    else:
        pass


if __name__ == '__main__':
    # flags
    skip_df_creation = True                 ############################################################################
    skip_train_val_test_split = True        ############################################################################
    skip_calc_stats = True                  ############################################################################
    skip_standardizing = True              ############################################################################

    skip_annual_df_creation = True          ############################################################################
    skip_standardizing_annual_df = False     ############################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Directories and variables
    # ------------------------------------------------------------------------------------------------------------------

    # predictor data paths
    data_path_dict = {
        'target': '../../Data_main/pumping/rasters/WestUS_pumping',
        'netGW_Irr': '../../Data_main/rasters/NetGW_irrigation/WesternUS',
        'peff': '../../Data_main/rasters/Effective_precip_prediction_WestUS/v19_grow_season_scaled',
        'SW_Irr': '../../Data_main/rasters/SW_irrigation',
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
        'pixelID': '../../Data_main/ref_rasters/pixelID',
        'stateID': '../../Data_main/ref_rasters/stateID'
    }

    datasets_to_include = data_path_dict.keys()  # datasets to include in the main dataframe
    static_vars = {'FC', 'stateID', 'pixelID'}  # static vars
    annual_data_path_dict = {i: j for i, j in data_path_dict.items() if i not in static_vars}  # annual data paths
    static_data_path_dict = {i: j for i, j in data_path_dict.items() if i in static_vars}  # static data paths

    # exclude columns during scaling
    exclude_columns_in_scaling = ['stateID', 'pixelID', 'year', 'lon', 'lat', 'target']

    # training time periods
    years_list = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    # ------------------------------------------------------------------------------------------------------------------
    # Dataframe creation and train-test split
    # ------------------------------------------------------------------------------------------------------------------

    # create dataframe
    dataframe_parquet_path = f'../../Model_run/MLP_model/Model_csv/dataframe.parquet'

    create_train_test_dataframe(years_list=years_list,
                                yearly_data_path_dict=annual_data_path_dict,
                                static_data_path_dict=static_data_path_dict,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_parquet_path,
                                n_partitions=5,
                                skip_processing=skip_df_creation)

    split_train_val_test_set_v2(data_parquet=dataframe_parquet_path,
                                output_dir=f'../../Model_run/MLP_model/Model_csv',
                                train_size=0.7, val_size=0.15, test_size=0.15,
                                random_state=42, skip_processing=skip_train_val_test_split)


    # ------------------------------------------------------------------------------------------------------------------
    # Calculating standardization statistics
    # ------------------------------------------------------------------------------------------------------------------
    train_csv = f'../../Model_run/MLP_model/Model_csv/train.csv'
    standardized_output_dir = f'../../Model_run/MLP_model/Model_csv/standardized'

    mean_dict, std_dict = \
        calc_scaling_statistics(train_csv=train_csv, features_to_exclude=exclude_columns_in_scaling,
                                output_dir=standardized_output_dir,
                                skip_processing=skip_calc_stats)

    # ------------------------------------------------------------------------------------------------------------------
    # Standardizing train-val-test
    # ------------------------------------------------------------------------------------------------------------------
    train_csv = f'../../Model_run/MLP_model/Model_csv/train.csv'
    val_csv = f'../../Model_run/MLP_model/Model_csv/val.csv'
    test_csv = f'../../Model_run/MLP_model/Model_csv/test.csv'
    standardized_output_dir = f'../../Model_run/MLP_model/Model_csv/standardized'

    standardize_train_val_test(split_csv=train_csv, mean_dict=mean_dict, std_dict=std_dict,
                               exclude_features_from_standardizing=exclude_columns_in_scaling,
                               output_dir=standardized_output_dir, split_type='train',
                               skip_processing=skip_standardizing)

    standardize_train_val_test(split_csv=val_csv, mean_dict=mean_dict, std_dict=std_dict,
                               exclude_features_from_standardizing=exclude_columns_in_scaling,
                               output_dir=standardized_output_dir, split_type='val',
                               skip_processing=skip_standardizing)

    standardize_train_val_test(split_csv=test_csv, mean_dict=mean_dict, std_dict=std_dict,
                               exclude_features_from_standardizing=exclude_columns_in_scaling,
                               output_dir=standardized_output_dir, split_type='test',
                               skip_processing=skip_standardizing)

    # ------------------------------------------------------------------------------------------------------------------
    # Create WestUS annual dataframe for prediction + standardizing
    # ------------------------------------------------------------------------------------------------------------------

    # Annual dataframe creation
    static_vars = {'FC', 'stateID', 'pixelID'}  # static vars
    annual_data_path_dict = {i: j for i, j in data_path_dict.items() if i not in list(static_vars) + ['target']}  # annual data paths
    static_data_path_dict = {i: j for i, j in data_path_dict.items() if i in static_vars}  # static data paths

    datasets_to_include = list(data_path_dict.keys())  # datasets to include in the main dataframe
    datasets_to_include.remove('target')

    annual_dataframes_dir = f'../../Model_run/MLP_model/Model_csv/annual_csv'

    create_annual_dataframes_for_pumping_prediction(years_list=list(range(2000, 2020)),
                                                    yearly_data_path_dict=annual_data_path_dict,
                                                    static_data_path_dict=static_data_path_dict,
                                                    datasets_to_include=datasets_to_include,
                                                    irrigated_cropland_dir='../../Data_main/rasters/Irrigated_cropland',
                                                    output_dir=annual_dataframes_dir,
                                                    skip_processing=skip_annual_df_creation)

    # Standardization
    annual_dataframes = glob(os.path.join(annual_dataframes_dir, '*.csv'))
    output_dir = f'../../Model_run/MLP_model/Model_csv/annual_csv/standardized'

    for csv in annual_dataframes:
        standardize_annual_df(annual_csv=csv, mean_dict=mean_dict, std_dict=std_dict,
                              exclude_features_from_standardizing=exclude_columns_in_scaling,
                              output_dir=output_dir, skip_processing=skip_standardizing_annual_df)