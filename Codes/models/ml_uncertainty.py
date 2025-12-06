import os
import sys
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
from glob import glob
from os.path import dirname, abspath

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.ML_ops import reindex_df, train_model
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster

# model resolution and reference raster/shapefile
no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'


def make_bootstrap_datasets(x_train_df, y_train_df, N_bootstrap, output_dir,
                            skip_processing=False):
    if not skip_processing:
        makedirs([output_dir])

        # the total number of data points in the dataframe
        N_train = len(x_train_df)

        # creating bootstrapped samples and saving them
        for n in range(N_bootstrap):
            # sample indices N_train times  with replacement using numpy
            idx = np.random.choice(N_train, size=N_train, replace=True)

            # bootstrap idx from the x_train and y_train
            x_boot, y_boot = x_train_df.iloc[idx, :], y_train_df.iloc[idx, :]

            # save bootstrapped samples
            x_boot.to_csv(os.path.join(output_dir, f'x_boot_{n}.csv'), index=False)
            y_boot.to_csv(os.path.join(output_dir, f'y_boot_{n}.csv'), index=False)


def jitter_params(base_params_dict, seed):
    """
    Randomly perturbs selected LightGBM parameters to simulate model uncertainty.
    This helps estimate prediction uncertainty via parameter jittering.

    Parameters
    ----------
    base_params_dict : dict
        Dictionary of trained model hyperparameters (e.g., from LightGBM DART).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Jittered parameter dictionary.
    """
    rn_gen = np.random.default_rng(seed)  # numpy generator object with random seed
    params = base_params_dict.copy()

    # jitter num_leaves
    params['num_leaves'] = int(params['num_leaves'] + rn_gen.integers(-5, 5))

    # jitter max_depth
    params['max_depth'] = params['max_depth'] + rn_gen.integers(-1, 2)

    # jitter learning rate
    params['learning_rate'] = params['learning_rate'] * rn_gen.uniform(0.9, 1.1)

    # jitter subsampling
    params['subsample'] = rn_gen.uniform(0.6, 0.9)

    # jitter colsample
    params['colsample_bytree'] = rn_gen.uniform(0.6, 0.9)

    # jitter drop_rate
    params['drop_rate'] = np.clip(params['drop_rate'] * rn_gen.uniform(0.8, 1.2), 0.01, 0.5)

    # jitter skip_drop
    params['skip_drop'] = np.clip(params['skip_drop'] + rn_gen.uniform(-0.05, 0.05), 0.0, 1.0)

    # jitter max_drop
    params['max_drop'] = max(5, int(params['max_drop'] + rn_gen.integers(-10, 10)))

    # params 'colsample_bynode', 'min_child_samples', 'n_estimators', 'path_smooth' have
    # been kept unchanged for now.

    return params


def predict_annual_mean_stdv(N_bootstrap, trained_model_dir, predictor_csv_and_nan_pos_dir,
                             exclude_columns, output_dir, irr_eff_dir, ref_raster=WestUS_raster,
                             skip_processing=False):
    if not skip_processing:
        # load all trained models and store them in a list
        models = []

        for n in range(N_bootstrap):
            # loading trained model
            model_Name = f'bootstrap_{n}.joblib'
            trained_model = joblib.load(os.path.join(trained_model_dir, model_Name))
            models.append(trained_model)

        print(f'Loaded all trained ({len(models)}) models..\n')

        # ref raster array load and shape extraction
        ref_arr, ref_file = read_raster_arr_object(ref_raster)
        ref_shape = ref_arr.shape

        for year in range(2000, 2024):  # for years 2000 to 2023
            print(f"Generating bootstrapped model's mean + stdv estimates for year {year}...")

            # loading input variable dataframe and nan position dict
            # also filtering out excluded columns
            predictor_csv = glob(os.path.join(predictor_csv_and_nan_pos_dir, f'*{year}.csv'))[0]
            nan_pos_dict_path = glob(os.path.join(predictor_csv_and_nan_pos_dir, f'*{year}.pkl'))[0]

            df = pd.read_csv(predictor_csv)
            df = df.drop(columns=exclude_columns)
            df = reindex_df(df)

            # empty storage list to add prediction array of that year using each model
            predictions = []

            # the prediction is consumptive use.need irrigation efficiency to convert it back to pumping.
            irr_eff_file = glob(os.path.join(irr_eff_dir, f'*{year}.tif'))[0]
            irr_eff_arr = read_raster_arr_object(irr_eff_file, get_file=False)

            # generating prediction raster with each trained model
            for model in models:
                pred_arr = model.predict(df)
                pred_arr = np.array(pred_arr)

                # converting consumptive use prediction back to pumping
                irr_eff_arr_shaped = irr_eff_arr.reshape(pred_arr.shape)
                pred_arr = np.where(~np.isnan(irr_eff_arr_shaped) & ~np.isnan(pred_arr), irr_eff_arr_shaped * pred_arr, -9999)

                # mask nodata per model (same as original function, but use np.nan here)
                nan_pos_dict = pickle.load(open(nan_pos_dict_path, "rb"))
                for _, nan_pos in nan_pos_dict.items():
                    pred_arr[nan_pos] = np.nan

                # adding in storage list
                predictions.append(pred_arr)

            # stack -> (N_bootstrap, n_samples)
            preds_stack = np.vstack(predictions)

            # calculate mean and stdv
            mean_arr = np.nanmean(preds_stack, axis=0)
            stdv_arr = np.nanstd(preds_stack, axis=0)

            # reshaping mean and stdv array to original raster shape
            mean_arr = mean_arr.reshape(ref_shape)
            stdv_arr = stdv_arr.reshape(ref_shape)

            # replacing np.nan with -9999
            mean_arr[np.isnan(mean_arr)] = -9999
            stdv_arr[np.isnan(stdv_arr)] = -9999

            # write mean and stdv rasters to array
            mean_output_tif = os.path.join(output_dir, f'mean_{year}.tif')
            write_array_to_raster(mean_arr, ref_file, ref_file.transform, mean_output_tif, nodata=-9999)

            stdv_output_tif = os.path.join(output_dir, f'stdv_{year}.tif')
            write_array_to_raster(stdv_arr, ref_file, ref_file.transform, stdv_output_tif, nodata=-9999)
    else:
        pass


def predict_total_mean_stdv_CV(N_bootstrap, trained_model_dir, predictor_csv_and_nan_pos_dir,
                               exclude_columns, output_dir, ref_raster=WestUS_raster,
                               skip_processing=False):
    if not skip_processing:
        # load all trained models and store them in a list
        models = []

        for n in range(N_bootstrap):
            # loading trained model
            model_Name = f'bootstrap_{n}.joblib'
            trained_model = joblib.load(os.path.join(trained_model_dir, model_Name))
            models.append(trained_model)

        print(f'Loaded all trained ({len(models)}) models..\n')

        # ref raster array load and shape extraction
        ref_arr, ref_file = read_raster_arr_object(ref_raster)
        ref_shape = ref_arr.shape

        for year in range(2000, 2021):  # for years 2000 to 2020
            print(f"Generating bootstrapped model's mean + stdv estimates for year {year}...")

            # loading input variable dataframe and nan position dict
            # also filtering out excluded columns
            predictor_csv = glob(os.path.join(predictor_csv_and_nan_pos_dir, f'*{year}.csv'))[0]
            nan_pos_dict_path = glob(os.path.join(predictor_csv_and_nan_pos_dir, f'*{year}.pkl'))[0]

            df = pd.read_csv(predictor_csv)
            df = df.drop(columns=exclude_columns)
            df = reindex_df(df)

            # empty storage list to add prediction array of that year using each model
            predictions = []

            # generating prediction raster with each trained model
            for model in models:
                pred_arr = model.predict(df)
                pred_arr = np.array(pred_arr)

                # mask nodata per model (same as original function, but use np.nan here)
                nan_pos_dict = pickle.load(open(nan_pos_dict_path, "rb"))
                for _, nan_pos in nan_pos_dict.items():
                    pred_arr[nan_pos] = np.nan

                # adding in storage list
                predictions.append(pred_arr)

        # stack -> (N_bootstrap, n_samples)
        preds_stack = np.vstack(predictions)

        # calculate mean and stdv
        mean_arr = np.nanmean(preds_stack, axis=0)
        stdv_arr = np.nanstd(preds_stack, axis=0)

        # reshaping mean and stdv array to original raster shape
        mean_arr = mean_arr.reshape(ref_shape)
        stdv_arr = stdv_arr.reshape(ref_shape)

        # replacing np.nan with -9999
        mean_arr[np.isnan(mean_arr)] = -9999
        stdv_arr[np.isnan(stdv_arr)] = -9999

        # write mean and stdv rasters to array
        mean_output_tif = os.path.join(output_dir, f'mean_all.tif')
        write_array_to_raster(mean_arr, ref_file, ref_file.transform, mean_output_tif, nodata=-9999)

        stdv_output_tif = os.path.join(output_dir, f'stdv_all.tif')
        write_array_to_raster(stdv_arr, ref_file, ref_file.transform, stdv_output_tif, nodata=-9999)

        # # calculating coef. of variation
        coef_variation_arr = np.where((mean_arr != -9999) & (stdv_arr != -9999),
                                       stdv_arr / mean_arr, -9999)

        coef_var_raster = os.path.join(output_dir, f'coef_variance.tif')
        write_array_to_raster(coef_variation_arr, ref_file, ref_file.transform,
                              coef_var_raster)

    else:
        pass


def create_uncertainty_bounds(model_prediction_dir, stdv_dir, output_dir,
                              skip_processing=False):
    """
    Calculates upper and lower 95% CI of model predicted pumping.

    :param model_prediction_dir: Path of directory with model predicted pumping rasters (the main prediction rasters).
    :param stdv_dir: Path of directory with standard deviation rasters.
    :param output_dir: Path of output directory.
    :param skip_processing: Set to True to skip processing of this function.

    :return: None.
    """
    if not skip_processing:
        print('\nCalculating lower and upper CI of predicted pumping...')

        for year in list(range(2000, 2024)):
            # loading data for the year
            pred_pumping, file = read_raster_arr_object(glob(os.path.join(model_prediction_dir, f'*{year}.tif'))[0])
            stdv = read_raster_arr_object(glob(os.path.join(stdv_dir, f'stdv_{year}.tif'))[0], get_file=False)

            # calculating the lower and upper bounds
            low_CI = np.where((pred_pumping != -9999) & (stdv != -9999), pred_pumping - 1.96 * stdv, -9999)
            high_CI = np.where((pred_pumping != -9999) & (stdv != -9999), pred_pumping + 1.96 * stdv, -9999)

            # save
            write_array_to_raster(low_CI, file, file.transform, os.path.join(output_dir, f'low_{year}.tif'))
            write_array_to_raster(high_CI, file, file.transform, os.path.join(output_dir, f'high_{year}.tif'))


########################################################################################################################
'''
Bootstrap
 - Load the train-test split (from trained model).
 - Bootstrap the train split multiple times and save each bootstrapped training sample.
 
Train LightGBM GBDT model
 - Train a model on each bootstrapped sample.
 - Add variation (jitter) in model params to introduce variety.
 - Save the trained models.
 
Predict using individual models
 - Predict pumping for WestUS using each model 
    (use the dataframe (csv) of predictors used to create WestUS pumping prediction).
 - Estimate the mean and stdv for each year (using bootstrapped models).
 - Save the mean and stdv estimates for each year.
'''
########################################################################################################################

if __name__ == '__main__':
    # --------------------------------------------------------------------------------------------------------------
    # flags
    # --------------------------------------------------------------------------------------------------------------
    model_version = 'v11'                           ##########

    skip_bootstrap_dataset = False                  ##########
    skip_train_bootstrap_models = False             ##########
    skip_create_annual_predictions = False          ##########
    skip_annual_calculate_low_high_CI = False       ##########
    skip_total_mean_stdv_cv = False                 ##########

    # control arguments
    n_bootstrap = 100
    columns_to_exclude = ['stateID', 'pixelID',
                          'shortRad', 'minRH', 'irr_eff',
                          'Canal_density', 'Canal_distance'
                          ]
    saved_model_dir = f'../../Model_run/ML_model/Model_trained/bootstrapped'
    input_csv_and_nan_pos_dir = f'../../Model_run/ML_model/Model_csv/annual_df'
    irr_eff_dir = '../../Data_main/rasters/HUC12_Irr_Eff'

    # --------------------------------------------------------------------------------------------------------------
    # Bootstrap dataset
    # --------------------------------------------------------------------------------------------------------------

    x_train = pd.read_csv(f'../../Model_run/ML_model/Model_csv/x_train_{model_version}.csv')
    y_train = pd.read_csv(f'../../Model_run/ML_model/Model_csv/y_train_{model_version}.csv')

    make_bootstrap_datasets(x_train_df=x_train, y_train_df=y_train,
                            N_bootstrap=n_bootstrap,
                            output_dir=f'../../Model_run/ML_model/Model_csv/boostrap',
                            skip_processing=skip_bootstrap_dataset)

    # --------------------------------------------------------------------------------------------------------------
    # Train model with bootstrapped datasets
    # --------------------------------------------------------------------------------------------------------------

    # directories
    bootstrap_data_dir = f'../../Model_run/ML_model/Model_csv/boostrap'
    save_model_to_dir = f'../../Model_run/ML_model/Model_trained/bootstrapped'
    mean_stdv_output_dir = f'../../Data_main/rasters/pumping_prediction/ML_uncertainty/{model_version}/mean_stdv'

    # base param dict (from the trained model)
    lgbm_param_dict = {'boosting_type': 'dart',
                       'subsample': 0.6424569150291416,
                       'drop_rate': 0.15193294688842177,
                       'max_drop': 55,
                       'skip_drop': 0.6997028395106997,
                       'colsample_bynode': 0.7201884877927008,
                       'colsample_bytree': 0.6594527779108277,
                       'data_sample_strategy': 'bagging',
                       'learning_rate': 0.019988466711388122,
                       'max_depth': 6,
                       'min_child_samples': 90,
                       'n_estimators': 400,
                       'num_leaves': 45,
                       'path_smooth': 0.7387036577638288,
                       'force_col_wise': True
                       }

    # random seed generator
    rng = np.random.default_rng(seed=34)
    seeds = rng.integers(10, 100, size=n_bootstrap)

    # training model with bootstrapped data and randomized jittered params
    if not skip_train_bootstrap_models:
        for i in range(n_bootstrap):
            # changing params slightly to add diversity among bootstrap models
            jittered_params = jitter_params(base_params_dict=lgbm_param_dict, seed=seeds[i])

            # loading bootstrapped data
            x_train = pd.read_csv(os.path.join(bootstrap_data_dir, f'x_boot_{i}.csv'))
            y_train = pd.read_csv(os.path.join(bootstrap_data_dir, f'y_boot_{i}.csv'))

            # training model
            model_name = f'bootstrap_{i}.joblib'
            train_model(x_train=x_train, y_train=y_train, params_dict=jittered_params,
                        load_model=False, save_model=True, save_folder=save_model_to_dir,
                        model_save_name=model_name, categorical_columns=None,
                        skip_tune_hyperparameters=True, iteration_csv=None,
                        verbose=False)

    # --------------------------------------------------------------------------------------------------------------
    # predict annual mean and stdv
    # --------------------------------------------------------------------------------------------------------------
    predict_annual_mean_stdv(N_bootstrap=n_bootstrap, trained_model_dir=saved_model_dir,
                             predictor_csv_and_nan_pos_dir=input_csv_and_nan_pos_dir,
                             exclude_columns=columns_to_exclude,
                             output_dir=mean_stdv_output_dir,
                             irr_eff_dir=irr_eff_dir,
                             ref_raster=WestUS_raster,
                             skip_processing=skip_create_annual_predictions)

    # --------------------------------------------------------------------------------------------------------------
    # Calculate annual lower and upper CI
    # --------------------------------------------------------------------------------------------------------------

    # directories
    predicted_pumping_dir = f'../../Data_main/rasters/pumping_prediction/ML/{model_version}/WestUS_pumping'
    low_high_CI_output_dir = f'../../Data_main/rasters/pumping_prediction/ML_uncertainty/{model_version}'

    # calculate lower and upper 95% CI
    create_uncertainty_bounds(model_prediction_dir=predicted_pumping_dir,
                              stdv_dir=mean_stdv_output_dir,
                              output_dir=low_high_CI_output_dir,
                              skip_processing=skip_annual_calculate_low_high_CI)

    # --------------------------------------------------------------------------------------------------------------
    # predict total mean and stdv + coefficient of variation
    # --------------------------------------------------------------------------------------------------------------

    predict_total_mean_stdv_CV(N_bootstrap=n_bootstrap, trained_model_dir=saved_model_dir,
                               predictor_csv_and_nan_pos_dir=input_csv_and_nan_pos_dir,
                               exclude_columns=columns_to_exclude,
                               output_dir=mean_stdv_output_dir,
                               ref_raster=WestUS_raster,
                               skip_processing=skip_total_mean_stdv_cv)