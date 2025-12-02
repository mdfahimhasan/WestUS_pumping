# author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
from os.path import dirname, abspath

import pandas as pd

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.plots import scatter_plot_of_same_vars
from Codes.utils.ML_ops import (create_train_test_dataframe, split_train_val_test_set_v2, \
                                train_model, test_model, plot_permutation_importance, \
                                cross_val_performance, plot_shap_summary_plot, plot_shap_interaction_plot, \
                                create_annual_dataframes_for_pumping_prediction, predict_annual_pumping_rasters, \
                                compute_pumping_from_consumptive_use)

# model resolution and reference raster/shapefile
no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'

# --------------------------------------------------------------------------------------------------------------
# Directories and variables
# --------------------------------------------------------------------------------------------------------------

# predictor data paths
data_path_dict = {
    # training data
    'pumping_mm': '../../Data_main/pumping/rasters/WestUS_pumping',
    'consumptive_gw': '../../Data_main/pumping/rasters/WestUS_consumptive_gw',

    # predictors
    'irr_eff': '../../Data_main/rasters/HUC12_Irr_Eff',
    'peff': '../../Data_main/rasters/Effective_precip_prediction_WestUS/v19_grow_season_scaled',
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

# training attribute selection
# we previously used 'pumping_mm' as attribute. Now using, 'consumptive_gw'

train_attr = 'consumptive_gw'                           ##############################
alternate_train_attr = 'pumping_mm'                     ##############################

# exclude columns during training
# train_attr will be excluded in train_val_test_split_v2() function
exclude_columns_in_training = ['stateID', 'pixelID', 'year',
                               'shortRad', 'minRH', 'irr_eff',
                               'Canal_density', 'Canal_distance'] + [alternate_train_attr]
# training time periods
years_list = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
              2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

if __name__ == '__main__':
    # --------------------------------------------------------------------------------------------------------------
    # flags
    # --------------------------------------------------------------------------------------------------------------
    model_version = 'v11'  ######

    skip_df_creation = True                             ######
    skip_train_test_split = True                        ######
    skip_hyperparam_tune = True                         ######
    load_model = False                                   ######
    save_model = True                                   ######
    skip_scatter_plots = False                           ######
    skip_cross_val = False                               ######
    skip_perm_imp_plot = False                           ######
    skip_SHAP_importance = False                         ######
    skip_SHAP_interact_plot = False                      ######
    skip_create_df_for_prediction = True                ######
    skip_create_prediction_raster = False                ######
    skip_convert_prediction_raster_to_pumping = False   ######

    # --------------------------------------------------------------------------------------------------------------
    # Dataframe creation and train-test split
    # --------------------------------------------------------------------------------------------------------------

    # create dataframe
    train_test_parquet_path = f'../../Model_run/ML_model/Model_csv/train_test_{model_version}.parquet'

    create_train_test_dataframe(years_list=years_list,
                                yearly_data_path_dict=annual_data_path_dict,
                                static_data_path_dict=static_data_path_dict,
                                datasets_to_include=datasets_to_include,
                                output_parquet=train_test_parquet_path,
                                n_partitions=5,
                                skip_processing=skip_df_creation)

    # train-test split
    output_dir = f'../../Model_run/ML_model/Model_csv'

    x_train, x_test, y_train, y_test = \
        split_train_val_test_set_v2(data_parquet=train_test_parquet_path, pred_attr=train_attr,
                                    exclude_columns=exclude_columns_in_training, output_dir=output_dir,
                                    model_version=model_version, train_size=0.7, test_size=0.3,
                                    random_state=42, skip_processing=skip_train_test_split)

    # --------------------------------------------------------------------------------------------------------------
    # Model training and hyperparameter tuning performance evaluation
    # --------------------------------------------------------------------------------------------------------------

    # model training  (if hyperparameter tuning is on, the default parameter dictionary will be disregarded)
    print('\n########## Model training')
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

    save_model_to_dir = f'../../Model_run/ML_model/Model_trained'
    model_name = f'westus_pumping_{model_version}.joblib'
    param_iteration_csv = f'../../Model_run/ML_model/Model_trained/hyperparam_iteration_{model_version}.csv'
    makedirs([save_model_to_dir])

    lgbm_reg_trained = train_model(x_train=x_train, y_train=y_train, params_dict=lgbm_param_dict,
                                   load_model=load_model, save_model=save_model, save_folder=save_model_to_dir,
                                   model_save_name=model_name, categorical_columns=None,
                                   skip_tune_hyperparameters=skip_hyperparam_tune,
                                   iteration_csv=param_iteration_csv, n_fold=10, max_evals=400)
    print(lgbm_reg_trained, '\n')

    # --------------------------------------------------------------------------------------------------------------
    # Model performance evaluation
    # --------------------------------------------------------------------------------------------------------------

    print('########## Model performance')

    # checking train accuracy
    train_prediction_csv_path = f'../../Model_run/ML_model/Model_csv/train_obsv_pred_{model_version}.csv'

    print('\nTrain performance:')
    print('------------------------------')

    test_model(trained_model=lgbm_reg_trained, x_test=x_train, y_test=y_train,
               prediction_csv_path=train_prediction_csv_path,
               categorical_columns=None)

    # checking test accuracy
    test_prediction_csv_path = f'../../Model_run/ML_model/Model_csv/test_obsv_pred_{model_version}.csv'

    print('\nTest performance:')
    print('------------------------------')

    test_model(trained_model=lgbm_reg_trained, x_test=x_test, y_test=y_test,
               prediction_csv_path=test_prediction_csv_path,
               categorical_columns=None)

    # cross validation performance
    cross_val_performance(trained_model_path=os.path.join(save_model_to_dir, model_name),
                          x_train_df=x_train, y_train_df=y_train, k_fold=10,
                          categorical_columns=None,
                          verbose=False, skip_processing=skip_cross_val)

    # --------------------------------------------------------------------------------------------------------------
    # Plotting scatters + permutation importance + PDP
    # --------------------------------------------------------------------------------------------------------------

    plot_dir = f'../../Model_run/ML_model/Plots'

    if not skip_scatter_plots:
        # plotting train scatters
        y_train_df = pd.read_csv(train_prediction_csv_path)
        scatter_plot_of_same_vars(Y_pred=y_train_df['predicted'], Y_obsv=y_train_df['actual'],
                                  x_label='Actual pumping (mm/year)', y_label='Predicted pumping (mm/year)',
                                  plot_name=f'train_scatter_{model_version}.jpg', savedir=plot_dir, alpha=0.5,
                                  marker_size=5, title=None,
                                  axis_lim=(0, 1000), tick_interval=200)

        # plotting test scatters
        y_test_df = pd.read_csv(test_prediction_csv_path)
        scatter_plot_of_same_vars(Y_pred=y_test_df['predicted'], Y_obsv=y_test_df['actual'],
                                  x_label='Actual pumping (mm/year)', y_label='Predicted pumping (mm/year)',
                                  plot_name=f'test_scatter_{model_version}.jpg', savedir=plot_dir, alpha=0.5,
                                  marker_size=5, title=None,
                                  axis_lim=(0, 1000), tick_interval=200)

    # permutation importance
    sorted_imp_vars = \
        plot_permutation_importance(trained_model=lgbm_reg_trained, x_test=x_test, y_test=y_test,
                                    output_dir=plot_dir, plot_name=f'perm_imp_{model_version}.jpg',
                                    sorted_var_list_name=f'sorted_imp_vars_{model_version}.pkl',
                                    categorical_columns=None,
                                    skip_processing=skip_perm_imp_plot)

    # Shap summary plot
    plot_shap_summary_plot(trained_model_path=os.path.join(save_model_to_dir, model_name),
                           use_samples=2000,
                           data_csv=x_train,
                           exclude_features=exclude_columns_in_training,
                           save_plot_path=os.path.join(plot_dir, f'SHAP_imp_{model_version}.jpg'),
                           skip_processing=skip_SHAP_importance)

    # Shap interaction plot
    features_to_plot = ['Effective precipitation', 'Irrigated crop fraction',
                        'ET', 'Field capacity', 'Precipitation',
                        'Reference ET', 'Temperature (max)']

    plot_shap_interaction_plot(model_version=model_version,
                               trained_model_path=os.path.join(save_model_to_dir, model_name),
                               use_samples=2000, features_to_plot=features_to_plot,
                               data_csv=x_train, exclude_features_from_df=exclude_columns_in_training,
                               save_plot_dir=os.path.join(plot_dir, 'SHAP_interaction'),
                               skip_processing=skip_SHAP_interact_plot)

    # --------------------------------------------------------------------------------------------------------------
    # Annual dataframe creation + prediction
    # -------------------------------------------------------------------------------------------------------
    # create prediction rasters
    # create dataframe and nan position dict

    datasets_to_include = list(datasets_to_include)
    for col in ['pumping_mm', 'consumptive_gw']:
        if col in datasets_to_include:
            datasets_to_include.remove(col)

    predictor_csv_and_nan_pos_dir = f'../../Model_run/ML_model/Model_csv/annual_df'

    create_annual_dataframes_for_pumping_prediction(years_list=years_list,
                                                    yearly_data_path_dict=annual_data_path_dict,
                                                    static_data_path_dict=static_data_path_dict,
                                                    datasets_to_include=datasets_to_include,
                                                    irrigated_cropland_dir='../../Data_main/rasters/Irrigated_cropland',
                                                    output_dir=predictor_csv_and_nan_pos_dir,
                                                    skip_processing=skip_create_df_for_prediction)

    # prediction (consumptive groundwater use)
    prediction_output_dir = f'../../Data_main/rasters/pumping_prediction/ML/{model_version}/consumptive_gw'

    exclude_columns_in_prediction = [i for i in exclude_columns_in_training if i != 'year']

    predict_annual_pumping_rasters(trained_model=lgbm_reg_trained, years_list=years_list,
                                   exclude_columns=exclude_columns_in_prediction,
                                   predictor_csv_and_nan_pos_dir=predictor_csv_and_nan_pos_dir,
                                   prediction_name_keyword='consumptive_gw',
                                   output_dir=prediction_output_dir,
                                   ref_raster=WestUS_raster,
                                   skip_processing=skip_create_prediction_raster)

    # convert consumptive groundwater use prediction to pumping estimates
    irr_eff_dir = '../../Data_main/rasters/HUC12_Irr_Eff'
    pumping_output_dir = f'../../Data_main/rasters/pumping_prediction/ML/{model_version}'

    compute_pumping_from_consumptive_use(consmp_gw_prediction_dir=prediction_output_dir,
                                         irr_eff_dir=irr_eff_dir, output_dir=pumping_output_dir,
                                         skip_processing=skip_convert_prediction_raster_to_pumping)
