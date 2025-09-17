# author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
import pandas as pd
from glob import glob

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.stats_ops import calculate_metrics
from Codes.utils.plots import scatter_plot_of_same_vars
from Codes.utils.raster_ops import mask_raster_by_shape, set_nodata_inside_shapefile
from Codes.utils.ml_ops import reindex_df, create_train_test_dataframe, split_train_val_test_set_v2, \
    train_model, predict_annual_pumping_rasters


WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'


def predict_LOBO_basin(trained_model, predictor_array, target_array, year_series,
                       prediction_csv_path, categorical_columns=None):
    """
    Test a trained LightGBM regressor model's performance.

    :param trained_model: trained lightgbm model.
    :param predictor_array : predictor array (input variable array) from the split_train_test_ratio() function
    :param target_array: x_test (predictor) and y_test (target) arrays from split_train_test_ratio() function.
    :param year_series: Dataframe series with year info of the LOBO dataframe.
    :param prediction_csv_path: Csv filepath to save the prediction.
    :param categorical_columns: List of categorical column names to convert to 'category' dtype. Default set to None.

    :return: trained LGBM regression model.
    """
    makedirs([os.path.dirname(prediction_csv_path)])

    # provision to include categorical data
    if categorical_columns is not None:
        for col in categorical_columns:
            predictor_array[col] = predictor_array[col].astype('category')

    # testing model performance
    y_pred_test = trained_model.predict(predictor_array)

    # performance/error metrics
    metrics_dict = calculate_metrics(predictions=y_pred_test, targets=target_array.values.ravel())

    rmse = metrics_dict['RMSE']
    mae = metrics_dict['MAE']
    r2 = metrics_dict['R2']
    nrmse = metrics_dict['Normalized RMSE']
    nmae = metrics_dict['Normalized MAE']

    print(
        f"Test Results:\n"
        f"---------------------\n"
        f"RMSE: {rmse:.4f}, MAE: {mae:.4f},\n"
        f"NRMSE: {nrmse:.4f}, NMAE: {nmae:.4f}, R²: {r2:.4f}\n"
    )

    # saving test prediction
    test_obsv_predict_df = pd.DataFrame({'actual': target_array.values.ravel(),
                                         'predicted': y_pred_test})
    test_obsv_predict_df['year'] = year_series
    test_obsv_predict_df.to_csv(prediction_csv_path, index=False)

def perform_LOBO(model_param_dict, model_version, basin_code, annual_data_path_dict,
                 static_data_path_dict, exclude_columns, basin_shape,
                 predictor_csv_and_nan_pos_dir, skip_processing=False):
    """
    Processes data for respective basins and perform Leave-One-Basin-Out test.

    Leave-One-Basin-Out (LOBO):
    LOBO is a cross-validation approach specifically designed for evaluating generalization capability of this model.
    It is a variant of Leave-One-Out Cross-Validation (LOO-CV), but instead of leaving out individual samples, it
    leaves out an entire hydrological basin during model training and then evaluates the model's performance on
    the excluded basin.

    :param model_param_dict: Model parameter dictionary. Shoudl come from hyperparam tuned model.
    :param model_version: str of model version, e.g., 'LOBO_GMD3_v1'.
    :param basin_code: Basin code, such as 'GMD3', 'RPB', or 'HQR'.
    :param annual_data_path_dict: A dictionary of annual datasets with directories as keys and variable names as values.
    :param static_data_path_dict: A dictionary of static datasets with directories as keys and variable names as values.
    :param exclude_columns: List of columns to remove during model training.
    :param predictor_csv_and_nan_pos_dir: Directory path of csv and nan position pickles to create prediction rasters.
    :param basin_shape: Filepath of basin's shapefile.
    :param skip_processing: Set to True to skip data processing.

    :return: None.
    """
    if not skip_processing:
        print('---------------------------------------------------------')
        print(f'\nRunning Leave-One-Basin-Out test for {basin_code}...')
        print('---------------------------------------------------------')

        # --------------------------------------------------------------------------------------------------------------
        # Boolean flags
        # --------------------------------------------------------------------------------------------------------------
        skip_process_pumping_data = False           ######
        skip_train_test_df_creation = False         ######
        skip_train_test_split = False               ######
        load_model = False                          ######
        save_model = True                           ######
        skip_create_prediction_raster = False       ######

        # --------------------------------------------------------------------------------------------------------------
        # Pumping data adjustment
        # during training, the pumping data will consist no pixel from inside the respective basin
        # during testing, the holdout will only have pumping data from inside the basin
        # --------------------------------------------------------------------------------------------------------------
        if not skip_process_pumping_data:
            westUS_pumping_dir = f'../../Data_main/pumping/rasters/WestUS_pumping'  # this is the WestUS-scale pumping data
            westUS_pumping_rasters = glob(os.path.join(westUS_pumping_dir, '*.tif'))

            # pumping for fitting (training)
            fitting_output_dir = f'../../Data_main/rasters/ML_LOBO/{basin_code}/fitting/pumping'

            for ras in westUS_pumping_rasters:
                raster_name = os.path.basename(ras)

                set_nodata_inside_shapefile(input_raster=ras, input_shape=basin_shape,
                                            output_dir=fitting_output_dir, raster_name=raster_name,)

            # pumping for holdout
            holdout_dir = f'../../Data_main/rasters/ML_LOBO/{basin_code}/holdout/pumping'

            for ras in westUS_pumping_rasters:
                raster_name = os.path.basename(ras)
                mask_raster_by_shape(input_raster=ras, input_shape=basin_shape, crop=False,
                                     output_dir=holdout_dir, raster_name=raster_name)
        else:
            fitting_output_dir = f'../../Data_main/rasters/ML_LOBO/{basin_code}/fitting/pumping'
            holdout_dir = f'../../Data_main/rasters/ML_LOBO/{basin_code}/holdout/pumping'

        # --------------------------------------------------------------------------------------------------------------
        # Parquet (tabular data) creation
        # --------------------------------------------------------------------------------------------------------------

        # adding pumping_mm to the annual dataset
        annual_training_data_path_dict = {**annual_data_path_dict, 'pumping_mm': fitting_output_dir}  # final annual data paths

        # datasets to include in the dataframe (not all will go into the final model)
        datasets_to_include = list(static_data_path_dict.keys()) + list(annual_training_data_path_dict.keys())
                              # pumping_mm will be excluded in train_val_test_split_v2

        # training time periods
        years_list = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                      2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

        # --------------------------------------------------------------------------------------------------------------
        # Dataframe creation and train-test split
        # --------------------------------------------------------------------------------------------------------------

        # create dataframe
        train_test_parquet_path = f'../../Model_run/ML_model/LOBO/{model_version}/interim_csv/{basin_code}/train_test_{model_version}.parquet'

        create_train_test_dataframe(years_list=years_list,
                                    yearly_data_path_dict=annual_training_data_path_dict,
                                    static_data_path_dict=static_data_path_dict,
                                    datasets_to_include=datasets_to_include,
                                    output_parquet=train_test_parquet_path,
                                    n_partitions=5,
                                    skip_processing=skip_train_test_df_creation)

        # train-test split
        output_dir = f'../../Model_run/ML_model/LOBO/{model_version}/interim_csv/{basin_code}'

        x_train, x_test, y_train, y_test = \
            split_train_val_test_set_v2(data_parquet=train_test_parquet_path, pred_attr='pumping_mm',
                                        exclude_columns=exclude_columns, output_dir=output_dir,
                                        model_version=model_version, train_size=0.7, test_size=0.3,
                                        random_state=42, skip_processing=skip_train_test_split)

        # --------------------------------------------------------------------------------------------------------------
        # Model training and performance evaluation
        # --------------------------------------------------------------------------------------------------------------
        print('\n########## Model training')

        save_model_to_dir = f'../../Model_run/ML_model/LOBO/{model_version}/models'
        model_name = f'westus_pumping_{model_version}.joblib'
        makedirs([save_model_to_dir])

        lgbm_reg_trained = train_model(x_train=x_train, y_train=y_train, params_dict=model_param_dict,
                                       load_model=load_model, save_model=save_model, save_folder=save_model_to_dir,
                                       model_save_name=model_name, categorical_columns=None,
                                       skip_tune_hyperparameters=True)
        print(lgbm_reg_trained, '\n')
        print('########## Model performance')

        # --------------------------------------------------------------------------------------------------------------
        # LOBO performance evaluation
        # --------------------------------------------------------------------------------------------------------------
        print(f'\n########## Leave-One-Basin-Out performance')

        annual_LOBO_data_path_dict = {**annual_data_path_dict, 'pumping_mm': holdout_dir}

        LOBO_df_parquet = f'../../Model_run/ML_model/LOBO/{model_version}/interim_csv/{basin_code}/{basin_code}.parquet'

        create_train_test_dataframe(years_list=years_list,
                                    yearly_data_path_dict=annual_LOBO_data_path_dict,
                                    static_data_path_dict=static_data_path_dict,
                                    datasets_to_include=datasets_to_include,
                                    output_parquet=LOBO_df_parquet,
                                    n_partitions=5,
                                    skip_processing=skip_train_test_df_creation)

        df_basin = pd.read_parquet(LOBO_df_parquet)

        x_basin = df_basin.drop(columns=['pumping_mm'] + exclude_columns)
        x_basin = reindex_df(x_basin)
        y_basin = df_basin['pumping_mm']
        year_series = df_basin['year']

        basin_prediction_csv_path = f'../../Model_run/ML_model/LOBO/{model_version}/{basin_code}/results/{basin_code}_results.csv'
        predict_LOBO_basin(trained_model=lgbm_reg_trained, predictor_array=x_basin, target_array=y_basin,
                           year_series=year_series, prediction_csv_path=basin_prediction_csv_path,
                           categorical_columns=None)

        # --------------------------------------------------------------------------------------------------------------
        # Plotting
        # --------------------------------------------------------------------------------------------------------------

        # plotting LOBO basin scatters
        lobo_df = pd.read_csv(basin_prediction_csv_path)

        plot_dir = f'../../Model_run/ML_model/LOBO/{model_version}/scatter_plots/'
        scatter_plot_of_same_vars(Y_pred=lobo_df['predicted'], Y_obsv=lobo_df['actual'],
                                  x_label='Actual pumping (mm/year)', y_label='Predicted pumping (mm/year)',
                                  plot_name=f'{basin_code}_LOBO.jpg',
                                  savedir=plot_dir, alpha=0.2,
                                  color_format='o', marker_size=4, title=None,
                                  axis_lim=None, tick_interval=500)

        # # --------------------------------------------------------------------------------------------------------------
        # # Prediction raster creation
        # # --------------------------------------------------------------------------------------------------------------
        #
        # prediction_interim_output_dir = f'../../Data_main/rasters/ML_LOBO/{basin_code}/interim'
        # predict_annual_pumping_rasters(trained_model=lgbm_reg_trained, years_list=list(range(2000, 2020)),
        #                                exclude_columns=exclude_columns,
        #                                predictor_csv_and_nan_pos_dir=predictor_csv_and_nan_pos_dir,
        #                                prediction_name_keyword=basin_code, output_dir=prediction_interim_output_dir,
        #                                ref_raster=WestUS_raster, skip_processing=skip_create_prediction_raster)
        #
        # # the generated prediction rasters are Western US scale. Clipping them to the basin
        # interim_predictions = glob(os.path.join(prediction_interim_output_dir, '*.tif'))
        # prediction_output_dir = f'../../Data_main/rasters/ML_LOBO/{basin_code}'
        #
        # for pred in interim_predictions:
        #     raster_name = os.path.basename(pred)
        #     mask_raster_by_shape(input_raster=pred, input_shape=basin_shape, crop=True,
        #                          output_dir=prediction_output_dir, raster_name=raster_name)

    else:
        pass


if __name__ == '__main__':
    skip_LOBO_GMD3 = False                              ##### GMD3, KS
    skip_LOBO_GMD4 = False                              ##### GMD4, KS
    skip_LOBO_RPB = False                               ##### Republican Basin, CO
    skip_LOBO_SPB = False                               ##### South Platte River Basin, CO
    skip_LOBO_AR = False                                ##### Arkansas River Basin, CO
    skip_LOBO_SLV = False                               ##### San Luis Valley, CO
    skip_LOBO_HQR = False                               ##### Harquahala INA, AZ
    skip_LOBO_DOUG = False                              ##### Douglas AMA, AZ
    skip_LOBO_PHX = False                               ##### Phoenix AMA, AZ
    skip_LOBO_PNL = False                               ##### Pinal AMA, AZ
    skip_LOBO_SCRUZ = False                             ##### Santa Cruz AMA, AZ

    # exclude columns during training
    exclude_columns_in_training = ['stateID', 'pixelID', 'year',
                                   'shortRad', 'minRH']         ##################################

    # hyperparameters from tuned model
    lgbm_param_dict = {'boosting_type': 'dart',
                       'colsample_bynode': 0.8659002108624019,
                       'colsample_bytree': 0.6446870885955625,
                       'data_sample_strategy': 'bagging',
                       'learning_rate': 0.049900540446145183,
                       'max_depth': 7,
                       'min_child_samples': 40,
                       'n_estimators': 525,
                       'num_leaves': 50,
                       'path_smooth': 0.6885800578339573,
                       'subsample': 0.5617613316667519,
                       'force_col_wise': True
                       }

    # datasets
    data_path_dict = {
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
        'Canal_density': '../../Data_main/rasters/Canal_density',
        'Canal_distance': '../../Data_main/rasters/Canal_distance',
        'pixelID': '../../Data_main/ref_rasters/pixelID',
        'stateID': '../../Data_main/ref_rasters/stateID'
    }

    static_vars = {'FC', 'Canal_density', 'Canal_distance', 'stateID', 'pixelID'}  # static vars

    # splitting into annual and static datasets
    # Note that, the annual data path dict don't have pumping data, it will be added inside LOBO function
    yearly_data_path_dict = {i: j for i, j in data_path_dict.items() if i not in static_vars}
    stat_data_path_dict = {i: j for i, j in data_path_dict.items() if i in static_vars}  # static data paths

    prediction_df_output_dir = f'../../Data_main/rasters/ML_LOBO/dataframes_for_prediction'

    model_version = 'v6'

    # GMD3, KS
    perform_LOBO(model_version=f'{model_version}', basin_code='GMD3',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/GMD3.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_GMD3)

    # GMD4, KS
    perform_LOBO(model_version=f'{model_version}', basin_code='GMD4',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/GMD4.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_GMD4)

    # Republican River Basin, CO
    perform_LOBO(model_version=f'{model_version}', basin_code='RPB',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_RPB)

    # South Platte River Basin, CO
    perform_LOBO(model_version=f'{model_version}', basin_code='SPB',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/South_Platte_Basin.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_SPB)

    # Arkansas River Basin, CO
    perform_LOBO(model_version=f'{model_version}', basin_code='AR',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/Arkansas_Basin.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_AR)

    # San Luis Valley, CO
    perform_LOBO(model_version=f'{model_version}', basin_code='SLV',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/Rio_Grande_Basin.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_SLV)

    # Harquahala INA, AZ
    perform_LOBO(model_version=f'{model_version}', basin_code='HQR',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_HQR)

    # Douglas AMA, AZ
    perform_LOBO(model_version=f'{model_version}', basin_code='DOUG',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_DOUG)

    # Phoenix AMA, AZ
    perform_LOBO(model_version=f'{model_version}', basin_code='PHX',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/Phoenix_AMA.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_PHX)

    # Pinal AMA, AZ
    perform_LOBO(model_version=f'{model_version}', basin_code='PNL',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/Pinal_AMA.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_PNL)

    # Santa Cruz AMA, AZ
    perform_LOBO(model_version=f'{model_version}', basin_code='SCRUZ',
                 model_param_dict=lgbm_param_dict, exclude_columns=exclude_columns_in_training,
                 annual_data_path_dict=yearly_data_path_dict, static_data_path_dict=stat_data_path_dict,
                 basin_shape='../../Data_main/shapefiles/Basins_of_interest/SantaCruz_AMA.shp',
                 predictor_csv_and_nan_pos_dir=prediction_df_output_dir,
                 skip_processing=skip_LOBO_SCRUZ)