# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
import pandas as pd
from glob import glob
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.plots import scatter_plot_of_same_vars
from Codes.models.utils_mlp import DataLoaderCreator, main, plot_learning_curve, test, \
    calc_rangeWise_RMSE, load_model_and_predict_raster, plot_shap_summary_plot, plot_shap_interaction_plot

if __name__ == '__main__':
    # # # model version
    model_version = 'v4'                                ################################################################

    # # # model switches
    # setting 'RUN_MODEL' to False will skip all model running processing
    # setting 'skip_create_prediction_rasters' to False will load a trained model to create prediction raster
    RUN_MODEL = True                                  ################################################################
    tune_params = False                                ################################################################
    n_trials_for_tuning = 100                           ################################################################
    implement_earlyStopping = False                     #################################################################
    plot_hyperparam_importance = True                  #################################################################

    skip_SHAP_summary_plot = True                      ################################################################
    skip_SHAP_interaction_plot = True                  ################################################################

    skip_create_prediction_rasters = True              ################################################################

    # # # default variables (from hyperparameter tuning process)
    batch_size = 256
    n_features = 14
    n_epochs = 70
    lr = 0.001
    lr_scheduler = 'CosineAnnealingLR'
    activation = 'leakyrelu'
    patience = 10
    start_earlyStopping_at_epoch = 20
    # using optimizer 'AdamW'

    exclude_features_from_training = ['lon', 'lat', 'year', 'pixelID', 'stateID', 'Canal']

    # default model architecture
    default_params = {
        'fc_units': [128, 64, 32, 16],
        'weight_decay':  1e-2,
        'dropout': 0.1
    }

    # # # directories
    train_csv = f'../../Model_run/MLP_model/Model_csv/standardized/train.csv'
    val_csv = f'../../Model_run/MLP_model/Model_csv/standardized/val.csv'
    test_csv = f'../../Model_run/MLP_model/Model_csv/standardized/test.csv'

    model_save_path = f'../../Model_run/MLP_model/model_{model_version}.pth'
    model_info_save_path = f'../../Model_run/MLP_model/model_info_{model_version}.pth'
    hyperparam_importance_plot = f'../../Model_run/MLP_model//hyperparam_imp_{model_version}.jpg'
    learning_curve_plot = f'../../Model_run/MLP_model/learning_curve_{model_version}.jpg'

    if RUN_MODEL:

        # --------------------------------------------------------------------------------------------------------------
        # 1. Training model
        # --------------------------------------------------------------------------------------------------------------
        trained_model, model_info = main(train_data_csv=train_csv, val_data_csv=val_csv,
                                         features_to_exclude=exclude_features_from_training,
                                         batch_size=batch_size,
                                         n_features=n_features, n_epochs=n_epochs,
                                         model_save_path=model_save_path, model_info_save_path=model_info_save_path,
                                         lr=lr, lr_scheduler=lr_scheduler,
                                         activation_func=activation,
                                         default_params=default_params,
                                         implement_earlyStopping=implement_earlyStopping,
                                         patience=patience, start_EarlyStop_count_from_epoch=start_earlyStopping_at_epoch,
                                         tune_parameters=tune_params, n_trials=n_trials_for_tuning,
                                         plot_hyperparams_importance=plot_hyperparam_importance,
                                         hyperparam_importance_plot_path=hyperparam_importance_plot)

        # --------------------------------------------------------------------------------------------------------------
        # 2. Model performance evaluation (on test + validation data)
        # --------------------------------------------------------------------------------------------------------------
        print('\n############## Model performance on test data ###############\n')

        print('Test performance:')
        testLoader = DataLoaderCreator(data_csv=test_csv,
                                       shuffle=False, features_to_exclude=exclude_features_from_training,
                                       batch_size=batch_size, verbose=False).get_dataloader()

        test_results = f'../../Model_run/MLP_model/output_csv/test_results_{model_version}.csv'
        test(model=trained_model, test_loader=testLoader, output_csv=test_results)

        calc_rangeWise_RMSE(results_csv=test_results, value_ranges=[0, 100, 200, 400, 500, 600, 800, 1400],
                            output_txt=f'../../Model_run/MLP_model/output_csv/test_results_rangeWise_{model_version}.txt')

        print('Validation performance:')
        valLoader = DataLoaderCreator(data_csv=val_csv,
                                       shuffle=False, features_to_exclude=exclude_features_from_training,
                                       batch_size=batch_size, verbose=False).get_dataloader()

        val_results = f'../../Model_run/MLP_model/output_csv/val_results_{model_version}.csv'
        test(model=trained_model, test_loader=valLoader, output_csv=val_results)

        # --------------------------------------------------------------------------------------------------------------
        # 3. Plotting learning curve + test scatter plot + validation scatter plot
        # --------------------------------------------------------------------------------------------------------------
        plot_learning_curve(train_loss=model_info['train_losses'],
                            val_loss=model_info['val_losses'],
                            plot_save_path=learning_curve_plot)

        test_results_df = pd.read_csv(test_results)
        scatter_plot_of_same_vars(Y_pred=test_results_df['predicted'], Y_obsv=test_results_df['actual'],
                                  x_label='actual pumping (mm/year)', y_label='predicted pumping (mm/year)',
                                  plot_name=f'test_scatter_{model_version}.jpeg',
                                  savedir=f'../../Model_run/MLP_model', alpha=0.5,
                                  color_format='o', marker_size=5, title='performance on test set',
                                  tick_interval=200)

        val_results_df = pd.read_csv(val_results)
        scatter_plot_of_same_vars(Y_pred=val_results_df['predicted'], Y_obsv=val_results_df['actual'],
                                  x_label='actual pumping (mm/year)', y_label='predicted pumping (mm/year)',
                                  plot_name=f'val_scatter_{model_version}.jpeg',
                                  savedir=f'../../Model_run/MLP_model', alpha=0.5,
                                  color_format='o', marker_size=5, title='performance on validation set',
                                  tick_interval=200)

    # --------------------------------------------------------------------------------------------------------------
    # 4. Plotting shapely values
    # --------------------------------------------------------------------------------------------------------------

    # SHAP summary plot
    plot_shap_summary_plot(trained_model_path=model_save_path, trained_model_info=model_info_save_path,
                           use_samples=2000, data_csv=test_csv,
                           exclude_features=exclude_features_from_training,
                           save_plot_path=f'../../Model_run/MLP_model/SHAP/MLP_SHAP_summary_{model_version}.png',
                           skip_processing=skip_SHAP_summary_plot)

    # SHAP interaction plot
    features_to_plot = ['Consumptive groundwater use', 'Effective precipitation', 'Shortwave radiation',
                        'Irrigated crop fraction', 'ET', 'Reference ET', 'Field capacity',
                        'Precipitation', 'Surface water irrigation']
    plot_shap_interaction_plot(model_version=model_version,
                               features_to_plot=features_to_plot, trained_model_path=model_save_path,
                               trained_model_info=model_info_save_path, use_samples=2000, data_csv=test_csv,
                               save_plot_dir=f'../../Model_run/MLP_model/SHAP',
                               skip_processing=skip_SHAP_interaction_plot)

    # ------------------------------------------------------------------------------------------------------------------
    # 5. Annual GW pumping prediction
    # ------------------------------------------------------------------------------------------------------------------
    annual_standardized_df_dir = f'../../Model_run/MLP_model/Model_csv/annual_csv/standardized'
    nan_pos_dir = f'../../Model_run/MLP_model/Model_csv/annual_csv'
    predicted_raster_dir = f'../../Data_main/rasters/pumping_prediction/{model_version}'
    WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'

    load_model_and_predict_raster(trained_model_path=model_save_path, trained_model_info=model_info_save_path,
                                  years_list=list(range(2000, 2020)),
                                  predictor_csv_dir=annual_standardized_df_dir, nan_pos_dir=nan_pos_dir,
                                  batch_size=batch_size, exclude_features_in_model=exclude_features_from_training,
                                  output_dir=predicted_raster_dir, ref_raster=WestUS_raster,
                                  irrig_fraction_dir=f'../../Data_main/rasters/Irrigated_cropland/Irrigated_Frac',
                                  verbose=False, skip_processing=skip_create_prediction_rasters)