# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.model.utils_cnn import main, plot_learning_curve, test, unstandardize_save_and_test, plot_shap_values

if __name__ == '__main__':
    # model version
    model_version = 'v7'                                                #####

    # model switches
    tune_params = False                                 ################################################################
    n_trials_for_tuning = 200                           ################################################################
    implement_earlyStopping = True                     #################################################################
    plot_hyperparam_importance = True                  #################################################################
    skip_unstandardizing_training = True               #################################################################
    skip_unstandardizing_testing = False                #################################################################
    skip_plot_SHAP_plot = False                         #################################################################

    # default variables (from hyperparameter tuning process)
    batch_size = 128                                                    ##### batch size of DataLoader
    n_features = 21                                                     ##### number of input channel in a tile
    n_epochs = 90                                                       #####
    input_size = 7                                                      ##### height/width dim of a tile
    padding = 'same'                                                    #####
    activation = 'relu'                                                 #####
    pooling = 'avgpool'                                                 #####
    patience = 10                                                       ##### early stopping counter patient set to 10 epoch
    start_earlyStopping_at_epoch = 40                                   ##### early stopping will initialize after 40 epochs

    # default model architecture
    default_params = {
        'filters': [64, 64],                                            ##### convolutional layers
        'kernel_size': [5, 5],                                          ##### kernel size for each Conv layer
        'fc_units': [256, 32],                                          ##### fully connected layer
        'lr': 0.0003011646825680101,                                    ##### learning rate
        'weight_decay': 6.86933341560855e-05,                           ##### weight decay
        'dropout': 0.5                                                  ##### dropout rate
    }

    # directories
    tile_dir_train = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/train'
    tile_dir_val = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/val'
    tile_dir_test = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/test'

    target_csv_train = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/train/y_train.csv'
    target_csv_val = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/val/y_val.csv'
    target_csv_test = '../../Data_main/rasters/multibands_westUS/train_val_test_splits/standardized/test/y_test.csv'

    mean_csv = '../../Data_main/rasters/multibands_westUS/scaling_stats/mean.csv'
    std_csv = '../../Data_main/rasters/multibands_westUS/scaling_stats/std.csv'

    model_save_path = f'../../Model_run/DL_model/model_{model_version}.pth'
    model_info_save_path = f'../../Model_run/DL_model/model_info_{model_version}.pth'
    hyperparam_importance_plot = f'../../Model_run/DL_model/hyperparam_imp_{model_version}.jpg'
    leaning_curve_plot = f'../../Model_run/DL_model/learning_curve_{model_version}.jpg'

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Training model
    # ------------------------------------------------------------------------------------------------------------------
    trained_model, model_info = main(tile_dir_train=tile_dir_train, target_csv_train=target_csv_train,
                                     tile_dir_val=tile_dir_val, target_csv_val=target_csv_val,
                                     batch_size=batch_size,
                                     n_features=n_features, input_size=input_size, n_epochs=n_epochs,
                                     padding=padding, pooling=pooling,
                                     activation_func=activation,
                                     model_save_path=model_save_path,
                                     model_info_save_path=model_info_save_path,
                                     implement_earlyStopping=implement_earlyStopping,
                                     tune_parameters=tune_params, n_trials=n_trials_for_tuning,
                                     default_params=default_params,
                                     plot_hyperparams_importance=plot_hyperparam_importance,
                                     hyperparam_importance_plot_path=hyperparam_importance_plot)


    # ------------------------------------------------------------------------------------------------------------------
    # 2. Plotting learning curve
    # ------------------------------------------------------------------------------------------------------------------
    plot_learning_curve(train_loss=model_info['train_losses'],
                        val_loss=model_info['val_losses'],
                        plot_save_path=leaning_curve_plot)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. Model performance evaluation
    # ------------------------------------------------------------------------------------------------------------------
    # model performances on standardized data
    print('\n############## Model performance on standardized data ###############\n')

    print('Test performance:')
    test(trained_model,
         tile_dir=tile_dir_test, target_csv=target_csv_test,
         batch_size=batch_size,
         data_type='test')


    # model performances on unstandardized (actual) data
    print('########## Model performance on unstandardized (actual) data ##########\n')

    # for train set, just using model and storing the unstandardized results
    # the performance metrices estimated using this function isn't representative of the true training performance
    unstandardize_save_and_test(trained_model,
                                tile_dir=tile_dir_train,
                                target_csv=target_csv_train,
                                batch_size=batch_size,
                                data_type='test',
                                mean_csv=mean_csv,
                                std_csv=std_csv,
                                output_csv=f'../../Model_run/DL_model/output_csv/trainSet_results.csv',
                                skip_processing=skip_unstandardizing_training)


    print('Test performance:')
    test_rmse, test_mae, test_r2, test_nrmse = \
        unstandardize_save_and_test(trained_model,
                                    tile_dir=tile_dir_test,
                                    target_csv=target_csv_test,
                                    batch_size=batch_size,
                                    data_type='test',
                                    mean_csv=mean_csv,
                                    std_csv=std_csv,
                                    output_csv=f'../../Model_run/DL_model/output_csv/testSet_results.csv',
                                    skip_processing=skip_unstandardizing_testing)

    print(f'Results -> RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, NRMSE: {test_nrmse:.4f}, R²: {test_r2:.4f}\n')

    # ------------------------------------------------------------------------------------------------------------------
    # 4. Explainable AI plots (using SHAP)
    # ------------------------------------------------------------------------------------------------------------------
    # current input variables' name in the model are as follows, replaces by representative names
    # ['netGWIrr', 'peff', 'ret', 'precip', 'tmax', 'ET', 'irr_crop_frac',
    # 'irr_cropland', 'maxRH', 'shortRad', 'vpd', 'sunHr', 'sw_huc12', 'gw_perc_huc12',
    # 'spi', 'spei', 'eddi', 'arid', 'cold', 'temp_Dry', 'temp_noDry']
    feature_names = ['consumptive groundwater use', 'effective precipitation', 'reference ET', 'precipitation',
                     'maximum temperature', 'ET', 'fraction of irrigated cropland', 'irrigated cropland',
                     'maximum relative humidity', 'downward shortwave radiation', 'vapor pressure deficit',
                     'daylight duration', 'HUC12 surface water irrigation', 'HUC12 groundwater use %',
                     'SPI', 'SPEI', 'EDDI', 'Arid', 'Cold', 'Temperate dry summer', 'Temperate no dry summer']

    plot_shap_values(trained_model, tile_dir=tile_dir_train,
                     target_csv=target_csv_train, batch_size=batch_size,
                     plot_save_path=f'../../Model_run/DL_model/SHAP_summary_{model_version}.jpg',
                     feature_names=feature_names, data_type='test',
                     skip_processing=skip_plot_SHAP_plot)