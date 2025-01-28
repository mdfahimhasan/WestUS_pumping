# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.model.cnn import     main, plot_learning_curve, test, unstandardize_save_and_test

if __name__ == '__main__':
    # model version
    model_version = 'v1'                                                 #####

    # directories
    tile_dir_train = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/train'
    tile_dir_val = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/val'
    tile_dir_test = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/test'

    target_csv_train = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/train/y_train.csv'
    target_csv_val = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/val/y_val.csv'
    target_csv_test = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/test/y_test.csv'

    mean_csv = '../../Data_main/rasters/multibands/scaling_stats/mean.csv'
    std_csv = '../../Data_main/rasters/multibands/scaling_stats/std.csv'

    model_save_path = f'../../Model_run/DL_model/model_{model_version}.pth'
    model_info_save_path = f'../../Model_run/DL_model/model_info_{model_version}.pth'
    hyperparam_importance_plot = f'../../Model_run/DL_model/hyperparam_imp_{model_version}.jpg'
    leaning_curve_plot = f'../../Model_run/DL_model/learning_curve_{model_version}.jpg'

    # Default variables
    n_features = 21                                                     ##### number of input channel in a tile
    n_epochs = 100                                                      #####
    input_size = 7                                                      ##### height/width dim of a tile
    padding = 'same'                                                    #####
    activation = 'relu'                                                 #####
    pooling = 'avgpool'                                                 #####
    batch_size = 128                                                    #####

    # Default model architecture
    default_params = {
        'filters': [32, 64],                          ##### convolutional layers
        'kernel_size': [3, 3],                        ##### kernel size for each Conv layer
        'fc_units': [128, 64],                        ##### fully connected layer
        'batch_size': 64,                             ##### batch size for DataLoader
        'lr': 0.001,                                  ##### learning rate
        'weight_decay': 1e-3,                         ##### weight decay
        'dropout': 0.2                                ##### dropout rate
    }

    # Model switches
    tune_params = True                   #################################################################
    n_trials_for_tuning = 100             #################################################################
    implement_earlyStopping = False       #################################################################
    plot_hyperparam_importance = True    #################################################################
    skip_unstandardizing_testing = False  #################################################################

    # Running the model
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


    # Plotting learning curve
    plot_learning_curve(train_loss=model_info['train_losses'],
                        val_loss=model_info['val_losses'],
                        plot_save_path=leaning_curve_plot)

    # Model performances on standardized data
    print('########## Model performance on standardized data ##########\n')

    print('Train performance:')
    test(trained_model,
         tile_dir=tile_dir_train, target_csv=target_csv_train,
         batch_size=batch_size,
         data_type='train')

    print('Test performance:')
    test(trained_model,
         tile_dir=tile_dir_test, target_csv=target_csv_test,
         batch_size=batch_size,
         data_type='test')

    print('########## *************************************** ##########\n')

    # Model performances on unstandardized (actual) data
    print('########## Model performance on unstandardized (actual) data ##########\n')

    print('Train performance:')
    unstandardize_save_and_test(trained_model,
                                tile_dir=tile_dir_train,
                                target_csv=target_csv_train,
                                batch_size=batch_size,
                                data_type='train',
                                mean_csv=mean_csv,
                                std_csv=std_csv,
                                output_csv=f'../../Model_run/DL_model/output_csv/trainSet_results.csv',
                                skip_processing=skip_unstandardizing_testing)

    print('Validation performance:')
    unstandardize_save_and_test(trained_model,
                                tile_dir=tile_dir_val,
                                target_csv=target_csv_val,
                                batch_size=batch_size,
                                data_type='validation',
                                mean_csv=mean_csv,
                                std_csv=std_csv,
                                output_csv=f'../../Model_run/DL_model/output_csv/valSet_results.csv',
                                skip_processing=skip_unstandardizing_testing)

    print('Test performance:')
    unstandardize_save_and_test(trained_model,
                                tile_dir=tile_dir_test,
                                target_csv=target_csv_test,
                                batch_size=batch_size,
                                data_type='test',
                                mean_csv=mean_csv,
                                std_csv=std_csv,
                                output_csv=f'../../Model_run/DL_model/output_csv/testSet_results.csv',
                                skip_processing=skip_unstandardizing_testing)


    print('########## ************************************************* ##########\n')


    # # to deal with overfitting
    # remove outliers in the training data (very high or very low pumping values)
    # try param tuning with dropout and weight decay
    # consider using learning rate scheduler