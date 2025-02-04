# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.model.cnn import main, plot_learning_curve, test, unstandardize_save_and_test

if __name__ == '__main__':
    # model version
    model_version = 'v2'                                                 #####

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
    batch_size = 128                                                    ##### batch size of DataLoader
    n_features = 21                                                     ##### number of input channel in a tile
    n_epochs = 90                                                       #####
    input_size = 7                                                      ##### height/width dim of a tile
    padding = 'same'                                                    #####
    activation = 'relu'                                                 #####
    pooling = 'avgpool'                                                 #####
    patience = 10                                                       ##### early stopping counter patient set to 10 epoch
    start_earlyStopping_at_epoch = 40                                   ##### early stopping will initialize after 40 epochs


    # Default model architecture
    default_params = {
        'filters': [48, 64],                                          ##### convolutional layers
        'kernel_size': [5, 5],                                        ##### kernel size for each Conv layer
        'fc_units': [32],                                             ##### fully connected layer
        'lr': 0.00034424691202725654,                                 ##### learning rate
        'weight_decay': 0.0002954114285538287,                        ##### weight decay
        'dropout': 0.5                                                ##### dropout rate
    }

    # Model switches
    tune_params = False                   #################################################################
    n_trials_for_tuning = 200             #################################################################
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
    print('\n############## Model performance on standardized data ###############\n')

    print('Test performance:')
    test(trained_model,
         tile_dir=tile_dir_test, target_csv=target_csv_test,
         batch_size=batch_size,
         data_type='test')


    # Model performances on unstandardized (actual) data
    print('########## Model performance on unstandardized (actual) data ##########\n')

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

    ####################################################################################################################
    ####################################################################################################################
    # # Model performances on Kansas (without training on kansas)
    # print('\n############## Model performance on Kansas ###############\n')
    #
    # print('Model performance on kansas (standardized):')
    #
    # tile_dir_KS = '../../Data_main/rasters/multibands/KS/train_val_test_splits/standardized/train'
    # target_csv_KS = '../../Data_main/rasters/multibands/KS/train_val_test_splits/standardized/train/y_train.csv'
    #
    # test(trained_model,
    #      tile_dir=tile_dir_KS, target_csv=target_csv_KS,
    #      batch_size=batch_size,
    #      data_type='test')
    #
    # print('Model performance on kansas (unstandardized):')
    # unstandardize_save_and_test(trained_model,
    #                             tile_dir=tile_dir_KS,
    #                             target_csv=target_csv_KS,
    #                             batch_size=batch_size,
    #                             data_type='test',
    #                             mean_csv=mean_csv,
    #                             std_csv=std_csv,
    #                             output_csv=f'../../Model_run/DL_model/KS/KS_results.csv',
    #                             skip_processing=skip_unstandardizing_testing)


    # Recommendations for Improvement:

    # Address Overfitting:
    # - Consider increasing weight decay
    # - Try adding more dropout layers
    # - Experiment with data augmentation if applicable

    # Architecture Adjustments:
    # - The model might benefit from batch normalization layers
    # - Consider adding skip connections
    # - Experiment with different kernel sizes

    # Training Strategy:
    # - Implement learning rate scheduling
    # - Try different optimizers
    # - Consider early stopping based on validation loss
    # - consider training without dropout entirely
