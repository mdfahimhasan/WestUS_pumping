# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.model.cnn import (
    DataLoaderCreator,
    CNNRegression,
    train_validate,
    plot_learning_curve,
    test
)

if __name__ == '__main__':
    # model version
    model_version = 'v1'            ####################################################################################

    # directories
    tile_dir_train = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/train'
    tile_dir_val = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/val'
    tile_dir_test = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/test'

    target_csv_train = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/train/y_train.csv'
    target_csv_val = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/val/y_val.csv'
    target_csv_test = '../../Data_main/rasters/multibands/train_val_test_splits/standardized/test/y_test.csv'

    model_save_path = f'../../Model_run/model/best_model_{model_version}.pth'
    loss_save_path = f'../../Model_run/model/losses_{model_version}.pkl'
    leaning_curve_plot = f'../../Model_run/model/learning_curve_{model_version}.jpg'

    # variables
    batch_size = 32                 ####################################################################################
    n_epochs = 20                   ####################################################################################
    learning_rate = 0.001           ####################################################################################
    padding = 'same'                ####################################################################################
    activation = 'relu'             ####################################################################################
    pooling = 'avgpool'             ####################################################################################
    patience = 10                    ###################################################################################

    # model architecture
    n_features = 15             # number of input channels                         #####################################
    input_size = 7              # tiles size                                       #####################################
    filters = [16, 32]          # number of filters for each convolutional layer   #####################################
    kernels = [3, 3]            # kernel sizes for convolutional layers            #####################################
    fc_layers = [128, 64]       # fully connected layers                           #####################################


    # dataLoader
    train_loader_creator = DataLoaderCreator(tile_dir_train, target_csv_train,
                                             batch_size=batch_size, data_type='train')
    train_loader = train_loader_creator.get_dataloader()

    val_loader_creator = DataLoaderCreator(tile_dir_val, target_csv_val,
                                           batch_size=batch_size, data_type='validation')
    val_loader = val_loader_creator.get_dataloader()

    test_loader_creator = DataLoaderCreator(tile_dir_test, target_csv_test,
                                            batch_size=batch_size, data_type='test')
    test_loader = test_loader_creator.get_dataloader()

    # mode initialization
    model = CNNRegression(
        n_features=n_features,
        input_size=input_size,
        filters=filters,
        kernels=kernels,
        fc_layers=fc_layers,
        padding=padding,
        pooling=pooling
    )

    # configuring optimizer
    optimizer = model.configure_optimizer(optimizer_name='adam', lr=learning_rate,
                                          momentum=0.9, weight_decay=1e-4)

    # training and validating model
    print("Starting training and validation...")
    train_losses, val_losses = train_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        optimizer=optimizer,
        model_save_path=model_save_path,
        loss_save_path=loss_save_path,
        patience=patience,
        verbose=True
    )

    # learning curve plot
    plot_learning_curve(loss_save_path=loss_save_path,
                        plot_save_path=leaning_curve_plot)

    # test the Model
    print('Testing the model on the test set...')

    test_loss, test_rmse, test_mae, test_r2 = test(model, test_loader)








