# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
import pandas as pd
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.plots import scatter_plot_of_same_vars
from Codes.models.utils_mlp import DataLoaderCreator, main, test


def perform_LOBO(basin_code, model_version, exclude_features_from_training, skip_processing=False):
    """
    This is the driver file for Leave-One-Basin-Out performance evaluation.

    Leave-One-Basin-Out (LOBO):
    LOBO is a cross-validation approach specifically designed for evaluating generalization capability of this model.
    It is a variant of Leave-One-Out Cross-Validation (LOO-CV), but instead of leaving out individual samples, it
    leaves out an entire hydrological basin during model training and then evaluates the model's performance on
    the excluded basin.

    :param basin_code: Basin code, such as 'GMD3', 'RPB', or 'HQR'.
    :param model_version: str of model version. Should match with the main model's version.
    :param skip_processing: Set To True to skip this step.

    :return: None.
    """
    if not skip_processing:
        print('----------------------------------------------------')
        print(f'Performing Leave-One-Basin-Out for {basin_code}')
        print('----------------------------------------------------\n')

        print("**Ensure that the model is being trained with the main model's optimized parameters**")

        # default variables (from hyperparameter tuning process)
        batch_size = 256
        n_features = 13
        n_epochs = 70
        activation = 'leakyrelu'
        lr = 0.001
        lr_scheduler = 'CosineAnnealingLR'

        # default model architecture
        default_params = {
            'fc_units': [128, 64, 32, 16],
            'weight_decay': 1e-2,
            'dropout': 0.1
        }

        # directories
        train_csv = f'../../Model_run/MLP_model/LOBO/{model_version}/{basin_code}/standardized/train.csv'
        val_csv = f'../../Model_run/MLP_model/LOBO/{model_version}/{basin_code}/standardized/val.csv'
        holdout_csv = f'../../Model_run/MLP_model/LOBO/{model_version}/{basin_code}/standardized/holdout.csv'

        model_save_path = f'../../Model_run/DL_model/LOBO/{model_version}/{basin_code}/MLP_model.pth'
        model_info_save_path = f'../../Model_run/DL_model/LOBO/{model_version}/{basin_code}/MLP_info.pth'

        # training model
        trained_model, model_info = main(train_data_csv=train_csv, val_data_csv=val_csv,
                                         features_to_exclude=exclude_features_from_training,
                                         batch_size=batch_size,
                                         n_features=n_features, n_epochs=n_epochs,
                                         model_save_path=model_save_path, model_info_save_path=model_info_save_path,
                                         lr=lr, lr_scheduler=lr_scheduler,
                                         activation_func=activation,
                                         default_params=default_params)

        print('Holdout performance:')
        holdout_Loader = DataLoaderCreator(data_csv=holdout_csv,
                                           shuffle=False, features_to_exclude=exclude_features_from_training,
                                           batch_size=batch_size, verbose=False).get_dataloader()

        holdout_results = f'../../Model_run/MLP_model/LOBO/{model_version}/{basin_code}/results/{basin_code}_results.csv'

        test(model=trained_model, test_loader=holdout_Loader, output_csv=holdout_results)

        holdout_results_df = pd.read_csv(holdout_results)
        scatter_plot_of_same_vars(Y_pred=holdout_results_df['predicted'], Y_obsv=holdout_results_df['actual'],
                                  x_label='actual pumping (mm/year)', y_label='predicted pumping (mm/year)',
                                  plot_name=f'holdout_scatter_{basin_code}.jpeg',
                                  savedir=f'../../Model_run/MLP_model/LOBO/{model_version}/{basin_code}/results',
                                  alpha=0.5, color_format='o', marker_size=5,
                                  title=f'performance on holdout set: {basin_code}',
                                  tick_interval=100)
        print('----------------------------------------------------------------------------\n')
    else:
        pass


# exclude columns during model training
exclude_features_from_training = ['lon', 'lat', 'year', 'pixelID', 'stateID']

if __name__ == '__main__':
    # # flags
    model_version = 'v4'
    skip_LOBO_GMD3 = False              ##### GMD3, KS
    skip_LOBO_GMD4 = False              ##### GMD4, KS
    skip_LOBO_RPB = False               ##### Republican Basin, CO
    skip_LOBO_HQR = False               ##### Harquahala INA, AZ
    skip_LOBO_DOUG = False              ##### Douglas AMA, AZ
    skip_LOBO_PHX = False               ##### Phoenix AMA, AZ
    skip_LOBO_PNL = False               ##### Pinal AMA, AZ
    skip_LOBO_SCRUZ = False             ##### Santa Cruz AMA, AZ

    # # GMD3, KS
    perform_LOBO(basin_code='GMD3', model_version=model_version,
                 exclude_features_from_training=exclude_features_from_training,
                 skip_processing=skip_LOBO_GMD3)

    # # GMD4, KS
    perform_LOBO(basin_code='GMD4', model_version=model_version,
                 exclude_features_from_training=exclude_features_from_training,
                 skip_processing=skip_LOBO_GMD4)

    # # RPB, CO
    perform_LOBO(basin_code='RPB', model_version=model_version,
                 exclude_features_from_training=exclude_features_from_training,
                 skip_processing=skip_LOBO_RPB)


    # # Douglas AMA, AZ
    perform_LOBO(basin_code='DOUG', model_version=model_version,
                 exclude_features_from_training=exclude_features_from_training,
                 skip_processing=skip_LOBO_DOUG)

    # # Harquahala, AZ
    perform_LOBO(basin_code='HQR', model_version=model_version,
                 exclude_features_from_training=exclude_features_from_training,
                 skip_processing=skip_LOBO_HQR)

    # # Phoenix, AZ
    perform_LOBO(basin_code='PHX', model_version=model_version,
                 exclude_features_from_training=exclude_features_from_training,
                 skip_processing=skip_LOBO_PHX)

    # # Pinal, AZ
    perform_LOBO(basin_code='PNL', model_version=model_version,
                 exclude_features_from_training=exclude_features_from_training,
                 skip_processing=skip_LOBO_PNL)

    # # Santa Cruz, AZ
    perform_LOBO(basin_code='SCRUZ', model_version=model_version,
                 exclude_features_from_training=exclude_features_from_training,
                 skip_processing=skip_LOBO_SCRUZ)
