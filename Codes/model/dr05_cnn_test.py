# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import torch
from Codes.model.utils_cnn import unstandardize_save_and_test


if __name__ == '__main__':
    model_version = 'v6'                                            #####

    # model switches
    skip_perState_performance_evaluation = True  #######################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Loading trained model and setting it to evaluation mode
    # ------------------------------------------------------------------------------------------------------------------
    trained_model_path = f'../../Model_run/DL_model/model_{model_version}.pth'
    trained_model = torch.load(trained_model_path)
    trained_model.eval()

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Testing model's performance on left out basins
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # 3. Testing model's overall performance on individual state
    # # note - the tiled dataset include tiles that were used in model training
    # ------------------------------------------------------------------------------------------------------------------
    if not skip_perState_performance_evaluation:

        # # # # # # # # # # # # # # # # # # # Model performances on Kansas # # # # # # # # # # # # # # # # # # # # # # #
        print('\n********************* Model performance on Kansas *********************\n')

        print('Standardized performance:')

        tile_dir_KS = '../../Data_main/rasters/multibands_perState/KS/train_val_test_splits/standardized/train'
        target_csv_KS = '../../Data_main/rasters/multibands_perState/KS/train_val_test_splits/standardized/train/y_train.csv'

        test(trained_model,
             tile_dir=tile_dir_KS, target_csv=target_csv_KS,
             batch_size=batch_size,
             data_type='test')

        print('UnStandardized performance:')
        ks_rmse, ks_mae, ks_r2, ks_nrmse = \
            unstandardize_save_and_test(trained_model, tile_dir=tile_dir_KS, target_csv=target_csv_KS,
                                        batch_size=batch_size, data_type='test', mean_csv=mean_csv, std_csv=std_csv,
                                        output_csv=f'../../Model_run/DL_model/output_csv/perState/KS_results.csv',
                                        skip_processing=False)

        print(f'Results -> RMSE: {ks_rmse:.4f}, MAE: {ks_mae:.4f}, NRMSE: {ks_nrmse:.4f}, R²: {ks_r2:.4f}\n')

        # # # # # # # # # # # # # # # # # # Model performances on Colorado # # # # # # # # # # # # # # # # # # # # # # #
        print('\n********************* Model performance on Colorado *********************\n')

        print('Standardized performance:')

        tile_dir_CO = '../../Data_main/rasters/multibands_perState/CO/train_val_test_splits/standardized/train'
        target_csv_CO = '../../Data_main/rasters/multibands_perState/CO/train_val_test_splits/standardized/train/y_train.csv'

        test(trained_model, tile_dir=tile_dir_CO, target_csv=target_csv_CO,
             batch_size=batch_size, data_type='test')

        print('UnStandardized performance:')
        co_rmse, co_mae, co_r2, co_nrmse = \
            unstandardize_save_and_test(trained_model, tile_dir=tile_dir_CO, target_csv=target_csv_CO,
                                        batch_size=batch_size, data_type='test', mean_csv=mean_csv, std_csv=std_csv,
                                        output_csv=f'../../Model_run/DL_model/output_csv/perState/CO_results.csv',
                                        skip_processing=False)

        print(f'Results -> RMSE: {co_rmse:.4f}, MAE: {co_mae:.4f}, NRMSE: {co_nrmse:.4f}, R²: {co_r2:.4f}\n')

        # # # # # # # # # # # # # # # # # # Model performances on Arizona # # # # # # # # # # # # # # # # # # # # # # #
        print('\n********************* Model performance on Arizona *********************\n')

        print('Standardized performance:')

        tile_dir_AZ = '../../Data_main/rasters/multibands_perState/AZ/train_val_test_splits/standardized/train'
        target_csv_AZ = '../../Data_main/rasters/multibands_perState/AZ/train_val_test_splits/standardized/train/y_train.csv'

        test(trained_model, tile_dir=tile_dir_AZ, target_csv=target_csv_AZ,
             batch_size=batch_size, data_type='test')

        print('UnStandardized performance:')
        az_rmse, az_mae, az_r2, az_nrmse = \
            unstandardize_save_and_test(trained_model, tile_dir=tile_dir_AZ, target_csv=target_csv_AZ,
                                        batch_size=batch_size, data_type='test', mean_csv=mean_csv, std_csv=std_csv,
                                        output_csv=f'../../Model_run/DL_model/output_csv/perState/AZ_results.csv',
                                        skip_processing=False)

        print(f'Results -> RMSE: {az_rmse:.4f}, MAE: {az_mae:.4f}, NRMSE: {az_nrmse:.4f}, R²: {az_r2:.4f}\n')