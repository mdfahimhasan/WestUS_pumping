# author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
import joblib
import timeit
import numpy as np
import pandas as pd
from glob import glob
import dask.dataframe as ddf
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.inspection import PartialDependenceDisplay as PDisp
from sklearn.model_selection import train_test_split, cross_val_score

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object
from Codes.utils.stats_ops import calculate_rmse, calculate_r2, calculate_mae


no_data_value = -9999
model_res = 0.02000000000000000389  # in deg, 2 km
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'


def reindex_df(df):
    """
    Reindex dataframe based on column names.

    :param df: Predictor dataframe.

    :return: Reindexed dataframe.
    """
    sorted_columns = sorted(df.columns)
    df = df.reindex(sorted_columns, axis=1)

    return df


def apply_OneHotEncoding(input_df):
    one_hot = OneHotEncoder()
    input_df_enc = one_hot.fit_transform(input_df)
    return input_df_enc


def create_train_test_dataframe(years_list, yearly_data_path_dict,
                                static_data_path_dict, datasets_to_include, output_parquet,
                                n_partitions=20, skip_processing=False):
    """
    Compile yearly/static datasets into a dataframe. This function-generated dataframe will be used as
    train-test data for ML model at annual scale.

    *** if there is no static data, set static_data_path_dict to None.

    :param years_list: A list of years_list for which data to include in the dataframe.
    :param yearly_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
                                  Can't be None.
    :param static_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
                                  Set to None if there is static dataset.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param output_parquet: Output filepath of the parquet file to save. Using parquet as it requires lesser memory.
                            Can also save smaller dataframe as csv file if name has '.csv' extension.
    :param n_partitions: Number of partitions to save the parquet file in using dask dataframe.
    :param skip_processing: Set to True to skip this dataframe creation process.

    :return: The filepath of the output parquet file.
    """
    if not skip_processing:
        print('\ncreating train-test dataframe for annual model...')

        output_dir = os.path.dirname(output_parquet)
        makedirs([output_dir])

        variable_dict = {}

        # annual data compilation
        for var in yearly_data_path_dict.keys():
            if var in datasets_to_include:
                print(f'processing data for {var}..')

                for year_count, year in enumerate(years_list):
                    yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]

                    data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                    if (year_count == 0) & (var not in variable_dict.keys()):
                        variable_dict[var] = list(data_arr)

                    else:
                        variable_dict[var].extend(list(data_arr))

        # static data compilation
        if static_data_path_dict is not None:
            for var in static_data_path_dict.keys():
                if var in datasets_to_include:
                    print(f'processing data for {var}..')

                    static_data = glob(os.path.join(static_data_path_dict[var], '*.tif'))[0]
                    data_arr = read_raster_arr_object(static_data, get_file=False).flatten()

                    data_duplicated_for_total_years = list(data_arr) * len(years_list)
                    variable_dict[var] = data_duplicated_for_total_years

        train_test_ddf = ddf.from_dict(variable_dict, npartitions=n_partitions)
        train_test_ddf = train_test_ddf.dropna()

        if '.parquet' in output_parquet:
            train_test_ddf.to_parquet(output_parquet, write_index=False)

        elif '.csv' in output_parquet:
            train_test_df = train_test_ddf.compute()
            train_test_df.to_csv(output_parquet, index=False)

        return output_parquet

    else:
        return output_parquet


def split_train_val_test_set(input_csv, pred_attr, exclude_columns, output_dir,
                             model_version, test_perc=0.3, validation_perc=0,
                             random_state=0, verbose=True, n_bins=6,
                             skip_processing=False):
    """
    Split dataset into train, validation, and test data based on a train/test/validation ratio.

    :param input_csv : Input csv file (with filepath) containing all the predictors.
    :param pred_attr : Variable name which will be predicted. Defaults to 'Subsidence'.
    :param exclude_columns : Tuple of columns that will not be included in training the fitted_model. Set to None if
                             no columns is required to drop.
    :param output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    :param model_version: Model version name. Can be 'v1' or 'v2'.
    :param test_perc : The percentage of test dataset. Defaults to 0.3.
    :param validation_perc : The percentage of validation dataset. Defaults to 0.
    :param random_state : Seed value. Defaults to 0.
    :param verbose : Set to True if want to print which columns are being dropped and which will be included
                     in the model.
    :param n_bins: Number of quartile-based bins to use it target variable stratifying. Default set to 6.
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    returns: X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """
    global x_val, y_val, x

    if not skip_processing:
        print('\nSplitting train-test dataframe into train and test dataset...')

        input_df = pd.read_parquet(input_csv)

        # dropping columns that has been specified to not include

        if exclude_columns is not None:
            drop_columns = exclude_columns + [pred_attr]
            x = input_df.drop(columns=drop_columns)

        else:
            drop_columns = [pred_attr]
            x = input_df.drop(columns=drop_columns)

        y = input_df[pred_attr]  # response attribute

        # Reindexing for ensuring that columns go into the model in same serial every time
        x = reindex_df(x)

        if verbose:
            print('Dropping Columns-', exclude_columns, '\n')
            print('Predictors:', x.columns)

        # binning the target variable based on quartile-based bins
        # useful to stratify the pumping data to ensure proper train-test split
        y_binned = pd.qcut(y, q=n_bins, labels=False)

        # train-test splitting
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_perc, random_state=random_state,
                                                            shuffle=True, stratify=y_binned)
        if validation_perc > 0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_perc,
                                                              random_state=random_state, shuffle=True)

        # creating dataframe and saving train/test/validation datasets as csv
        makedirs([output_dir])

        x_train_df = pd.DataFrame(x_train)
        x_train_df.to_csv(os.path.join(output_dir, f'x_train_{model_version}.csv'), index=False)

        y_train_df = pd.DataFrame(y_train)
        y_train_df.to_csv(os.path.join(output_dir, f'y_train_{model_version}.csv'), index=False)

        x_test_df = pd.DataFrame(x_test)
        x_test_df.to_csv(os.path.join(output_dir, f'x_test_{model_version}.csv'), index=False)

        y_test_df = pd.DataFrame(y_test)
        y_test_df.to_csv(os.path.join(output_dir, f'y_test_{model_version}.csv'), index=False)

        if validation_perc > 0:
            x_val_df = pd.DataFrame(x_val)
            x_val_df.to_csv(os.path.join(output_dir, f'x_val_{model_version}.csv'), index=False)

            y_val_df = pd.DataFrame(y_val)
            y_val_df.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)

        if validation_perc == 0:
            return x_train, x_test, y_train, y_test
        else:
            return x_train, x_val, x_test, y_train, y_val, y_test

    else:
        if validation_perc == 0:
            x_train = pd.read_csv(os.path.join(output_dir, f'x_train_{model_version}.csv'))
            x_test = pd.read_csv(os.path.join(output_dir, f'x_test_{model_version}.csv'))
            y_train = pd.read_csv(os.path.join(output_dir, f'y_train_{model_version}.csv'))
            y_test = pd.read_csv(os.path.join(output_dir, f'y_test_{model_version}.csv'))

            return x_train, x_test, y_train, y_test

        else:
            x_train = pd.read_csv(os.path.join(output_dir, f'x_train_{model_version}.csv'))
            x_test = pd.read_csv(os.path.join(output_dir, f'x_test_{model_version}.csv'))
            x_val = pd.read_csv(os.path.join(output_dir, f'x_val_{model_version}.csv'))
            y_train = pd.read_csv(os.path.join(output_dir, f'y_train_{model_version}.csv'))
            y_test = pd.read_csv(os.path.join(output_dir, f'y_test_{model_version}.csv'))
            y_val = pd.read_csv(os.path.join(output_dir, f'y_val_{model_version}.csv'))

            return x_train, x_val, x_test, y_train, y_val, y_test


def train_model(x_train, y_train, params_dict,
                categorical_columns=None,
                load_model=False, save_model=False,
                save_folder=None, model_save_name=None):
    """
    Train a LightGBM regressor model with given hyperparameters.

    *******
    # To run the model without saving/loading the trained model, use load_model=False, save_model=False, save_folder=None,
        model_save_name=None.
    # To run the model and save it without loading any trained model, use load_model=False, save_model=True,
        save_folder='give a folder path', model_save_name='give a name'.
    # To load a pretrained model without running a new model, use load_model=True, save_model=False,
        save_folder='give the saved folder path', model_save_name='give the saved name'.
    *******

    :param x_train, y_train : x_train (predictor) and y_train (target) arrays from split_train_test_ratio() function.
    :param params_dict : ML model param dictionary. Currently supports LGBM model 'gbdt', 'goss', and 'dart'.
                  **** when tuning hyperparameters set params_dict=None.
                    For LGBM the dictionary should be like the following with user defined values-
                    param_dict = {'boosting_type': 'gbdt',
                                  'colsample_bynode': 0.7,
                                  'colsample_bytree': 0.8,
                                  'learning_rate': 0.05,
                                  'max_depth': 13,
                                  'min_child_samples': 40,
                                  'n_estimators': 250,
                                  'num_leaves': 70,
                                  'path_smooth': 0.2,
                                  'subsample': 0.7}
    :param categorical_columns: List of categorical column names to convert to 'category' dtype. Default set to None.
    :param load_model : Set to True if want to load saved model. Default set to False.
    :param save_model : Set to True if want to save model. Default set to False.
    :param save_folder : Filepath of folder to save model. Default set to None for save_model=False..
    :param model_save_name : Model's name to save with. Default set to None for save_model=False.

    :return: trained LGBM regression model.
    """
    global reg_model

    if not load_model:
        print(f'Training model...')
        start_time = timeit.default_timer()

        # provision to include categorical data
        if categorical_columns is not None:
            for col in categorical_columns:
                x_train[col] = x_train[col].astype('category')

        # Configuring the regressor with the parameters
        reg_model = LGBMRegressor(tree_learner='serial', random_state=0,
                                  deterministic=True, force_row_wise=True,
                                  n_jobs=-1, **params_dict)

        if categorical_columns is not None:
            trained_model = reg_model.fit(x_train, y_train)
        else:
            trained_model = reg_model.fit(x_train, y_train, categorical_feature=categorical_columns)

        y_pred = trained_model.predict(x_train)

        print('Train RMSE = {:.3f}'.format(calculate_rmse(Y_pred=y_pred, Y_obsv=y_train)))
        print('Train R2 = {:.3f}'.format(calculate_r2(Y_pred=y_pred, Y_obsv=y_train)))

        if save_model:
            makedirs([save_folder])
            if '.joblib' not in model_save_name:
                model_save_name = model_save_name + '.joblib'

            save_path = os.path.join(save_folder, model_save_name)
            joblib.dump(trained_model, save_path, compress=3)

        # printing and saving runtime
        end_time = timeit.default_timer()
        runtime = (end_time - start_time) / 60
        run_str = f'model training time {runtime} mins'
        print('model training time {:.3f} mins'.format(runtime))

        runtime_save = os.path.join(save_folder, model_save_name + '_training_runtime.txt')
        with open(runtime_save, 'w') as file:
            file.write(run_str)

    else:
        print('Loading trained model...')

        if '.joblib' not in model_save_name:
            model_save_name = model_save_name + '.joblib'
        saved_model_path = os.path.join(save_folder, model_save_name)
        trained_model = joblib.load(saved_model_path)
        print('Loaded trained model.')

    return trained_model


def test_model(trained_model, x_test, y_test, prediction_csv_path, categorical_columns=None):
    """
    Test a trained LightGBM regressor model's performance.

    :param trained_model: trained lightgbm model.
    :param x_test, y_test: x_test (predictor) and y_test (target) arrays from split_train_test_ratio() function.
    :param prediction_csv_path: Csv filepath to save the prediction.
    :param categorical_columns: List of categorical column names to convert to 'category' dtype. Default set to None.

    :return: trained LGBM regression model.
    """
    makedirs([os.path.basename(prediction_csv_path)])

    # provision to include categorical data
    if categorical_columns is not None:
        for col in categorical_columns:
            x_test[col] = x_test[col].astype('category')

    # testing model performonce
    y_pred_test = trained_model.predict(x_test)
    test_rmse = calculate_rmse(Y_pred=y_pred_test, Y_obsv=y_test)
    test_r2 = calculate_r2(Y_pred=y_pred_test, Y_obsv=y_test)
    test_mae = calculate_mae(Y_pred=y_pred_test, Y_obsv=y_test)

    print(f'\nRMSE = {test_rmse:.4f}')
    print(f'MAE = {test_mae:.4f}')
    print(f'R2 = {test_r2:.4f}')

    # saving test prediction
    test_obsv_predict_df = pd.DataFrame({'observed': y_test.values.ravel(),
                                         'predicted': y_pred_test})
    test_obsv_predict_df.to_csv(prediction_csv_path, index=False)


def cross_val_performance(trained_model_path,
                          x_train_df, y_train_df, k_fold=10,
                          categorical_columns=None,
                          verbose=True, skip_processing=False):
    """
    Performs k-fold cross-validation on a pre-trained model to evaluate its performance using R2 and RMSE metrics.

    :param trained_model_path: Path to the pre-trained model saved using joblib.
    :param x_train_df: The predictor variables (features) dataframe for training the model. Comes from train set.
    :param y_train_df: The target variable dataframe corresponding to the predictors. Comes from train set.
    :param k_fold: Number of folds for k-fold cross-validation. Default is 10.
    :param categorical_columns: List of categorical column names to convert to 'category' dtype. Default set to None.
    :param verbose: If True, prints detailed R2 and RMSE scores for each fold and their means.
                          If False, prints only the mean R2 and RMSE scores. Default is True.
    :param skip_processing: If True, skips cross-validation.

    :returns: None. The function prints the cross-validation scores (R2 and RMSE) to the console.

    """
    if not skip_processing:
        # provision to include categorical data
        if categorical_columns is not None:
            for col in categorical_columns:
                x_train_df[col] = x_train_df[col].astype('category')

        # loading trained model
        trained_model = joblib.load(trained_model_path)

        # performing k-fold cross-validation on the dataset (score r2)
        r2_scores = cross_val_score(trained_model, x_train_df, y_train_df,
                                    cv=k_fold, scoring='r2')

        # performing k-fold cross-validation on the dataset (score rmse)
        rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
        rmse_scores = cross_val_score(trained_model, x_train_df, y_train_df,
                                      cv=k_fold, scoring=rmse_scorer)
        rmse_scores = [-(i) for i in rmse_scores]  # making the rmse values positive

        if verbose:
            print(f'Cross-Validation R2 Scores: {r2_scores}')
            print(f'Cross-Validation Mean R2: {np.mean(r2_scores):.4f}\n')

            print(f'Cross-Validation RMSE Scores: {rmse_scores}')
            print(f'Cross-Validation Mean RMSE: {np.mean(rmse_scores):.4f}\n')

        else:
            print(f'Cross-Validation Mean R2: {np.mean(r2_scores):.4f}')
            print(f'Cross-Validation Mean RMSE: {np.mean(rmse_scores):.4f}\n')


    else:
        pass


def create_pdplots(trained_model, x_train, features_to_include, output_dir, plot_name,
                   ylabel='Effective Precipitation \n (mm)',
                   skip_processing=False):
    """
    Plot partial dependence plot.

    :param trained_model: Trained model object.
    :param x_train: x_train dataframe (if the model was trained with a x_train as dataframe) or array.
    :param features_to_include: List of features for which PDP plots will be made. If set to 'All', then PDP plot for
                                all input variables will be created.
    :param output_dir: Filepath of output directory to save the PDP plot.
    :param plot_name: str of plot name. Must include '.jpeg' or 'png'.
    :param ylabel: Ylabel for partial dependence plot. Default set to Effective Precipitation \n (mm)' for monthly model.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        # creating variables for unit degree and degree celcius
        deg_unit = r'$^\circ$'
        deg_cel_unit = r'$^\circ$C'

        # plotting
        if features_to_include == 'All':  # to plot PDP for all attributes
            features_to_include = list(x_train.columns)

        plt.rcParams['font.size'] = 30

        pdisp = PDisp.from_estimator(trained_model, x_train, features=features_to_include,
                                     percentiles=(0.05, 1), subsample=0.8, grid_resolution=20,
                                     n_jobs=-1, random_state=0)


        # Subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)',
                          '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)']

        # replacing x and y axis labels
        row_num = range(0, pdisp.axes_.shape[0])
        col_num = range(0, pdisp.axes_.shape[1])

        feature_idx = 0
        for r in row_num:
            for c in col_num:
                if pdisp.axes_[r][c] is not None:
                    # subplot num
                    pdisp.axes_[r][c].text(0.1, 0.9, subplot_labels[feature_idx], transform=pdisp.axes_[r][c].transAxes,
                                           fontsize=35, va='top', ha='left')

                    feature_idx += 1
                else:
                    pass

        for row_idx in range(0, pdisp.axes_.shape[0]):
            pdisp.axes_[row_idx][0].set_ylabel(ylabel)

        fig = plt.gcf()
        fig.set_size_inches(30, 30)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')

        print('PDP plots generated...')

    else:
        pass


def plot_permutation_importance(trained_model, x_test, y_test, output_dir, plot_name,
                                saved_var_list_name,
                                exclude_columns=None, skip_processing=False):
    """
    Plot permutation importance for model predictors.

    :param trained_model: Trained ML model object.
    :param x_test: Filepath of x_test csv or dataframe. In case of dataframe, it has to come directly from the
                    split_train_val_test_set() function.
    :param y_test: Filepath of y_test csv or dataframe.
    :param exclude_columns: List of predictors to be excluded.
                            Exclude the same predictors for which model wasn't trained. In case the x_test comes as a
                            dataframe from the split_train_val_test_set() function, set exclude_columns to None.
    :param output_dir: Output directory filepath to save the plot.
    :param plot_name: Plot name. Must contain 'png', 'jpeg'.
    :param saved_var_list_name: The name of to use to save the sorted important vars list. Must contain 'pkl'.
    :param skip_processing: Set to True to skip this process.

    :return: List of sorted (most important to less important) important variable names.
    """
    if not skip_processing:
        makedirs([output_dir])

        if '.csv' in x_test:
            # Loading x_test and y_test
            x_test_df = pd.read_csv(x_test)
            x_test_df = x_test_df.drop(columns=exclude_columns)
            x_test_df = reindex_df(x_test_df)

            y_test_df = pd.read_csv(y_test)
        else:
            x_test_df = x_test
            y_test_df = y_test

        # ensure arrays are writable  (the numpy conversion code block was added after a conda env upgrade threw 'WRITABLE array'
        #                              error, took chatgpt's help to figure this out. The error meant - permutation_importance() was
        #                              trying to change the array but could not as it was writable before. This code black makes the
        #                              arrays writable)
        x_test_np = x_test_df.to_numpy()
        y_test_np = y_test_df.to_numpy()

        x_test_np.setflags(write=True)
        y_test_np.setflags(write=True)

        # generating permutation importance score on test set
        result_test = permutation_importance(trained_model, x_test_np, y_test_np,
                                             n_repeats=30, random_state=0, n_jobs=-1, scoring='r2')

        sorted_importances_idx = result_test.importances_mean.argsort()
        predictor_cols = x_test_df.columns
        importances = pd.DataFrame(result_test.importances[sorted_importances_idx].T,
                                   columns=predictor_cols[sorted_importances_idx])

        # sorted important variables
        sorted_imp_vars = importances.columns.tolist()[::-1]
        print('\n', 'Sorted Important Variables:', sorted_imp_vars, '\n')

        # plotting
        plt.figure(figsize=(6, 4))

        ax = importances.plot.box(vert=False, whis=10)
        ax.axvline(x=0, color='k', linestyle='--')
        ax.set_xlabel('Relative change in accuracy', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_name), dpi=200)

        # saving the list to avoid running the permutation importance plot if not required (saves model running time)
        joblib.dump(sorted_imp_vars, os.path.join(output_dir, saved_var_list_name))

        print('Permutation importance plot generated...')

    else:
        sorted_imp_vars = joblib.load(os.path.join(output_dir, saved_var_list_name))

    return sorted_imp_vars
