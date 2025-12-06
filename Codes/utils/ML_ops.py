# author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
import csv
import pickle
import joblib
import timeit
import numpy as np
import pandas as pd
from glob import glob
import dask.dataframe as ddf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from timeit import default_timer as timer

# import skexplain
import shap
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.inspection import PartialDependenceDisplay as PDisp
from sklearn.model_selection import train_test_split, cross_val_score

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, clip_resample_reproject_raster
from Codes.utils.stats_ops import calculate_metrics

no_data_value = -9999
model_res = 0.02000000000000000389  # in deg, 2 km
WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'


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

        makedirs([os.path.dirname(output_parquet)])

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
                        variable_dict['year'] = [year] * len(data_arr)
                    else:
                        variable_dict[var].extend(list(data_arr))
                        variable_dict['year'].extend([year] * len(data_arr))

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

        # applying irrigated cropland filter
        if 'irr_crop_frac' in train_test_ddf.columns:
            train_test_ddf = train_test_ddf[train_test_ddf['irr_crop_frac'] >= 0.02]

        if '.parquet' in output_parquet:
            train_test_ddf.to_parquet(output_parquet, write_index=False)

        elif '.csv' in output_parquet:
            train_test_df = train_test_ddf.compute()
            train_test_df.to_csv(output_parquet, index=False)

        return output_parquet

    else:
        return output_parquet


def split_train_val_test_set_v1(input_csv, pred_attr, exclude_columns, output_dir,
                                model_version, test_perc=0.3, validation_perc=0,
                                random_state=0, verbose=True, stratify=True,
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
    :param stratify: Set to True if want to stratify data based on 'stateID'.
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    returns: X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """

    if not skip_processing:
        print('\nSplitting train-test dataframe into train and test dataset...')

        # reading data
        input_df = pd.read_parquet(input_csv)

        # stratification column extraction
        if stratify:  # based on StateID
            stratify_col = input_df['stateID']
        else:
            stratify_col = None

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

        # train-test splitting
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_perc, random_state=random_state,
                                                            shuffle=True, stratify=stratify_col)
        if validation_perc > 0:
            stratify_val = stratify_col.loc[x_train.index] if stratify else None
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_perc,
                                                              random_state=random_state, shuffle=True,
                                                              stratify=stratify_val)

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


def split_train_val_test_set_v2(data_parquet, pred_attr, exclude_columns, output_dir,
                                model_version, train_size=0.7, test_size=0.3,
                                random_state=42, skip_processing=False):
    """
    Splits the tiles into train, validation, and test datasets, along with their target values.

    :return: None.
    """
    if not skip_processing:

        df = pd.read_parquet(data_parquet)

        # replacing samples (very few) with stateID 3 - Nebraska and 1 -  Oklahoma. They belong to kansas
        # but came in Nebraska and Oklahoma during stateID raster creation (along state border)
        # otherwise train_val_test split function throws error
        df.loc[df['stateID'] == 3, 'stateID'] = 12
        df.loc[df['stateID'] == 1, 'stateID'] = 12

        # getting unique pixelID and stateID
        unique_pixels = df[['pixelID', 'stateID']].drop_duplicates()

        # Splitting at the pixelID level
        train_pixels, test_pixels = train_test_split(unique_pixels, train_size=train_size, test_size=test_size,
                                                     stratify=unique_pixels['stateID'],
                                                     random_state=random_state)

        # assigning dataset labels to full dataset
        df['split'] = 'test'  # Default to test
        df.loc[df['pixelID'].isin(train_pixels['pixelID']), 'split'] = 'train'

        # splitting into separate DataFrames
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']

        # dropping columns that has been specified to not include
        if exclude_columns is not None:
            drop_columns = exclude_columns + ['split', pred_attr]
        else:
            drop_columns = ['split', pred_attr]

        # separating x_train and y_train
        x_train = train_df.drop(columns=drop_columns)
        x_test = test_df.drop(columns=drop_columns)

        y_train = train_df[[pred_attr]]
        y_test = test_df[[pred_attr]]

        # Reindexing for ensuring that columns go into the model in same serial every time
        x_train = reindex_df(x_train)
        x_test = reindex_df(x_test)

        # saving
        x_train.to_csv(os.path.join(output_dir, f'x_train_{model_version}.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, f'y_train_{model_version}.csv'), index=False)
        x_test.to_csv(os.path.join(output_dir, f'x_test_{model_version}.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, f'y_test_{model_version}.csv'), index=False)

        return x_train, x_test, y_train, y_test

    else:
        x_train = pd.read_csv(os.path.join(output_dir, f'x_train_{model_version}.csv'))
        x_test = pd.read_csv(os.path.join(output_dir, f'x_test_{model_version}.csv'))
        y_train = pd.read_csv(os.path.join(output_dir, f'y_train_{model_version}.csv'))
        y_test = pd.read_csv(os.path.join(output_dir, f'y_test_{model_version}.csv'))

        return x_train, x_test, y_train, y_test


def objective_func_bayes(params, train_set, iteration_csv, n_fold):
    """
    Objective function for Bayesian optimization using Hyperopt and LightGBM.

    **** Bayesian optimization doesn't directly optimize the objective function. Instead, it builds a probabilistic
    model (a surrogate) to predict promising regions. In our case its the TPE in bayes_hyperparam_opt() function.

    :param params: Hyperparameter space to use while optimizing.
    :param train_set: A LGBM dataset. Constructed within the bayes_hyperparam_opt() func using x_train and y_train.
    :param iteration_csv: Filepath of a csv where hyperparameter iteration step will be stored.
    :param n_fold: KFold cross validation number. Usually 5 or 10.

    :return : A dictionary after each iteration holding rmse, params, run_time, etc.
    """
    global ITERATION
    ITERATION += 1

    start = timer()

    # converting the train_set (dataframe) to LightGBM Dataset
    train_set = lgb.Dataset(train_set.iloc[:, :-1], label=train_set.iloc[:, -1])

    # retrieve the boosting type and subsample (if not present set subsample to 1)
    subsample = params['boosting_type'].get('subsample', 1)
    params['subsample'] = subsample
    params['boosting_type'] = params['boosting_type']['boosting_type']

    # inserting a new parameter in the dictionary to handle 'goss'
    # the new version of LIGHTGBM handles 'goss' as 'boosting_type' = 'gdbt' & 'data_sample_strategy' = 'goss'
    if params['boosting_type'] == 'goss':
        params['boosting_type'] = 'gbdt'
        params['data_sample_strategy'] = 'goss'

    # ensure integer type for integer hyperparameters
    for parameter_name in ['n_estimators', 'num_leaves', 'min_child_samples', 'max_depth', 'max_drop']:
        params[parameter_name] = int(params[parameter_name])

    # callbacks
    callbacks = [
        # lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0)
    ]

    # perform n_fold cross validation
    # ** not using num_boost_round and early stopping as we are providing n_estimators in the param_space **
    cv_results = lgb.cv(params, train_set,
                        # num_boost_round=10000,
                        nfold=n_fold,
                        stratified=False, metrics='rmse', seed=50,
                        callbacks=callbacks)

    run_time = timer() - start

    # best score extraction
    # the try-except block was inserted because of two versions of LIGHTGBM is desktop and server. The server
    # version used keyword 'valid rmse-mean' while the desktop version was using 'rmse-mean'
    try:
        best_rmse = np.min(cv_results[
                               'valid rmse-mean'])  # valid rmse-mean stands for mean RMSE value across all the folds for each boosting round

    except:
        best_rmse = np.min(cv_results['rmse-mean'])

    # result of each iteration will be store in the iteration_csv
    makedirs([os.path.dirname(iteration_csv)])

    if ITERATION == 1:
        write_to = open(iteration_csv, 'w')
        writer = csv.writer(write_to)
        writer.writerows([['loss', 'params', 'iteration', 'run_time'],
                          [best_rmse, params, ITERATION, run_time]])
        write_to.close()

    else:  # when ITERATION > 0, will append result on the existing csv/file
        write_to = open(iteration_csv, 'a')
        writer = csv.writer(write_to)
        writer.writerow([best_rmse, params, ITERATION, run_time])

    # dictionary with information for evaluation
    return {'loss': best_rmse, 'params': params,
            'iteration': ITERATION, 'train_time': run_time, 'status': STATUS_OK}


def bayes_hyperparam_opt(x_train, y_train, iteration_csv, n_fold=10, max_evals=1000, skip_processing=False):
    """
    Hyperparameter optimization using Bayesian optimization method.

    *****
    good resources for building LGBM model

    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    https://lightgbm.readthedocs.io/en/latest/Parameters.html
    https://neptune.ai/blog/lightgbm-parameters-guide

    Bayesian Hyperparameter Optimization:
    details at:https://www.geeksforgeeks.org/bayesian-optimization-in-machine-learning/

    coding help from:
    1. https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb
    2. https://www.kaggle.com/code/prashant111/bayesian-optimization-using-hyperopt
    *****

    :param x_train, y_train : Predictor and target arrays from split_train_test_ratio() function.
    :param iteration_csv: Filepath of a csv where hyperparameter iteration step will be stored.
    :param n_fold : Number of folds in K Fold CV. Default set to 10.
    :param max_evals : Maximum number of evaluations during hyperparameter optimization. Default set to 1000.
    :param skip_processing: Set to True to skip hyperparameter tuning. Default set to False.

    :return : Best hyperparameters' dictionary.
    """
    if not skip_processing:
        print(f'performing bayesian hyperparameter optimization...')

        # merging x_train and y_train into a single dataset
        train_set = pd.concat([x_train, y_train], axis=1)

        # creating hyperparameter space for LGBM models
        param_space = {'boosting_type': hp.choice('boosting_type',
                                                  [
                                                      {
                                                          'boosting_type': 'dart',
                                                          'subsample': hp.uniform('dart_subsample', 0.5, 0.8)
                                                      },
                                                  ]),
                       'drop_rate': hp.uniform('drop_rate', 0.05, 0.3),  # 5–30% dropout rate
                       'max_drop': hp.quniform('max_drop', 10, 80, 5),  # number of trees that can be dropped
                       'skip_drop': hp.uniform('skip_drop', 0.4, 0.7),  # probability to skip dropout in an iteration
                       'n_estimators': hp.quniform('n_estimators', 200, 400, 25),
                       'max_depth': hp.uniform('max_depth', 4, 8),
                       'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.02)),
                       'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 0.8),
                       'colsample_bynode': hp.uniform('colsample_bynode', 0.6, 0.8),
                       'path_smooth': hp.uniform('path_smooth', 0.5, 0.8),
                       'num_leaves': hp.quniform('num_leaves', 15, 50, 5),
                       'min_child_samples': hp.quniform('min_child_samples', 50, 100, 5),
                       'force_col_wise': True  # set to False to choose between colum/row-wise parallelization
                       }

        # optimization algorithm
        tpe_algorithm = tpe.suggest  # stand for Tree-structured Parzen Estimator. A surrogate (probabilistic) model
        # of the objective function, which predicts which hyperparameters are promising
        # instead of directly testing every hyperparameter. The hyperparameter tuning
        # approach, Sequential model-based optimization (SMBO), will try to closely match the
        # surrogate function to the objective function

        # keeping track of results
        bayes_trials = Trials()  # The Trials object will hold everything returned from the objective function in the

        # .results attribute. It also holds other information from the search, but we return
        # everything we need from the objective.

        # creating a wrapper function to bring all arguments of objective_func_bayes() under a single argument
        def objective_wrapper(params):
            return objective_func_bayes(params, train_set, iteration_csv, n_fold)

        # implementation of Sequential model-based optimization (SMBO)
        global ITERATION
        ITERATION = 0

        # run optimization
        # hyperopt fmin inherently has a acquisition function that decides where to sample next.
        # It balances exploitation (testing regions known to be good) vs. exploration (trying new regions).
        fmin(fn=objective_wrapper, space=param_space, algo=tpe_algorithm,
             max_evals=max_evals, trials=bayes_trials, rstate=np.random.default_rng(50))

        # sorting the trials to get the set of hyperparams with lowest loss
        bayes_trials_results = sorted(bayes_trials.results[1:],
                                      key=lambda x: x['loss'],
                                      reverse=False)  # the indexing in the results is done to remove {'status': 'new'} at 0 index
        best_hyperparams = bayes_trials_results[0]['params']

        print('\n')
        print('best hyperparameter set', '\n', best_hyperparams, '\n')
        print('best RMSE:', bayes_trials.results[1]['loss'])

        return best_hyperparams

    else:
        pass


def train_model(x_train, y_train, params_dict,
                categorical_columns=None,
                load_model=False, save_model=False,
                save_folder=None, model_save_name=None,
                skip_tune_hyperparameters=False,
                iteration_csv=None, n_fold=10, max_evals=1000,
                verbose=True):
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
                                  'subsample': 0.7,
                                  'drop_rate': 0.2,
                                  'max_drop': 50,
                                  'skip_drop': 0.4,
                                  'colsample_bynode': 0.7,
                                  'colsample_bytree': 0.8,
                                  'learning_rate': 0.05,
                                  'max_depth': 13,
                                  'min_child_samples': 40,
                                  'n_estimators': 250,
                                  'num_leaves': 70,
                                  'path_smooth': 0.2,
                                  }
    :param categorical_columns: List of categorical column names to convert to 'category' dtype. Default set to None.
    :param load_model : Set to True if want to load saved model. Default set to False.
    :param save_model : Set to True if want to save model. Default set to False.
    :param save_folder : Filepath of folder to save model. Default set to None for save_model=False..
    :param model_save_name : Model's name to save with. Default set to None for save_model=False.
    :param skip_tune_hyperparameters: Set to True to skip hyperparameter tuning. Default set to False.
    :param iteration_csv : Filepath of a csv where hyperparameter iteration step will be stored.
    :param n_fold : Number of folds in K Fold CV. Default set to 10.
    :param max_evals : Maximum number of evaluations during hyperparameter optimization. Default set to 1000.
    :param verbose : Set to False to skip printing model metrics.

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

        # hyperparameter tuning if enables
        if not skip_tune_hyperparameters:
            params_dict = bayes_hyperparam_opt(x_train, y_train, iteration_csv,
                                               n_fold=n_fold, max_evals=max_evals,
                                               skip_processing=skip_tune_hyperparameters)

        # Configuring the regressor with the parameters
        reg_model = LGBMRegressor(tree_learner='serial', random_state=0,
                                  deterministic=True, n_jobs=-1, **params_dict)

        if categorical_columns is not None:
            trained_model = reg_model.fit(x_train, y_train,
                                          categorical_feature=categorical_columns)
        else:
            trained_model = reg_model.fit(x_train, y_train)

        y_pred = trained_model.predict(x_train)

        # performance/error metrics
        metrics_dict = calculate_metrics(predictions=y_pred, targets=y_train.values.ravel())
        rmse = metrics_dict['RMSE']
        mae = metrics_dict['MAE']
        r2 = metrics_dict['R2']
        nrmse = metrics_dict['Normalized RMSE']
        nmae = metrics_dict['Normalized MAE']

        if verbose:
            print(
                f"Train Results:\n"
                f"---------------------\n"
                f"RMSE: {rmse:.4f}, MAE: {mae:.4f},\n"
                f"NRMSE: {nrmse:.4f}, NMAE: {nmae:.4f}, R²: {r2:.4f}\n"
            )

        # save trained model
        if save_model:
            makedirs([save_folder])
            if '.joblib' not in model_save_name:
                model_save_name = model_save_name + '.joblib'

            save_path = os.path.join(save_folder, model_save_name)
            joblib.dump(trained_model, save_path, compress=3)

        # printing and saving runtime
        if verbose:
            end_time = timeit.default_timer()
            runtime = (end_time - start_time) / 60
            run_str = f'model training time {runtime} mins'
            print('model training time {:.3f} mins'.format(runtime))

            runtime_save = os.path.join(save_folder, model_save_name + '_training_runtime.txt')
            with open(runtime_save, 'w') as file:
                file.write(run_str)

    else:
        print('Loading trained model...')

        if verbose:
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
    makedirs([os.path.dirname(prediction_csv_path)])

    # provision to include categorical data
    if categorical_columns is not None:
        for col in categorical_columns:
            x_test[col] = x_test[col].astype('category')

    # testing model performance
    y_pred_test = trained_model.predict(x_test)

    # performance/error metrics
    metrics_dict = calculate_metrics(predictions=y_pred_test, targets=y_test.values.ravel())

    rmse = metrics_dict['RMSE']
    mae = metrics_dict['MAE']
    r2 = metrics_dict['R2']
    nrmse = metrics_dict['Normalized RMSE']
    nmae = metrics_dict['Normalized MAE']

    print(
        f"RMSE: {rmse:.4f}, MAE: {mae:.4f},\n"
        f"NRMSE: {nrmse:.4f}, NMAE: {nmae:.4f}, R²: {r2:.4f}\n"
    )

    # saving test prediction
    test_obsv_predict_df = pd.DataFrame({'actual': y_test.values.ravel(),
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
                   ylabel='Pumping (mm/year)', categorical_columns=None,
                   skip_processing=False):
    """
    Plot partial dependence plot.

    :param trained_model: Trained model object.
    :param x_train: x_train dataframe (if the model was trained with a x_train as dataframe) or array.
    :param features_to_include: List of features for which PDP plots will be made. If set to 'All', then PDP plot for
                                all input variables will be created.
    :param output_dir: Filepath of output directory to save the PDP plot.
    :param plot_name: str of plot name. Must include '.jpeg' or 'png'.
    :param ylabel: Ylabel for partial dependence plot. Default set to 'Pumping (mm/year)'.
    :param categorical_columns: List of categorical column names to convert to 'category' dtype. Default set to None.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        # print(trained_model._Booster.dump_model()['feature_infos']) # prints features info like min-max values

        # creating variables for unit degree and degree celsius
        deg_unit = r'$^\circ$'
        deg_cel_unit = r'$^\circ$C'

        # provision to include categorical data
        if categorical_columns is not None:
            for col in categorical_columns:
                x_train[col] = x_train[col].astype('category')

        # plotting
        if features_to_include == 'All':  # to plot PDP for all attributes
            features_to_include = list(x_train.columns)

        plt.rcParams.update({'font.size': 30})

        pdisp = PDisp.from_estimator(trained_model, x_train, features=features_to_include,
                                     categorical_features=categorical_columns, percentiles=(0.05, 1),
                                     subsample=0.8, grid_resolution=20, n_jobs=-1, random_state=0)

        # creating a dictionary to rename PDP plot labels
        feature_dict = {
            'netGW_Irr': 'Consumptive groundwater use (mm/gs)', 'peff': 'Effective precipitation (mm/gs)',
            'ret': 'Reference ET (mm/gs)', 'precip': 'Precipitation (mm/gs)',
            'tmax': f'Max. temperature ({deg_cel_unit})',
            'ET': 'ET (mm/gs)', 'irr_crop_frac': 'Fraction of irrigated cropland',
            'maxRH': 'Max. relative humidity (%)',
            'minRH': 'Min. relative humidity (%)', 'shortRad': 'Downward shortwave radiation (W/$m^2$)',
            'vpd': 'Vapor pressure deficit (kpa)', 'sunHr': 'Daylight duration (hr)',
            'SW_Irr': 'Surface water irrigation (mm/gs)', 'FC': 'Field capacity',
            'Canal_density': 'Canal density', 'Canal_distance': 'Distance from canals'
        }

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
                    # changing axis labels
                    pdisp.axes_[r][c].set_xlabel(feature_dict[features_to_include[feature_idx]], fontsize=30)

                    # subplot num
                    pdisp.axes_[r][c].text(0.1, 0.9, subplot_labels[feature_idx], transform=pdisp.axes_[r][c].transAxes,
                                           fontsize=30, va='top', ha='left')

                    # adjusting size of tick params
                    pdisp.axes_[r][c].tick_params(axis='both', labelsize=30)

                    feature_idx += 1
                else:
                    pass

        for row_idx in range(0, pdisp.axes_.shape[0]):
            pdisp.axes_[row_idx][0].set_ylabel(ylabel, fontsize=30)

        fig = plt.gcf()
        fig.set_size_inches(30, 30)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(os.path.join(output_dir, plot_name), dpi=100, bbox_inches='tight')

        print('PDP plots generated...')

    else:
        pass


def plot_permutation_importance(trained_model, x_test, y_test, output_dir, plot_name,
                                sorted_var_list_name, categorical_columns=None,
                                skip_processing=False):
    """
    Plot permutation importance for model predictors.

    :param trained_model: Trained ML model object.
    :param x_test: Filepath of x_test csv or dataframe. In case of dataframe, it has to come directly from the
                    split_train_val_test_set_v1() function.
    :param y_test: Filepath of y_test csv or dataframe.
    :param output_dir: Output directory filepath to save the plot.
    :param plot_name: Plot name. Must contain 'png', 'jpeg'.
    :param sorted_var_list_name: The name of to use to save the sorted important vars list. Must contain 'pkl'.
    :param categorical_columns: List of categorical column names to convert to 'category' dtype. Default set to None.
    :param skip_processing: Set to True to skip this process.

    :return: List of sorted (most important to less important) important variable names.
    """
    if not skip_processing:
        makedirs([output_dir])

        if '.csv' in x_test:
            # Loading x_test and y_test
            x_test_df = pd.read_csv(x_test)
            x_test_df = reindex_df(x_test_df)

            y_test_df = pd.read_csv(y_test)
        else:
            x_test_df = x_test
            y_test_df = y_test

        # provision to include categorical data
        if categorical_columns is not None:
            for col in categorical_columns:
                x_test_df[col] = x_test_df[col].astype('category')

        # Ensure arrays are writable to prevent errors during permutation importance.
        # Some NumPy arrays created from Pandas DataFrames are read-only by default.
        # The permutation_importance() function modifies arrays, so they must be writable.
        x_test_np = x_test_df.to_numpy()
        y_test_np = y_test_df.to_numpy()

        x_test_np.setflags(write=True)
        y_test_np.setflags(write=True)

        # generating permutation importance score on test set
        # using n_job = 1 here to avoid error due to parallelization issue in Linux (nothing wrong with the code)
        result_test = permutation_importance(trained_model, x_test_np, y_test_np,
                                             n_repeats=30, random_state=0, n_jobs=1, scoring='r2')

        sorted_importances_idx = result_test.importances_mean.argsort()
        predictor_cols = x_test_df.columns
        importances = pd.DataFrame(result_test.importances[sorted_importances_idx].T,
                                   columns=predictor_cols[sorted_importances_idx])

        # sorted important variables
        sorted_imp_vars = importances.columns.tolist()[::-1]
        print('\nSorted Important Variables:', sorted_imp_vars, '\n')

        # renaming input variables
        feature_name_dict = {
            'netGW_Irr': 'Consumptive groundwater use', 'peff': 'Effective precipitation',
            'ret': 'Reference ET', 'precip': 'Precipitation', 'tmax': f'Max. temperature',
            'ET': 'ET', 'irr_crop_frac': 'Fraction of irrigated cropland', 'maxRH': 'Max. relative humidity',
            'minRH': 'Min. relative humidity', 'shortRad': 'Downward shortwave radiation',
            'vpd': 'Vapor pressure deficit', 'sunHr': 'Daylight duration',
            'sw_huc12': 'Normalized HUC12 Surface water irrigation', 'gw_perc_huc12': 'Groundwater use % at HUC12',
            'climate': 'Climate', 'FC': 'Field capacity', 'spi': 'Standardized precipitation index',
            'spei': 'Standardized precipitation evapotranspiration index', 'eddi': 'Evaporative demand drought index',
            'Canal_density': 'Canal density', 'Canal_distance': 'Distance from canals',
            'irr_eff': 'Irrigation Efficiency'
        }

        importances = importances.rename(columns=feature_name_dict)

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
        joblib.dump(sorted_imp_vars, os.path.join(output_dir, sorted_var_list_name))

        print('Permutation importance plot generated...\n')

    else:
        sorted_imp_vars = joblib.load(os.path.join(output_dir, sorted_var_list_name))

    return sorted_imp_vars


def plot_shap_summary_plot(trained_model_path, use_samples,
                           data_csv, exclude_features, save_plot_path,
                           skip_processing=False):
    """
    Generate and save a SHAP summary (beeswarm) plot to visualize feature importance.

    :param trained_model_path: str
        Path to the `.joblib` file containing the saved model's state_dict.

    :param use_samples: int
        Number of samples to randomly draw from the dataset for SHAP analysis.

    :param data_csv: str or DataFrame
        Path to the CSV file containing the input feature data. Or a dataframe of input features.

    :param exclude_features: list
        List of column names to exclude from the input (e.g., pumping, IDs).

    :param save_plot_path: str
        File path (with extension) where the SHAP summary plot will be saved.

    :param skip_processing: bool, optional (default=False)
        If True, skip execution.

    :return: None
    """
    if not skip_processing:
        makedirs([os.path.dirname(save_plot_path)])

        # loading model
        trained_model = joblib.load(trained_model_path)

        print(trained_model)

        print('\n___________________________________________________________________________')
        print(f'\nplotting SHAP feature importance...')

        # making sure to exclude training data.
        # only one of these was used for training.
        for col in ['pumping_mm', 'consumptive_gw']:
            if col not in exclude_features:
                exclude_features.append(col)

        # loading data + random sampling + renaming dataframe features
        if '.csv' in data_csv:
            df = pd.read_csv(data_csv)
            df = df.drop(columns=exclude_features)

        elif isinstance(data_csv, pd.DataFrame):
            df = data_csv.drop(columns=exclude_features, errors="ignore")

        df = df.sample(n=use_samples, random_state=43)  # sampling 'use_samples' of rows for SHAP plotting

        feature_names_dict = {'netGW_Irr': 'Consumptive groundwater use', 'peff': 'Effective precipitation',
                              'SW_Irr': 'Surface water irrigation', 'ret': 'Reference ET', 'precip': 'Precipitation',
                              'tmax': 'Temperature (max)', 'ET': 'ET', 'irr_crop_frac': 'Irrigated crop fraction',
                              'maxRH': 'Relative humidity (max)', 'minRH': 'Relative humidity (min)',
                              'shortRad': 'Shortwave radiation', 'vpd': 'Vapor pressure deficit',
                              'sunHr': 'Sun hour', 'FC': 'Field capacity',
                              'Canal_distance': 'Distance from canal', 'Canal_density': 'Canal density',
                              'irr_eff': 'Irrigation Efficiency'
                              }
        df = df.rename(columns=feature_names_dict)
        feature_names = np.array(df.columns.tolist())

        # using SHAP TreeExplainer to estimate shap values
        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer(df)

        # converting SHAP values to numpy for plotting
        # shap_values_np = shap_values.values.squeeze(-1)     # Remove singleton third dimension from SHAP array: shape [n, m, 1] → [n, m]
        # print(shap_values_np.shape)
        # plotting
        fig = plt.figure()
        shap.summary_plot(shap_values, df, feature_names=feature_names)

        fig.savefig(save_plot_path, dpi=200, bbox_inches='tight')

    else:
        pass


def plot_shap_interaction_plot(model_version, trained_model_path, use_samples, features_to_plot,
                               data_csv, exclude_features_from_df, save_plot_dir,
                               skip_processing=False):
    """
    Generate and save a SHAP summary (beeswarm) plot to visualize feature importance.

    :param model_version: str
        Model version.
    :param trained_model_path: str
        Path to the `.joblib` file containing the saved model's state_dict.

    :param use_samples: int
        Number of samples to randomly draw from the dataset for SHAP analysis.

    :param features_to_plot: List
        List of features to include in the interaction plot.

    :param data_csv: str or DataFrame
        Path to the CSV file containing the input feature data. Or a dataframe of input features.

    :param exclude_features_from_df: list
        List of column names to exclude from the input (e.g., pumping_mm, IDs).

    :param save_plot_dir: str
        Directory path where the SHAP interaction plot will be saved.

    :param skip_processing: bool, optional (default=False)
        If True, skip execution.

    :return: None
    """
    if not skip_processing:
        makedirs([save_plot_dir])

        # loading model
        trained_model = joblib.load(trained_model_path)

        print(trained_model)

        print('\n___________________________________________________________________________')
        print(f'\nplotting SHAP interaction plot...')

        # making sure to exclude training data.
        # only one of these was used for training.
        for col in ['pumping_mm', 'consumptive_gw']:
            if col not in exclude_features_from_df:
                exclude_features_from_df.append(col)

        # loading data + random sampling + renaming dataframe features
        if '.csv' in data_csv:
            df = pd.read_csv(data_csv)
            df = df.drop(columns=exclude_features_from_df)

        elif isinstance(data_csv, pd.DataFrame):
            df = data_csv.drop(columns=exclude_features_from_df, errors="ignore")

        df = df.sample(n=use_samples, random_state=43)  # sampling 'use_samples' of rows for SHAP plotting

        feature_names_dict = {'netGW_Irr': 'Consumptive groundwater use', 'peff': 'Effective precipitation',
                              'SW_Irr': 'Surface water irrigation', 'ret': 'Reference ET', 'precip': 'Precipitation',
                              'tmax': 'Temperature (max)', 'ET': 'ET', 'irr_crop_frac': 'Irrigated crop fraction',
                              'maxRH': 'Relative humidity (max)', 'minRH': 'Relative humidity (min)',
                              'shortRad': 'Shortwave radiation', 'vpd': 'Vapor pressure deficit',
                              'sunHr': 'Sun hour', 'FC': 'Field capacity',
                              'Canal_distance': 'Distance from canal',
                              'Canal_density': 'Canal density', 'irr_eff': 'Irrigation Efficiency'}

        df = df.rename(columns=feature_names_dict)
        feature_names = np.array(df.columns.tolist())

        # using SHAP TreeExplainer to estimate shap values
        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer(df)

        if hasattr(shap_values, "values"):
            shap_values_np = shap_values.values
        else:
            shap_values_np = shap_values

        # plotting and saving individual shap dependence plots
        for feature in features_to_plot:
            shap.dependence_plot(feature, shap_values_np, df, feature_names=feature_names,
                                 interaction_index=None, show=False)
            plt.gca().set_ylabel('SHAP value', fontsize=14)
            plt.gca().set_xlabel(feature, fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            plt.savefig(os.path.join(save_plot_dir, f'{feature}.png'), dpi=400, bbox_inches='tight')

        # compiling individual shap plot in a grid plot
        n_cols = 3
        n_rows = (len(features_to_plot) + n_cols - 1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axs = axs.flatten()

        for i, feature in enumerate(features_to_plot):
            img = mpimg.imread(os.path.join(save_plot_dir, f'{feature}.png'))
            axs[i].imshow(img)
            axs[i].axis('off')

        # hiding unused axes if any
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        # saving plot
        plt.tight_layout(pad=0.1)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(os.path.join(save_plot_dir, f'SHAP_interaction_all_{model_version}.png'), dpi=200,
                    bbox_inches='tight')
    else:
        pass


def create_annual_dataframes_for_pumping_prediction(years_list, yearly_data_path_dict,
                                                    static_data_path_dict, datasets_to_include,
                                                    irrigated_cropland_dir,
                                                    output_dir, skip_processing=False):
    """
    Create annual dataframes of predictors to generate annual pumping prediction.

    :param years_list: A list of years_list for which data to include in the dataframe.
    :param yearly_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
                                  Set to None if there is static dataset.
    :param static_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
                                  Set to None if there is no yearly dataset.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param irrigated_cropland_dir: Filepath of directory of irrigated cropland raster.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip this dataframe creation process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        for year in years_list:  # 1st loop controlling years_list
            print(f'creating dataframe for prediction - year {year}...')

            variable_dict = {}  # empty dict to store data for each variable

            # reading yearly data and storing it in a dictionary
            for var in yearly_data_path_dict.keys():
                if var in datasets_to_include:
                    yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]
                    data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                    data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                    variable_dict[var] = list(data_arr)

            # reading static data and storing it in a dictionary
            if static_data_path_dict is not None:
                for var in static_data_path_dict.keys():
                    if var in datasets_to_include:
                        static_data = glob(os.path.join(static_data_path_dict[var], '*.tif'))[0]
                        data_arr = read_raster_arr_object(static_data, get_file=False).flatten()

                        # storing data
                        data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                        variable_dict[var] = list(data_arr)

            # storing collected data into a dataframe
            predictor_df = pd.DataFrame(variable_dict)
            predictor_df = predictor_df.dropna()

            # saving input predictor csv
            annual_output_csv = os.path.join(output_dir, f'predictors_{year}.csv')
            predictor_df.to_csv(annual_output_csv, index=False)

            # creating nan position dictionary based on irrigated cropland data and saving it
            irrigated_cropland_raster = glob(os.path.join(irrigated_cropland_dir, f'*{year}.tif'))[0]
            irr_arr = read_raster_arr_object(irrigated_cropland_raster, get_file=False)

            nan_pos_dict = {'irr': np.isnan(irr_arr).flatten()}
            dict_path = os.path.join(output_dir, f'nan_pos_{year}.pkl')
            pickle.dump(nan_pos_dict, open(dict_path, mode='wb+'))
    else:
        pass


def predict_annual_pumping_rasters(trained_model, years_list, exclude_columns,
                                   predictor_csv_and_nan_pos_dir,
                                   prediction_name_keyword, output_dir,
                                   ref_raster=WestUS_raster, skip_processing=False):
    """
    Create annual pumping prediction raster.

    :param trained_model: Trained ML model object.
    :param years_list: A list of years_list for which data to include in the dataframe.
    :param exclude_columns: List of predictors to exclude from model prediction.
    :param predictor_csv_and_nan_pos_dir: Filepath of directory holding annual predictor csv and
                                          nan position pkl files.
    :param prediction_name_keyword: A str that will be added before prediction file name.
    :param output_dir: Filepath of output directory to store predicted rasters..
    :param ref_raster: Filepath of ref raster. Default set to WestUS reference raster.
    :param skip_processing: Set to true to skip this processing step.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        # ref raster shape
        ref_arr, ref_file = read_raster_arr_object(ref_raster)
        ref_shape = ref_arr.shape

        for year in years_list:
            print(f'\nGenerating {prediction_name_keyword} prediction raster for year: {year}...')

            # loading input variable dataframe and nan position dict
            # also filtering out excluded columns
            predictor_csv = glob(os.path.join(predictor_csv_and_nan_pos_dir, f'*{year}.csv'))[0]
            nan_pos_dict_path = glob(os.path.join(predictor_csv_and_nan_pos_dir, f'*{year}.pkl'))[0]

            df = pd.read_csv(predictor_csv)
            df = df.drop(columns=exclude_columns, errors='ignore')
            df = reindex_df(df)

            # generating prediction raster with trained model
            pred_arr = trained_model.predict(df)
            pred_arr = np.array(pred_arr)

            # replacing nan positions with -9999
            nan_pos_dict = pickle.load(open(nan_pos_dict_path, mode='rb'))
            for var_name, nan_pos in nan_pos_dict.items():
                pred_arr[nan_pos] = ref_file.nodata  # ref raster has -9999 as no data

            # reshaping the prediction raster for Western US and saving
            pred_arr = pred_arr.reshape(ref_shape)

            output_prediction_raster = os.path.join(output_dir, f'{prediction_name_keyword}_{year}.tif')
            write_array_to_raster(raster_arr=pred_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_prediction_raster)
    else:
        pass


def compute_and_clip_pumping_from_consumptive_use(consmp_gw_prediction_dir,
                                                  irr_eff_dir, westernUS_output_dir,
                                                  Western_US_ROI_shp,
                                                  skip_processing=False):
    """
    Computes annual pumping rasters by dividing consumptive groundwater use
    by irrigation efficiency rasters for each year.

    Parameters
    ----------
    consmp_gw_prediction_dir : str
        Directory containing annual consumptive groundwater use rasters.
    irr_eff_dir : str
        Directory containing annual irrigation efficiency rasters from USGS HUC12 dataset.
    westernUS_output_dir : str
        Directory to save the computed pumping rasters.
    Western_US_ROI_shp:  str
        path of model reporting ROI's shapefile consisting of 8 states.
    skip_processing : bool
        Set True to skip this step.

    :return None.
    """
    if not skip_processing:
        print("\nConverting consumptive gw use prediction to pumping raster...")

        makedirs([westernUS_output_dir])

        years = list(range(2000, 2023 + 1))

        for year in years:
            # Find the matching rasters
            cnsmp_file = glob(os.path.join(consmp_gw_prediction_dir, f'*{year}.tif'))[0]
            irr_eff_file = glob(os.path.join(irr_eff_dir, f'*{year}.tif'))[0]

            cnsmp_arr, file = read_raster_arr_object(cnsmp_file)
            irr_eff_arr = read_raster_arr_object(irr_eff_file, get_file=False)

            # Compute pumping only where both arrays have valid data
            pump_arr = np.where(
                (~np.isnan(cnsmp_arr)) & (~np.isnan(irr_eff_arr)),
                cnsmp_arr / irr_eff_arr,
                -9999
            )

            output_path = os.path.join(westernUS_output_dir, f'WestUS_pumping_{year}.tif')
            write_array_to_raster(pump_arr, file, file.transform, output_path)

            # clipping for the region of interest (ROI) - 8 states
            clip_resample_reproject_raster(input_raster=output_path, input_shape=Western_US_ROI_shp,
                                           output_raster_dir=os.path.dirname(westernUS_output_dir),
                                           keyword=' ', raster_name=f'WestUS_pumping_{year}.tif',
                                           clip_and_resample=False, clip=True, use_ref_width_height=False,
                                           resample_algorithm='near', resolution=model_res,
                                           ref_raster='../../Data_main/ref_rasters/Western_US_ROI_refraster_2km.tif')
