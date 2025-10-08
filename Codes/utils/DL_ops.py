# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from glob import glob
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import shap
import torch
import optuna
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
from Codes.utils.system_ops import makedirs

# Setting seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DataLoaderCreator:
    """
    A dataloader class to batchify input features and target value for the model.
    """

    def __init__(self, data_csv, shuffle, features_to_exclude=None,
                 batch_size=64, verbose=False):
        """
        Initialize the DataLoader to batch the data.
        """
        # reading data and separating x and y
        input_df = pd.read_csv(data_csv)
        x = input_df.drop(columns=['pixelID', 'stateID', 'year', 'target'])
        y = input_df['target'].tolist()

        # removing columns to exclude
        if features_to_exclude is None:
            pass
        else:
            columns_to_keep = [i for i in x.columns if not i in features_to_exclude]
            x = x[columns_to_keep]

        pixelID = input_df['pixelID'].tolist()
        year = input_df['year'].tolist()

        # creating numpy arrays for features, target, and pixelID
        features_np = x.to_numpy()
        target_np = np.asarray(y)
        pixel_np = np.asarray(pixelID)  # for tracking purpose
        year_np = np.asarray(year)  # for tracking purpose

        # converting to pyTorch tensor
        self.features_tensor = torch.tensor(features_np, dtype=torch.float32)
        self.target_tensor = torch.tensor(target_np, dtype=torch.float32)
        self.pixel_tensor = torch.tensor(pixel_np, dtype=torch.int32)  # for tracking purpose
        self.year_tensor = torch.tensor(year_np, dtype=torch.int32)  # for tracking purpose

        # creating a TensorDataset
        # pixelID kept in the DataLoader to keep track of associated target values
        # pixelID  isn't used during model training/validation/testing
        self.dataset = TensorDataset(self.features_tensor, self.target_tensor, self.pixel_tensor, self.year_tensor)

        # creating the DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,
                                     shuffle=shuffle)  # shuffle True is randomizing the data

        if verbose:
            # checking and printing the shapes of the tensors before batching
            print('Creating DataLoader..')
            print('-------------------------------------------------------------------')
            print('Features Tensor Shape before batching:', self.features_tensor.shape)
            print('Target Tensor Shape before batching:', self.target_tensor.shape, '\n')

            # checking and printing the shapes of the tensors after batching
            for (feature_batch, target_batch, pixel_batch, year_batch) in self.dataloader:
                print('Features Tensor Shape after batching (for each batch):', feature_batch.shape)
                print('Target Tensor Shape after batching (for each batch):', target_batch.shape)
                break

    def get_dataloader(self):
        return self.dataloader


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the model with the lowest validation loss.
    """

    def __init__(self, patience=10, delta=0.001):
        """
        Initializes the early stopping mechanism.

        :param patience: int. Number of epochs to wait before stopping if no improvement.
                         Default is 10.
        :param delta: float. Minimum change in the monitored quantity to qualify as an improvement.
                      Default is 0.001.
        """

        self.patience = patience  # Number of epochs to wait for improvement
        self.delta = delta  # Minimum change to qualify as improvement
        self.counter = 0  # Count epochs with no improvement
        self.best_loss = None  # Best loss seen so far
        self.early_stop = False  # Whether to stop training

    def __call__(self, val_loss, model):
        """
        Checks the validation loss and updates the early stopping state.

        :param val_loss: float. Validation loss for the current epoch.
        :param model: PyTorch model. The model being trained.
        """

        # if it's the first epoch or validation loss has improved significantly
        if (self.best_loss is None) or (val_loss < self.best_loss - self.delta):
            self.best_loss = val_loss  # Update best loss
            self.counter = 0  # Reset the counter

        # increment the counter for no improvement and early stops training if patience is reached
        else:
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # this triggers early stopping at the training function
                print('Early stopping triggered')


class MLPRegression(nn.Module):
    """
    A Multilayer Perceptron (MLP) model for regression tasks.
    """

    def __init__(self, n_features, fc_layers, activation_func, dropout_rate):
        """
        Initializes a MLP with specified architecture.

        :param n_features (int): Number of channelsA in the input image.
        :param fc_layers (list): Number of units in each fully connected layer.
        :param activation_func (str): Type of activation function ('relu', 'leakyrelu').
        :param dropout_rate (float): Dropout rate for fully connected layers.
        """
        # initialize the parent class (nn.Module) to properly set up the model.
        # This ensures that layers and parameters are registered, and functionality like
        # .parameters(), .state_dict(), and device management works correctly.
        super().__init__()

        # device
        self.device = 'cuda'  # running the code on GPU
        print(f'\nModel running on {self.device}....')

        # activation
        self.activations = {'relu': nn.ReLU(),
                            'leakyrelu': nn.LeakyReLU(negative_slope=0.01)}  # neg slope of leakyRelu is by default

        if activation_func not in self.activations.keys():
            print(f'Activation function "{activation_func}" is not available for implementation. \n '
                  f'Using activation function: ReLu()')
            activation_func = 'relu'

        self.activation = self.activations[activation_func]

        # Creating a placeholder for MLp layes
        self.fc_layers = nn.ModuleList()  # will hold the model components

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

        # assigning n_features as input size. input size will be updated for each layer
        input_size = n_features

        for fc_unit in fc_layers:
            self.fc_layers.append(nn.Linear(input_size, fc_unit))
            self.fc_layers.append(self.activation)
            self.fc_layers.append(self.dropout)

            # output of each hidden layer step will be input of next hidden layer
            input_size = fc_unit

        # final output layer
        # Final output size set to 1 as each input tile (variable set) will give prediction over 1 pixel
        self.output_layer = nn.Linear(input_size, 1)

        # weight initialization
        self._initialize_weights()

        # transfers the model to 'cuda' (GPU)
        self.to(self.device)

    def _initialize_weights(self):
        """
        Initializes weight for the Neural Network model. For 'relu' and 'leakyrelu', initialization method has been
        set to 'kaiming_normal' (he_normal) as a popular choice.

        resources about xavier and kaiming initialization -
        https://pouannes.github.io/blog/initialization/#:~:text=The%20only%20difference%20is%20that,find%20it%20simpler%20and%20clearer.

        """
        nonlinearity = 'relu'  # Default nonlinearity
        if isinstance(self.activation, nn.LeakyReLU):
            nonlinearity = 'leaky_relu'

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: torch.Tensor. Standardized input array representing attributes as columns and samples as row.

        :return Model output prediction.
        """
        # Passing through fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Final output layer
        x = self.output_layer(x)

        return x

    def __repr__(self):
        return f'{self.__class__.__name__} (\n' \
               f'fc_layers={self.fc_layers}, \n' \
               f'activation={self.activation})'

    def configure_optimizer(self, optimizer_name='adamw', lr=0.001, momentum=0.9, weight_decay=1e-4):
        """
          Configures and returns an optimizer for the model.

          Supported optimizers:
          - SGD (Stochastic Gradient Descent) with momentum.
          - Adam (Adaptive Moment Estimation).
          - AdamW (Adaptive Moment Estimation with decoupled Weight decay).
          - Adagrad (Adaptive Gradient Algorithm).

          :param optimizer_name:str. The name of the optimizer to use ('sgd', 'adam', 'adamw', 'adagrad').
                                Default is 'adamw'.
          :param lr : float. The learning rate for the optimizer. Default is 0.001.
          :param momentum: float. The momentum factor (only applicable for 'sgd'). Default is 0.9.
          :param weight_decay: float. Weight decay (L2 regularization) coefficient to control overfitting.
                               Default is 1e-4. A smaller value (e.g., 1e-6) penalizes large weights
                               less, while a larger value (e.g., 1e-3) penalizes them more.

          :return: torch.optim.Optimizer. The configured optimizer.

          :raises ValueError: If the specified optimizer name is not supported.
          """
        if optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        elif optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

        elif optimizer_name.lower() == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

        elif optimizer_name.lower() == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)

        else:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. Choose from 'adam', 'adamw', 'sgd', or 'adagrad'.")

    @staticmethod
    def configure_LRScheduler(optimizer, scheduler_name, epochs):
        """
        Configures learning rate scheduler.

        Details on schedulers - https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863/

        :param optimizer: Optimizer from configure_optimizer() function.
        :param scheduler_name: Scheduler name. Must be from 'CosineAnnealingLR', 'ExponentialLR'.
        :param epochs: Number of epochs.

        :return: Configured learning rate scheduler.
        """
        if scheduler_name == 'CosineAnnealingLR':
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        elif scheduler_name == 'ExponentialLR':
            return lr_scheduler.ExponentialLR(optimizer,
                                              gamma=0.95)  # (1- gamma) represents the rate of decay per epoch

        else:
            raise ValueError(
                f"Unsupported lr scheduler: {scheduler_name}, Choose from 'CosineAnnealingLR', 'ExponentialLR'")


def calculate_metrics(predictions, targets):
    """
    Calculates regression metrics: RMSE, MAE, R², Normalized RMSE, and Normalized MAE.

    :param predictions: array-like or list. Predicted values.
    :param targets: array-like or list. True target values.

    :return: dict. Dictionary containing:
        - 'RMSE': Root Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'R2': Coefficient of Determination
        - 'Normalized RMSE': RMSE divided by the mean of targets
        - 'Normalized MAE': MAE divided by the mean of targets
    """
    if isinstance(predictions, list):
        predictions = np.array(predictions)
        targets = np.array(targets)

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1 - (np.sum((predictions - targets) ** 2) /
              np.sum((targets - np.mean(targets)) ** 2))

    normalized_rmse = rmse / np.mean(targets)
    normalized_mae = mae / np.mean(targets)

    return {'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Normalized RMSE': normalized_rmse,
            'Normalized MAE': normalized_mae}


def train(model, train_loader, optimizer, verbose=False):
    """
    Trains the model for one epoch using mean squared error loss.

    :param model: torch.nn.Module. The model to train.
    :param train_loader: DataLoader. Batches of training data.
    :param optimizer: Optimizer. The optimizer to update model weights.
    :param verbose: bool. If True, print batch-level loss information.

    :return: tuple. (average_loss, RMSE, R²) for the epoch.
    """
    device = 'cuda'  # device is GPU by default

    # setting the model to training mode
    model.train()

    # initiating running_loss to accumulate the total loss across all batches
    running_loss = 0.0

    # empty lists to store predictions and actual values
    predictions, actuals = [], []

    # mse function
    mse_func = torch.nn.MSELoss()

    for batch_idx, (features, targets, _, _) in enumerate(train_loader):

        features, targets = features.to(device), targets.to(device).view(-1, 1)

        # forward pass
        preds = model(features)
        loss = mse_func(preds, targets)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulates loss for each batch
        running_loss += loss.item()

        # storing predictions and actuals
        predictions.extend(preds.cpu().detach().numpy().flatten())
        actuals.extend(targets.cpu().detach().numpy().flatten())

        # print every 10 batches
        if verbose and batch_idx % 10 == 0:
            print(f'Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}')

    # average loss for the epoch
    avg_loss = running_loss / len(train_loader)

    # calculating performance metrics
    predictions, actuals = np.array(predictions), np.array(actuals)
    metrics_dict = calculate_metrics(predictions, actuals)
    rmse = metrics_dict['RMSE']
    mae = metrics_dict['MAE']
    r2 = metrics_dict['R2']
    nrmse = metrics_dict['Normalized RMSE']
    nmae = metrics_dict['Normalized MAE']

    return avg_loss, rmse, mae, r2, nrmse, nmae


def validate(model, val_loader, verbose=False):
    """
    Validates the model for one epoch without updating weights.

    :param model: torch.nn.Module. The trained model.
    :param val_loader: DataLoader. Batches of validation data.
    :param verbose: bool. If True, print batch-level loss information.

    :return: tuple. (average_loss, RMSE, MAE, R², NRMSE, NMAE) for the validation epoch.
    """
    device = 'cuda'  # device is GPU by default

    # setting the model to evaluation mode
    # in evaluation mode, PyTorch removes all dropout layers
    model.eval()

    # initiating running_loss to accumulate the total loss across all batches
    running_loss = 0.0

    # empty lists to store predictions and actual values
    predictions, actuals = [], []

    # mse function
    mse_func = torch.nn.MSELoss()

    with torch.no_grad():  # disable gradient computation
        for batch_idx, (features, targets, _, _) in enumerate(val_loader):
            features, targets = features.to(device), targets.to(device).view(-1, 1)

            # forward pass
            preds = model(features)
            loss = mse_func(preds, targets)

            # accumulates loss for each batch
            running_loss += loss.item()

            # storing predictions and actuals
            predictions.extend(preds.cpu().detach().numpy().flatten())
            actuals.extend(targets.cpu().detach().numpy().flatten())

            # print every 10 batches
            if verbose and batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx + 1}/{len(val_loader)} - Loss: {loss.item():.4f}")

    # average loss for the epoch
    avg_loss = running_loss / len(val_loader)

    # calculating performance metrics
    predictions, actuals = np.array(predictions), np.array(actuals)
    metrics_dict = calculate_metrics(predictions, actuals)
    rmse = metrics_dict['RMSE']
    mae = metrics_dict['MAE']
    r2 = metrics_dict['R2']
    nrmse = metrics_dict['Normalized RMSE']
    nmae = metrics_dict['Normalized MAE']

    return avg_loss, rmse, mae, r2, nrmse, nmae


def test(model, test_loader, output_csv):
    """
    Evaluates the model on the test dataset and save prediction results.
    However, it can be employed for train and validation dataset as well.

    :param model: The model to validate.
    :param test_loader: DataLoader for test data. The dataset should already be standardized/normalized.
    :param output_csv: str. File path to save the test results CSV with actual and predicted values.

    :return: Average loss and additional metrics.
    """
    makedirs([os.path.dirname(output_csv)])

    # setting model to evaluation mode
    # in evaluation mode, PyTorch removes all dropout layers
    model.eval()

    # initiating running_loss to accumulate the total loss across all batches
    running_loss = 0.0

    # mse function
    mse_func = torch.nn.MSELoss()

    # empty lists to store predictions, actuals, and years for estimating metrics
    predictions, actuals = [], []
    year_list = []

    with torch.no_grad():  # disable gradient computation
        for features, targets, _, years in test_loader:
            features, targets = features.to(model.device), targets.to(model.device).view(-1, 1)
            years = years.to(model.device).view(-1, 1)

            # forward pass
            preds = model(features)

            # accumulate loss for each batch
            loss = mse_func(preds, targets)
            running_loss += loss.item()

            # collect predictions and actuals for metrics
            predictions.extend(preds.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
            year_list.extend(years.cpu().numpy().flatten())

    # average loss for the epoch
    avg_loss = running_loss / len(test_loader)

    # additional metrics (e.g., R², MAE)
    metrics_dict = calculate_metrics(predictions, actuals)

    rmse = metrics_dict['RMSE']
    mae = metrics_dict['MAE']
    r2 = metrics_dict['R2']
    nRMSE = metrics_dict['Normalized RMSE']
    nMAE = metrics_dict['Normalized MAE']

    print(f'Results -> Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, \n'
          f'NRMSE: {nRMSE:4f}, NMAE:{nMAE:.4f}, R²: {r2:.4f}\n')

    # creating an output dataframe with tile_no, actual values, and model predictions
    output_df = pd.DataFrame({'year': year_list, 'actual': actuals, 'predicted': predictions})
    output_df.to_csv(output_csv, index=False)

    return avg_loss, rmse, mae, r2


def run_default_model(train_loader, val_loader,
                      n_features, fc_units, n_epochs,
                      lr_scheduler, lr=1e-3,
                      activation_func='leakyrelu',
                      weight_decay=1e-2, dropout_rate=0.3,
                      implement_earlyStopping=False,
                      patience=10, start_EarlyStop_count_from_epoch=20,
                      verbose=True):
    """
    Trains an MLP regression model with specified default hyperparameters and returns model performance.

    :param train_loader: DataLoader. Batches of training data.
    :param val_loader: DataLoader. Batches of validation data.
    :param n_features: int. Number of input features.
    :param fc_units: list. List of integers specifying the number of neurons in each fully connected layer.
    :param n_epochs: int. Number of training epochs.
    :param lr_scheduler: str. Learning rate scheduler to use ('CosineAnnealingLR' or 'ExponentialLR').
    :param lr: float. Initial learning rate.
    :param activation_func: str. Activation function ('relu' or 'leakyrelu').
    :param weight_decay: float. L2 regularization strength.
    :param dropout_rate: float. Dropout probability (between 0 and 1).
    :param implement_earlyStopping: bool. Whether to use early stopping.
    :param patience: int. Patience for early stopping.
    :param start_EarlyStop_count_from_epoch: int. Epoch to start checking early stopping.
    :param verbose: bool. If True, prints training progress.

    :return: tuple. (trained_model, model_state_dict, model_info_dict)
        - trained_model: the trained PyTorch model
        - model_state_dict: `state_dict()` of the trained model
        - model_info_dict: dictionary with hyperparameters and loss history
    """

    train_loss, val_loss, train_rmse, train_r2, val_rmse, val_r2 = None, None, None, None, None, None

    # initializing the model
    model = MLPRegression(
        n_features=n_features,
        activation_func=activation_func,
        fc_layers=fc_units,
        dropout_rate=dropout_rate
    )

    # configuring optimizer
    optimizer = model.configure_optimizer(optimizer_name='adamw', lr=lr, weight_decay=weight_decay)

    # configuring lr scheduler
    scheduler = model.configure_LRScheduler(optimizer, scheduler_name=lr_scheduler, epochs=n_epochs)

    # initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience)

    # empty dictionary and lists to track parameters and losses
    model_info = {'params':
                      {'lr': [], 'n_features': None, 'fc_units': None,
                       'activation_func': None, 'dropout_rate': None, 'weight_decay': None},
                  'train_losses': [], 'val_losses': [], 'val_loss': None}

    train_losses = []
    val_losses = []
    last_epoch = None  # Track the last trained epoch

    for epoch in range(n_epochs):

        epoch = epoch + 1  # making epoch starting from 1

        # training and validation for one epoch
        train_loss, train_rmse, train_mae, train_r2, train_nrmse, train_nmae = \
            train(model, train_loader, optimizer)

        val_loss, val_rmse, val_mae, val_r2, val_nrmse, val_nmae = \
            validate(model, val_loader)

        # storing losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # storing lr before updating
        # updating lr using scheduler
        model_info['params']['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        # storing the last trained epoch
        last_epoch = epoch

        # printing progress
        if verbose and epoch % 10 == 0:
            print(
                f'Train Epoch: {epoch} | Loss: {train_loss:.3f} | RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f} | NRMSE: {train_nrmse:.3f} | NMAE: {train_nmae:.3f}')
            print(
                f'Val   Epoch: {epoch} | Loss: {val_loss:.3f}   | RMSE: {val_rmse:.3f}   | MAE: {val_mae:.3f}   | R²: {val_r2:.3f}   | NRMSE: {val_nrmse:.3f}   | NMAE: {val_nmae:.3f}')
            print('---------------------------------------------------------------------')

        # checking for early stopping
        if epoch >= start_EarlyStop_count_from_epoch and implement_earlyStopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    # printing final performance (last trained epoch)
    print(
        f'Final Train Epoch: {epoch} | Loss: {train_loss:.3f} | RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f} | NRMSE: {train_nrmse:.3f} | NMAE: {train_nmae:.3f}')
    print(
        f'Final Val   Epoch: {epoch} | Loss: {val_loss:.3f}   | RMSE: {val_rmse:.3f}   | MAE: {val_mae:.3f}   | R²: {val_r2:.3f}   | NRMSE: {val_nrmse:.3f}   | NMAE: {val_nmae:.3f}')
    print('---------------------------------------------------------------------')

    print('\n*** NOTE: MSE (loss) is averaged per batch, while RMSE/MAE/NRMSE/NMAE/R² are calculated on the entire dataset ***')

    # handling val_loss when early stopping is disabled
    if not implement_earlyStopping:
        best_val_loss = min(val_losses)

    else:
        best_val_loss = early_stopping.best_loss

    # saving model state_dict, hyperparamters and architecture information and losses
    model_state = model.state_dict()

    model_info['params'] = {
        'n_features': n_features,
        'fc_units': fc_units,
        'activation_func': activation_func,
        'dropout_rate': dropout_rate,
        'weight_decay': weight_decay
    }
    model_info['train_losses'] = train_losses
    model_info['val_losses'] = val_losses
    model_info['val_loss'] = best_val_loss

    return model, model_state, model_info


def run_and_tune_model(trial, train_loader, val_loader,
                       n_features, n_epochs,
                       lr, lr_scheduler,
                       activation_func='leakyrelu',
                       implement_earlyStopping=False,
                       patience=10, start_EarlyStop_count_from_epoch=40):
    """
    Objective function used by Optuna to sample, train, and evaluate an MLP model configuration.

    :param trial: optuna.trial.Trial. Trial object used to sample hyperparameters.
    :param train_loader: DataLoader. DataLoader containing the training data.
    :param val_loader: DataLoader. DataLoader containing the validation data.
    :param n_features: int. Number of input features to the model.
    :param n_epochs: int. Number of epochs to train the model.
    :param lr: float. Learning rate for the optimizer.
    :param lr_scheduler: str. Name of the learning rate scheduler ('CosineAnnealingLR' or 'ExponentialLR').
    :param activation_func: str. Activation function to use ('relu' or 'leakyrelu').
    :param implement_earlyStopping: bool. Whether to use early stopping during training.
    :param patience: int. Number of epochs to wait for improvement before triggering early stopping.
    :param start_EarlyStop_count_from_epoch: int. Epoch after which early stopping begins to check for improvements.

    :return: float. Objective value for the trial (validation loss + penalty term).
    """
    print(f'\nStarting trial number {trial.number}...\n')

    # sample hyperparameters
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout', 0.2, 0.5, step=0.05)

    # sample fully connected layer configuration
    # ensure each subsequent layer has fewer units than the previous layer
    num_fc_layers = trial.suggest_int('num_fc_layers', 3, 5)  # Between 2 and 5 layers

    fc_units = []
    fc_units.append(trial.suggest_int('fc_units_layer_0', 128, 256, step=32))  # First layer: largest

    for i in range(1, num_fc_layers):
        min_val = 32  # Minimum neurons per layer
        high_val = max(64, fc_units[i - 1] // 2)  # Ensure gradual reduction

        fc_units.append(trial.suggest_int(f'fc_units_layer_{i}', min_val, high_val, step=32))

    # training the model with the sampled parameters
    _, model_state, model_info = \
        run_default_model(train_loader, val_loader,
                          n_features=n_features, fc_units=fc_units, n_epochs=n_epochs,
                          lr=lr, lr_scheduler=lr_scheduler,
                          activation_func=activation_func,
                          weight_decay=weight_decay, dropout_rate=dropout_rate,
                          implement_earlyStopping=implement_earlyStopping,
                          patience=patience, start_EarlyStop_count_from_epoch=start_EarlyStop_count_from_epoch,
                          verbose=True)

    # objective function
    best_val_loss = model_info['val_loss']  # for a specific trial
    train_loss_at_best_val = model_info['train_losses'][model_info['val_losses'].index(best_val_loss)]

    alpha = 0.3  # Weight for the penalty term to minimize the gap between train and validation loss
    objective_value = best_val_loss + alpha * abs(
        train_loss_at_best_val - best_val_loss)  # param tuning will minimize this value

    # storing additional information for later use
    best_epoch = model_info['val_losses'].index(best_val_loss)  # best epoch
    trial.set_user_attr('model_info', model_info)  # saving model info for the trial
    trial.set_user_attr('best_epoch', best_epoch)  # saving best epoch for the trial

    return objective_value


def save_param_importance_plot(study, save_path):
    """
    Generates and saves the parameter importance plot for an Optuna study.

    :param study: The Optuna study object after optimization.
    :param save_path: The file path to save the parameter importance plot.

    :return None.
    """
    # extracting parameter importance values
    importance = optuna.importance.get_param_importances(study)
    params = list(importance.keys())
    scores = list(importance.values())

    # creating a bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(params, scores, color='skyblue')
    plt.xlabel('Importance Score')
    plt.ylabel('Parameter')
    plt.title('Parameter Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(save_path, format='png', dpi=300)
    plt.close()

    print(f'\nParameter importance plot saved from parameter tuning process')


def main(train_data_csv, val_data_csv,
         features_to_exclude, batch_size,
         n_features, n_epochs, lr, lr_scheduler,
         model_save_path, model_info_save_path,
         activation_func='relu',
         default_params=None,
         implement_earlyStopping=False,
         patience=10, start_EarlyStop_count_from_epoch=40,
         tune_parameters=False, n_trials=50,
         plot_hyperparams_importance=False,
         hyperparam_importance_plot_path=None
         ):
    """
    Entry point to run MLP model training using either default parameters or Optuna-based hyperparameter tuning.

    :param train_data_csv: str. Path to the training CSV file.
    :param val_data_csv: str. Path to the validation CSV file.
    :param features_to_exclude: list. List of feature names to exclude.
    :param batch_size: int. Batch size for DataLoaders.
    :param n_features: int. Number of features used by the model.
    :param n_epochs: int. Number of training epochs.
    :param lr: float. Initial learning rate.
    :param lr_scheduler: str. Name of learning rate scheduler ('CosineAnnealingLR', 'ExponentialLR').
    :param model_save_path: str. File path to save the model state dictionary.
    :param model_info_save_path: str. File path to save the model metadata dictionary.
    :param activation_func: str. Activation function to use ('relu' or 'leakyrelu').
    :param default_params: dict. Dictionary with default hyperparameters (used if `tune_parameters=False`).
    :param implement_earlyStopping: bool. Whether to use early stopping.
    :param patience: int. Patience for early stopping.
    :param start_EarlyStop_count_from_epoch: int. Epoch to start early stopping.
    :param tune_parameters: bool. If True, run Optuna hyperparameter tuning.
    :param n_trials: int. Number of Optuna trials (if tuning).
    :param plot_hyperparams_importance: bool. Whether to plot hyperparameter importance (Optuna).
    :param hyperparam_importance_plot_path: str. File path to save the plot (if enabled).

    :return: tuple. (trained_model, model_info_dict)
    """
    # creating storage directory
    makedirs([os.path.dirname(model_info_save_path)])

    # creating train and validation DataLoaders
    train_loader = DataLoaderCreator(data_csv=train_data_csv, shuffle=True,
                                     features_to_exclude=features_to_exclude,
                                     batch_size=batch_size, verbose=True).get_dataloader()

    val_loader = DataLoaderCreator(data_csv=val_data_csv, shuffle=False,
                                   features_to_exclude=features_to_exclude,
                                   batch_size=batch_size, verbose=True).get_dataloader()

    # # parameter tuning mode
    if tune_parameters:
        # checking conditions for default parameters and tuning mode
        if default_params is not None:
            print('\n`default_params` is ignored when `tune_parameters=True`.')

        # performing hyperparameter + configuration parameters tuning using Optuna
        print('Starting parameter tuning with Optuna...')

        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: run_and_tune_model(
            trial=trial, train_loader=train_loader, val_loader=val_loader,
            n_features=n_features, n_epochs=n_epochs,
            lr=lr, lr_scheduler=lr_scheduler,
            activation_func=activation_func,
            implement_earlyStopping=implement_earlyStopping,
            patience=patience, start_EarlyStop_count_from_epoch=start_EarlyStop_count_from_epoch), n_trials=n_trials)

        # best parameters achieved from hyperparameter training
        best_params = study.best_trial.params
        best_epoch = study.best_trial.user_attrs["best_epoch"]

        # retraining the best model (optional)
        print('\nRetraining the best model...')
        trained_model, best_model_state, best_model_info = \
            run_default_model(train_loader=train_loader,
                              val_loader=val_loader,
                              n_features=n_features,
                              fc_units=[best_params[f'fc_units_layer_{i}'] for i in
                                        range(best_params['num_fc_layers'])],
                              n_epochs=best_epoch,
                              lr=lr, lr_scheduler=lr_scheduler,
                              activation_func=activation_func,
                              weight_decay=best_params['weight_decay'],
                              dropout_rate=best_params['dropout'],
                              implement_earlyStopping=implement_earlyStopping,
                              patience=patience,
                              start_EarlyStop_count_from_epoch=start_EarlyStop_count_from_epoch,
                              verbose=True)

        # save the best model's state_dict and  information dictionary
        torch.save(best_model_state, model_save_path)

        with open(model_info_save_path, 'wb') as f:
            pickle.dump(best_model_info, f)

        print(f'\nFinal model saved at {model_save_path}')

        # printing model best parameters' summary
        print('\nBest parameters found:')
        print(best_params)
        print(f'\nBest epoch - {best_epoch}')
        print(f"\nBest val loss - {best_model_info['val_loss']}")

        # plotting hyperparameter importance plot from tuning process
        if plot_hyperparams_importance and hyperparam_importance_plot_path is not None:
            save_param_importance_plot(study, save_path=hyperparam_importance_plot_path)

        return trained_model, best_model_info

    # # default parameter mode
    else:
        # check if default_params is provided for non-tuning mode
        if default_params is None:
            raise ValueError('`default_params` must be provided when `tune_parameters=False`.')

        print('\nRunning the model with parameters:')
        print(f"'n_features': {n_features}, 'batch size': {batch_size},\n"
              f"'initial learning rate': {lr}, 'lr scheduler': {lr_scheduler},\n"
              f"'activation_func': {activation_func}")
        print(', '.join(f"'{i}': {j}" for i, j in default_params.items()))

        # running the model with default hyperparameters and configuration parameters
        trained_model, model_state, model_info = \
            run_default_model(train_loader=train_loader,
                              val_loader=val_loader,
                              n_features=n_features,
                              fc_units=default_params['fc_units'],
                              n_epochs=n_epochs,
                              lr=lr, lr_scheduler=lr_scheduler,
                              activation_func=activation_func,
                              weight_decay=default_params['weight_decay'],
                              dropout_rate=default_params['dropout'],
                              implement_earlyStopping=implement_earlyStopping,
                              patience=patience,
                              start_EarlyStop_count_from_epoch=start_EarlyStop_count_from_epoch,
                              verbose=True)

        # saving the model state and information
        torch.save(model_state, model_save_path)

        with open(model_info_save_path, 'wb') as f:
            pickle.dump(model_info, f)

        print(f'\nModel saved at {model_save_path}')
        print(f"\nEpochs ran - {len(model_info['val_losses'])}")
        print(f"\nLast val loss (MSE) - {model_info['val_losses'][-1]:.3f}")

        return trained_model, model_info


def plot_learning_curve(train_loss, val_loss, plot_save_path):
    """
    Plots and saves the learning curve (loss vs. epoch).

    :param train_loss: list or array-like. Training loss values per epoch.
    :param val_loss: list or array-like. Validation loss values per epoch.
    :param plot_save_path: str. File path to save the PNG plot.

    :return: None. The function saves the plot to disk.
    """
    # plotting losses
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (mean squared error)', fontsize=14)

    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # saving the plot as an image file
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=200)
    plt.close()
    print(f'\nLoss plot saved...')


def calc_rangeWise_RMSE(results_csv, value_ranges, output_txt):
    """
    Computes and logs RMSE and MAE for prediction errors across specified value ranges.

    This function reads a CSV file containing actual and predicted values, divides the data
    into defined value ranges based on the 'actual' column, computes RMSE and MAE for each
    range, and writes the results to a text file. The percentage of test data in each range
    is also reported.

    :param results_csv: str. Path to the CSV file containing 'actual' and 'predicted' columns.
    :param value_ranges: list. A list of numeric thresholds defining intervals to group the 'actual' values.
                         Example: [0, 100, 200, 400] will create ranges [0–100], [100–200], [200–400].
    :param output_txt: str. Path to the output text file where results will be saved. If the file exists, it will be overwritten.

    :return: None. Results are written to the specified text file.
    """

    # remove previously saved output_txt
    if os.path.exists(output_txt):
        os.remove(output_txt)

    # loading results csv with actual values and prediction
    data_df = pd.read_csv(results_csv)

    # calculating results and writing to the text file
    with open(output_txt, 'w') as f:
        for i in range(len(value_ranges) - 1):
            sel_df = data_df[(data_df['actual'] >= value_ranges[i]) & (data_df['actual'] <= value_ranges[i + 1])]
            perc_of_test_data = len(sel_df) * 100 / len(data_df)
            actual_arr = sel_df['actual']
            pred_arr = sel_df['predicted']
            scores = calculate_metrics(predictions=pred_arr, targets=actual_arr)
            rmse, mae, nrmse, nmae = scores['RMSE'], scores['MAE'], scores['Normalized RMSE'], scores['Normalized MAE']

            results_str = (
                f'value ranges [{value_ranges[i]}-{value_ranges[i + 1]}]- % of test data {perc_of_test_data:.2f}%\
                    RMSE:{rmse:.2f}, MAE:{mae:.2f}, NRMSE:{nrmse:.2f}, NMAE:{nmae:.2f}')

            f.write(results_str + '\n')


def plot_shap_summary_plot(trained_model_path, trained_model_info, use_samples,
                           data_csv, exclude_features, save_plot_path,
                           skip_processing=False):
    """
    Generate and save a SHAP summary (beeswarm) plot to visualize feature importance.

    :param trained_model_path: str
        Path to the `.pth` file containing the saved model's state_dict.

    :param trained_model_info: str
        Path to the `.pkl` file with model configuration and metadata.

    :param use_samples: int
        Number of samples to randomly draw from the dataset for SHAP analysis.

    :param data_csv: str
        Path to the CSV file containing the input feature data.

    :param exclude_features: list
        List of column names to exclude from the input (e.g., target, IDs).

    :param save_plot_path: str
        File path (with extension) where the SHAP summary plot will be saved.

    :param skip_processing: bool, optional (default=False)
        If True, skip execution.

    :return: None
    """
    if not skip_processing:
        makedirs([os.path.dirname(save_plot_path)])

        # loading model
        trained_model_info = pickle.load(open(trained_model_info, 'rb'))
        trained_model = MLPRegression(
            n_features=trained_model_info['params']['n_features'],
            fc_layers=trained_model_info['params']['fc_units'],
            activation_func=trained_model_info['params']['activation_func'],
            dropout_rate=trained_model_info['params']['dropout_rate']
        )

        # loading state_dict of the trained model
        trained_model.load_state_dict(torch.load(f=trained_model_path, weights_only=True))

        print(trained_model)

        # model set to evaluation mode
        trained_model = trained_model.to('cuda')
        trained_model.eval()

        print('\n___________________________________________________________________________')
        print(f'\nplotting SHAP feature importance...')

        # loading data + random sampling + renaming dataframe features
        if 'target' not in exclude_features:
            exclude_features = exclude_features + ['target']

        df = pd.read_csv(data_csv)
        df = df.drop(columns=exclude_features)
        df = df.sample(n=use_samples, random_state=43)  # sampling 'use_samples' of rows for SHAP plotting

        feature_names_dict = {'netGW_Irr': 'Consumptive groundwater use', 'peff': 'Effective precipitation',
                              'SW_Irr': 'Surface water irrigation', 'ret': 'Reference ET', 'precip': 'Precipitation',
                              'tmax': 'Temperature (max)', 'ET': 'ET', 'irr_crop_frac': 'Irrigated crop fraction',
                              'maxRH': 'Relative humidity (max)', 'minRH': 'Relative humidity (min)',
                              'shortRad': 'Shortwave radiation', 'vpd': 'Vapor pressure deficit',
                              'sunHr': 'Sun hour', 'FC': 'Field capacity',
                              'Canal_distance': 'Distance from canal', 'Canal_density': 'Canal density'}
        df = df.rename(columns=feature_names_dict)
        feature_names = np.array(df.columns.tolist())

        # converting to numpy to convert to torch tensor
        data_tensor = torch.tensor(df.values, dtype=torch.float32).to('cuda')

        # using SHAP GradientExplainer designed for PyTorch/TensorFlow (DeepExplainer doesn't work for some reasons)
        explainer = shap.GradientExplainer(trained_model, data_tensor)
        shap_values = explainer(data_tensor)

        # converting SHAP values to numpy for plotting
        shap_values_np = shap_values.values.squeeze(
            -1)  # Remove singleton third dimension from SHAP array: shape [n, m, 1] → [n, m]
        data_np = data_tensor.cpu().numpy()

        # plotting
        fig = plt.figure()
        shap.summary_plot(shap_values_np, data_np, feature_names=feature_names)

        fig.savefig(save_plot_path, dpi=200, bbox_inches='tight')

    else:
        pass


def plot_shap_interaction_plot(model_version, features_to_plot,
                               trained_model_path, trained_model_info,
                               feature_excluded_in_training, use_samples,
                               data_csv, save_plot_dir,
                               skip_processing=False):
    """
    Generate individual SHAP dependence plots for selected features and compile them into a grid image.

    :param model_version: str
        Model version. Used to save and track corresponding SHAP plots for respective model.

    :param features_to_plot: list
        Human-readable feature names to generate SHAP dependence plots for.
        Must match the renamed columns after feature mapping.

        Select from this list-
         ['Consumptive groundwater use', 'Effective precipitation',
         'Surface water irrigation', 'Reference ET', 'Precipitation',
         'Temperature (max)', 'ET', 'Irrigated crop fraction',
         'Relative humidity (max)', 'Relative humidity (min)',
         'Shortwave radiation', 'Vapor pressure deficit',
         'Sun hour', 'Field capacity']

    :param trained_model_path: str
        Path to the `.pth` file containing the saved model's state_dict.

    :param trained_model_info: str
        Path to the `.pkl` file with model configuration and metadata.

    :param feature_excluded_in_training: List of features excluded in trainined model.

    :param use_samples: int
        Number of samples to randomly draw from the dataset for SHAP analysis.

    :param data_csv: str
        Path to the CSV file containing input features.

    :param save_plot_dir: str
        Directory where individual dependence plots and the compiled grid image will be saved.

    :param skip_processing: bool, optional (default=False)
        If True, skip execution.

    :return: None
    """
    if not skip_processing:
        makedirs([save_plot_dir])

        # loading model
        trained_model_info = pickle.load(open(trained_model_info, 'rb'))
        trained_model = MLPRegression(
            n_features=trained_model_info['params']['n_features'],
            fc_layers=trained_model_info['params']['fc_units'],
            activation_func=trained_model_info['params']['activation_func'],
            dropout_rate=trained_model_info['params']['dropout_rate']
        )

        # loading state_dict of the trained model
        trained_model.load_state_dict(torch.load(f=trained_model_path, weights_only=True))

        print(trained_model)

        # model set to evaluation mode
        trained_model = trained_model.to('cuda')
        trained_model.eval()

        print('\n___________________________________________________________________________')
        print(f'\nplotting SHAP feature importance...')

        # loading data + random sampling + renaming dataframe features

        if 'target' not in feature_excluded_in_training:
            feature_excluded_in_training = feature_excluded_in_training + ['target']

        df = pd.read_csv(data_csv)
        df = df.drop(columns=feature_excluded_in_training)
        df = df.sample(n=use_samples, random_state=43)  # sampling 'use_samples' of rows for SHAP plotting

        feature_names_dict = {'netGW_Irr': 'Consumptive groundwater use', 'peff': 'Effective precipitation',
                              'SW_Irr': 'Surface water irrigation', 'ret': 'Reference ET', 'precip': 'Precipitation',
                              'tmax': 'Temperature (max)', 'ET': 'ET', 'irr_crop_frac': 'Irrigated crop fraction',
                              'maxRH': 'Relative humidity (max)', 'minRH': 'Relative humidity (min)',
                              'shortRad': 'Shortwave radiation', 'vpd': 'Vapor pressure deficit',
                              'sunHr': 'Sun hour', 'FC': 'Field capacity',
                              'Canal_distance': 'Distance from canal',
                              'Canal_density': 'Canal density'}

        df = df.rename(columns=feature_names_dict)
        feature_names = np.array(df.columns.tolist())

        # converting to numpy to convert to torch tensor
        data_tensor = torch.tensor(df.values, dtype=torch.float32).to('cuda')

        # using SHAP GradientExplainer designed for PyTorch/TensorFlow (DeepExplainer doesn't work for some reasons)
        explainer = shap.GradientExplainer(trained_model, data_tensor)
        shap_values = explainer(data_tensor)

        # converting SHAP values to numpy for plotting
        shap_values_np = shap_values.values.squeeze(
            -1)  # Remove singleton third dimension from SHAP array: shape [n, m, 1] → [n, m]
        data_np = data_tensor.cpu().numpy()

        # plotting and saving individual shap dependence plots
        for feature in features_to_plot:
            shap.dependence_plot(feature, shap_values_np, data_np, feature_names=feature_names,
                                 interaction_index=None, show=False)
            plt.gca().set_ylabel('SHAP value')
            plt.gca().set_xlabel(feature)

            plt.savefig(os.path.join(save_plot_dir, f'{feature}.png'), dpi=200, bbox_inches='tight')

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


def write_array_to_raster(raster_arr, raster_file, transform, output_path, dtype=None,
                          ref_file=None, nodata=-9999):
    """
    Write raster array to Geotiff format.

    :param raster_arr: Raster array data to be written.
    :param raster_file: Original rasterio raster file containing geo-coordinates.
    :param transform: Affine transformation matrix.
    :param output_path: Output filepath.
    :param dtype: Output raster data type. Default set to None.
    :param ref_file: Write output raster considering parameters from reference raster file.
    :param nodata: no_data_value set as -9999.

    :return: Output filepath.
    """
    makedirs([os.path.dirname(output_path)])

    if dtype is None:
        dtype = raster_arr.dtype

    if ref_file:
        raster_file = rio.open(ref_file)
        transform = raster_file.transform

    with rio.open(
            output_path,
            'w',
            driver='GTiff',
            height=raster_arr.shape[0],
            width=raster_arr.shape[1],
            dtype=dtype,
            count=raster_file.count,
            crs=raster_file.crs,
            transform=transform,
            nodata=nodata
    ) as dst:
        dst.write(raster_arr, raster_file.count)

    return output_path


def load_model_and_predict_raster(trained_model_path, trained_model_info, years_list,
                                  predictor_csv_dir, nan_pos_dir, batch_size, exclude_features_in_model,
                                  output_dir, ref_raster,
                                  verbose=False,
                                  skip_processing=False):
    """
    Loads a trained MLP model and make prediction raster.

    :param trained_model_path: str. Path to the `.pth` file containing the saved model's state_dict.
    :param trained_model_info: str. Path to the `.pkl` file containing model configuration and metadata.
    :param years_list: list. A list of years_list for which data to include in the dataframe.
    :param predictor_csv_dir: Directory path holding annual predictor csv (standardized).
    :param nan_pos_dir: str. Directory path holding nan position .pkl files.
    :param batch_size: int. Batch size to use for the DataLoader during prediction.
    :param exclude_features_in_model: list. List of feature column names to exclude from the input data.
    :param output_dir: str. Path to the output directory where predicted rasters will be saved.
    :param ref_raster: Western US reference raster for reshaping prediction raster's shape.
    :param verbose: Set to True to print the loaded trained model.
    :param skip_processing: bool. Set to True to skip this code run.

    :return: None. The function saves the prediction results as a CSV file to the specified path.
    """
    if not skip_processing:
        makedirs([output_dir])

        # # #configuring model structure (using the params from the trained model)
        trained_model_info = pickle.load(open(trained_model_info, 'rb'))
        trained_model = MLPRegression(
            n_features=trained_model_info['params']['n_features'],
            fc_layers=trained_model_info['params']['fc_units'],
            activation_func=trained_model_info['params']['activation_func'],
            dropout_rate=trained_model_info['params']['dropout_rate']
        )

        # loading state_dict of the trained model
        trained_model.load_state_dict(torch.load(trained_model_path, weights_only=True))
        trained_model.to('cuda')

        if verbose:
            print(trained_model)

        # model set to evaluation mode
        trained_model.eval()

        for year in years_list:
            print(f'\nGenerating prediction raster for year: {year}...')

            # # # selecting csv for the year
            predictor_csv = glob(os.path.join(predictor_csv_dir, f'*{year}.csv'))[0]

            # # # configuring DataLoader
            predictor_df = pd.read_csv(predictor_csv)

            # removing columns that are not needed
            if exclude_features_in_model is None:
                pass
            else:
                columns_to_keep = [i for i in predictor_df.columns if not i in exclude_features_in_model]
                predictor_df = predictor_df[columns_to_keep]

            # converting to torch.tensor
            features_np = predictor_df.to_numpy()
            features_tensor = torch.tensor(features_np, dtype=torch.float32)

            # TensorDataset > DataLoader
            # shuffle must be False to keep the serial of the dataframe intact
            tensordataset = TensorDataset(features_tensor)

            dataloader = DataLoader(tensordataset, batch_size=batch_size, shuffle=False)

            # # # predicting with trained_model

            predictions = []  # empty list generated for each year to store that year's prediction

            with torch.no_grad():  # disable gradient computation
                for (features,) in dataloader:  # dataLoader returns a tuple
                    features = features.to(trained_model.device)  # transferring data to 'cuda'

                    preds = trained_model(features)

                    predictions.extend(preds.cpu().numpy().flatten())

            # # # converting prediction to numpy array
            pred_arr = np.array(predictions)

            # # # replacing nan positions with -9999
            nan_pos_dict_path = glob(os.path.join(nan_pos_dir, f'*{year}.pkl'))[0]
            nan_pos_dict = pickle.load(open(nan_pos_dict_path, mode='rb'))

            for var_name, nan_pos in nan_pos_dict.items():
                pred_arr[nan_pos] = -9999

            # # # reshaping prediction with reference raster
            ref_file = rio.open(ref_raster)
            ref_arr = ref_file.read(1)
            pred_arr = pred_arr.reshape(ref_arr.shape)

            # noticed some -ve pumping pixels near oregon, replcing them with zero
            pred_arr = np.where(~np.isnan(pred_arr) & (pred_arr < 0), 0, pred_arr)

            # # # saving
            output_prediction_raster = os.path.join(output_dir, f'pumping_{year}.tif')
            write_array_to_raster(raster_arr=pred_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_prediction_raster)
    else:
        pass
