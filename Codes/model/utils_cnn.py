# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

"""
Acknowledgment:
This script was developed by the author based on his knowledge and experience on machine/deep learning models.
Some assistance and insights have been taken from  ChatGPT, an AI model by OpenAI, to improve  efficiency, accuracy,
and readability of the script, considering the complex nature of this script.
"""

import os
import sys
import shap
import random
import pickle
import numpy as np
import pandas as pd
from glob import glob
import rasterio as rio
import matplotlib.pyplot as plt

import torch
import optuna
import torch.nn as nn
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
    A dataloader class to batchify tiles of input features and a single target value per tile for the model.
    """

    def __init__(self, tile_dir, target_csv, batch_size=64, data_type='train', verbose=False):
        """
        Initialize the DataLoader to batch the data.

        :param tile_dir (str): Directory containing .tif files for tiles.
        :param target_csv (csv): Target csv. Must have a tile_no and value columns.
                                 The tile_no column represents the corresponding tile no and value
                                 represents the value to train/validate/test on.
        :param batch_size (int): Batch size for the DataLoader.
        :param data_type (str): Type of data (train/validation/test) passed to the DataLoader class.
        :param verbose (bool): Set to True if want to print before and after batching tensor size. 
        """
        if data_type in ['train', 'validation', 'test']:
            print(f'Initializing DataLoader to batch the {data_type} data...\n')
        else:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'train', 'validation', or 'test'.")


        # reading target and input datasets:
        # we have to make sure that target values and tiles are matching (tiles have corresponding target values).
        # so, creating a dictionary for mapping tile_no (from the file name) to its full file path for quick lookup.
        # then, sorting tiles to match the order of tile_no_list from the target CSV.
        # finally, reading the sorted tiles as arrays
        target_df = pd.read_csv(target_csv)
        target_values = target_df['standardized_value'].tolist()
        tile_no_list = target_df['tile_no'].tolist()

        tiles = glob(os.path.join(tile_dir, '*.tif'))
        tile_dict = {os.path.basename(tile).split('_')[-1].replace('.tif', ''): tile for tile in tiles}
        tiles_sorted = [tile_dict[str(tile_no)] for tile_no in tile_no_list if str(tile_no) in tile_dict]

        features_arrs = [rio.open(tt).read() for tt in tiles_sorted]  # storing multi-band features as array

        # creating numpy arrays for features, target, and tile_no
        features_np = np.stack(features_arrs)  # dimensions are - num of image * num features * height * width
        target_np = np.asarray(target_values)
        tileNo_np = np.asarray(tile_no_list)  # for tracking purpose

        # converting to pyTorch tensor
        self.features_tensor = torch.tensor(features_np, dtype=torch.float32)
        self.target_tensor = torch.tensor(target_np, dtype=torch.float32)
        self.tileNo_tensor = torch.tensor(tileNo_np, dtype=torch.int32)  # for tracking purpose

        # creating a TensorDataset
        # tileNo kept in the DataLoader to keep track of tileNo for associated target values
        # tileNo isn't used during initial model training/validation/testing
        # tileNo only employed after model run has completed and to test the unstandardized value's performance
        self.dataset = TensorDataset(self.features_tensor, self.target_tensor, self.tileNo_tensor)

        # creating the DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,
                                     shuffle=True)  # shuffle True is randomizing the the data

        if verbose:
            # checking and printing the shapes of the tensors before batching
            print('Features Tensor Shape before batching:', self.features_tensor.shape)
            print('Target Tensor Shape before batching:', self.target_tensor.shape, '\n')

            # checking and printing the shapes of the tensors after batching
            for (feature_batch, target_batch, tileNo_batch) in self.dataloader:
                print('Features Tensor Shape after batching (for each batch):', feature_batch.shape)
                print('Target Tensor Shape after batching (for each batch):', target_batch.shape)
                print('Tile No Tensor Shape after batching (for each batch):', tileNo_batch.shape, '\n')
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


class CNNRegression(nn.Module):
    """
    A CNN model for regression tasks with a fully connected layer at the end.
    """

    def __init__(self, n_features, input_size, filters, kernels, stride=1, padding='valid',
                 pooling='maxpool', activation_func='relu', fc_layers=None, dropout_rate=0.3):
        """
        Initializes a CNN with specified architecture.

        :param n_features (int): Number of channelsA in the input image.
        :param input_size (int): Height/width of the square input image (e.g., 64 for 64x64 images).
        :param filters (list): Number of filters in each convolutional layer.
        :param kernels (list): Kernel size for each convolutional layer.
        :param stride (int): Stride for convolutional layers. Defaults to 1.
        :param padding (str or int): Padding type ('same' or 'valid'). 'valid' resembles to 0 padding.
                             'same' padding pads the input so the output has the shape as the input.
                             Can also be integer values like 1 or 2.
        :param pooling (str): Pooling option ('maxpool', 'avgpool').
        :param activation_func (str): Type of activation function ('relu', 'leakyrelu').
        :param fc_layers (list): Number of units in each fully connected layer. Defaults to [128] if set to None.
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

        # pooling
        # the default MaxPool2d() and AvgPool2d() uses stride 2 with kernel size of 2*2
        self.poolings = {'maxpool': nn.MaxPool2d(2),
                         'avgpool': nn.AvgPool2d(2)}
        pooling = self.poolings.get(pooling, nn.MaxPool2d(2))  # by default MaxPool2d if not specified

        # stride
        stride = 1 if padding == 'same' else stride

        # # # CNN layers
        # CNN steps: conv -> activation -> pooling
        self.conv_layers = nn.ModuleList()  # will hold the model components

        in_channels = n_features

        for out_channels, kernel_size in zip(filters, kernels):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.conv_layers.append(self.activation)
            self.conv_layers.append(pooling)  # pooling (downsampling)

            in_channels = out_channels  # at the end of each block, the in_channels of the next block is set as equal to the out_channel of the previous block

        # calculating the flattened size after all operations from convolutional layers
        flattened_size = self._calculate_flattened_size(input_size, filters, kernels, stride,
                                                        padding, pooling_stride=2)

        # # # Fully connected layers
        # CNN steps: conv -> activation -> pooling
        if fc_layers is None:
            fc_layers = [128]  # default FC structure if fc_layers=None

        self.fc_layers = nn.ModuleList()  # will hold the model components

        input_size = flattened_size  # flattened size from CNN becomes the 1st input size of the fc layers

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

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
        self.initialize_weights()

        # transfers the model to 'cuda' (GPU)
        self.to(self.device)


    @staticmethod
    def _calculate_flattened_size(input_size, filters, kernels, stride, padding, pooling_stride=2):
        """
        Dynamically calculates the flattened size after all convolutional and pooling layers.

        :param input_size (int): Size of the input (e.g., 64 for 64x64 image).
        :param filters (list): Number of filters for each convolutional layer.
        :param kernels (list): Kernel size for each convolutional layer.
        :param stride (int): Stride for each convolutional layer.
        :param padding (str): Type of padding ('same' or 'valid').
        :param pooling_stride (int): Stride of the pooling layer.

        :return: The flattened size after the final pooling layer.
        """
        size = input_size
        for kernel in kernels:
            if padding == 'same':
                size = size  # 'same' padding keeps the spatial dimensions unchanged
            else:
                size = (size - kernel + 1) // stride  # 'valid' padding reduces spatial dimensions

            size //= pooling_stride  # output size change due to pooling

        # final flattened size = (spatial size after pooling) * (number of filters in the last layer)
        return size * size * filters[-1]

    def initialize_weights(self):
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
        # Validating input dimensions
        if x.ndim != 4:  # expecting (batch_size, channels, height, width)
            raise ValueError(
                f'Input tensor must have 4 dimensions (batch_size, channels, height, width), but got {x.ndim}')

        # Passing through the convolutional layer
        for cnn_layer in self.conv_layers:
            x = cnn_layer(x)

        # Flattening the tensor starting from dimension 1, preserving the batch size (dimension 0).
        # This combines all subsequent dimensions (channels, height, width) into a single dimension,
        # resulting in a 2D tensor of shape (batch_size, flattened_size).
        x = torch.flatten(x, start_dim=1)

        # Passing through fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Final output layer
        x = self.output_layer(x)

        return x

    def __repr__(self):
        return f'{self.__class__.__name__} (\n' \
               f'CNN_layers={self.conv_layers}, \n \
               fc_layers={self.fc_layers}, \n' \
               f'activation={self.activation})'

    def configure_optimizer(self, optimizer_name='adam', lr=0.001, momentum=0.9, weight_decay=1e-4):
        """
          Configures and returns an optimizer for the model.

          Supported optimizers:
          - SGD (Stochastic Gradient Descent) with momentum.
          - Adam (Adaptive Moment Estimation).
          - Adagrad (Adaptive Gradient Algorithm).


          :param optimizer_name:str. The name of the optimizer to use ('sgd', 'adam', 'adagrad').
                                Default is 'adam'.
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

        elif optimizer_name.lower() == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)

        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from 'adam', 'sgd', or 'adagrad'.")


def calculate_metrics(predictions, targets):
    """
    Calculates regression metrics.

    :param predictions: Predicted values.
    :param targets: Actual values.

    :return: Dictionary with metrics .
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
    Trains the model for one epoch.

    :param model: The model to train.
    :param train_loader: DataLoader for training data. The dataset should already be standardized/normalized.
    :param optimizer: Optimizer to update the model's parameters.
    :param verbose: If True, prints batch-wise loss during training.

    :return: Average training loss for the epoch.
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

    for batch_idx, (features, targets, tileNo) in enumerate(train_loader):  # the 'tileNo' not used during model training/validation
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
    r2 = metrics_dict['R2']

    return avg_loss, rmse, r2


def validate(model, val_loader, verbose=False):
    """
    Validates the model for one epoch.

    :param model: The model to validate.
    :param val_loader: DataLoader for validation data. The dataset should already be standardized/normalized.
    :param verbose: If True, prints batch-wise loss during validation.

    :return: Average validation loss for the epoch.
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
        for batch_idx, (features, targets, tileNo) in enumerate(val_loader):  # the 'tileNo' not used during model training/validation
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
    r2 = metrics_dict['R2']

    return avg_loss, rmse, r2


def test(model, tile_dir, target_csv, batch_size, data_type='test'):
    """
    Evaluates the model on the test dataset. However, it can be employed for train and validation
    dataset as well.

    :param model: The trained model.
    :param tile_dir: Directory containing input data tiles. Can be from the tran/validation/test set.
    :param target_csv: CSV file with validation targets. Can be from the tran/validation/test set.
    :param batch_size: int. Batch size used in the DataLoader.
    :param data_type (str): Type of data passed to the DataLoader class.

    :return: Average loss and additional metrics.
    """
    # DataLoader
    DataLoader = DataLoaderCreator(tile_dir, target_csv,
                                   batch_size=batch_size,
                                   data_type=data_type).get_dataloader()

    # setting model to evaluation mode
    # in evaluation mode, PyTorch removes all dropout layers
    model.eval()

    # initiating running_loss to accumulate the total loss across all batches
    running_loss = 0.0

    # mse function
    mse_func = torch.nn.MSELoss()

    # empty lists to store predictions and actuals for estimating metrics
    predictions, actuals = [], []

    with torch.no_grad():    # disable gradient computation
        for features, targets, tileNo in DataLoader:  # the 'tileNo' not used during model primary model testing
            features, targets = features.to(model.device), targets.to(model.device).view(-1, 1)

            # forward pass
            preds = model(features)

            # accumulate loss for each batch
            loss = mse_func(preds, targets)
            running_loss += loss.item()

            # collect predictions and actuals for metrics
            predictions.extend(preds.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    # average loss for the epoch
    avg_loss = running_loss / len(DataLoader)

    # additional metrics (e.g., R², MAE)
    metrics_dict = calculate_metrics(predictions, actuals)

    rmse = metrics_dict['RMSE']
    mae = metrics_dict['MAE']
    r2 = metrics_dict['R2']

    print(f'Results -> Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\n')

    return avg_loss, rmse, mae, r2


def run_default_model(train_loader, val_loader,
                      n_features, input_size, n_epochs, 
                      filters, padding='same', pooling='maxpool',
                      lr=1e-3, kernel_size=3, stride=1,
                      activation_func='relu', fc_units=None,
                      weight_decay=1e-4, dropout_rate=0.2,
                      implement_earlyStopping=False,
                      patience=10, start_EarlyStop_count_from_epoch=40,
                      verbose=True):
    """
    Main training function to train and validate the model.

    :param train_loader: Train DataLoader.
    :param val_loader: Validation DataLoader.
    :param n_features: int.  Number of input channels in the image.
    :param input_size : int. Height/width of the square input image (e.g., 64 for 64x64).
    :param n_epochs: int. Number of training epochs.
    :param filters: list of int. Number of filters in each CNN layer.
    :param kernel_size: list of int. Kernel size for convolutional layers.
    :param padding: str. Padding type for CNN layers ('same' or 'valid'). Defaults to 'same'.
    :param pooling: str. Pooling type for CNN layers ('maxpool' or 'avgpool'). Defaults to 'maxpool'.
    :param lr: float. Learning rate for the optimizer. Defaults to 1e-3.
    :param stride: int. Stride for kernel operation. Defaults to 1.
    :param activation_func: str. Have to be either 'relu' or 'leakyrelu'. Defaults to 'relu'.
    :param fc_units: int. Number of units in the fully connected layer. Default set to None to use [128].
                     neurons in a single layer.
    :param weight_decay: float. Weight decay (L2 regularization) coefficient to control overfitting.
                                Default is 1e-4. A smaller value (e.g., 1e-6) penalizes large weights
                                less, while a larger value (e.g., 1e-3) penalizes them more.
    :param dropout_rate: float. Dropout rate for fully connected layers.
    :param implement_earlyStopping: boolean. Set to True to initiate early stopping.
    :param patience: int. Number of epochs to wait before early stopping. Default to 10.
    :param start_EarlyStop_count_from_epoch: int. Epoch to start checking early stopping. Defaults to 40 to enable
                                             more early epoch before initializing early stopping.
    :param verbose: Set to True to print training progress at each 10 epoch.

    :return: - The trained model.
             - A dictionary containing hyperparameters, training losses, and validation losses.
    """

    global train_loss, val_loss, train_rmse, train_r2, val_rmse, val_r2

    # initializing the model
    model = CNNRegression(
        n_features=n_features,
        input_size=input_size,
        filters=filters,
        kernels=kernel_size,
        stride=stride,
        activation_func=activation_func,
        padding=padding,
        pooling=pooling,
        fc_layers=fc_units,
        dropout_rate=dropout_rate
    )

    # configuring optimizer
    optimizer = model.configure_optimizer(optimizer_name='adam', lr=lr, weight_decay=weight_decay)

    # initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience)

    # empty dictionary to store model parameters and losses
    model_info = {}

    # empty lists to track losses
    train_losses = []
    val_losses = []
    last_epoch = None  # Track the last trained epoch

    for epoch in range(n_epochs):

        epoch = epoch + 1 # making epoch starting from 1

        # training and validation for one epoch
        train_loss, train_rmse, train_r2 = train(model, train_loader, optimizer)
        val_loss, val_rmse, val_r2 = validate(model, val_loader)

        # storing losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # storing the last trained epoch
        last_epoch = epoch

        # printing progress
        if verbose and epoch % 10 == 0:
            print(f'Train Epoch: {epoch} | Loss: {train_loss:.3f} | RMSE: {train_rmse:.3f} | R²: {train_r2:.3f}')
            print(f'Val   Epoch: {epoch} | Loss: {val_loss:.3f}   | RMSE: {val_rmse:.3f}   | R²: {val_r2:.3f}')
            print('---------------------------------------------------------------------')

        # checking for early stopping
        if epoch >= start_EarlyStop_count_from_epoch and implement_earlyStopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    # printing final performance (last trained epoch)
    print(f'Final Train Epoch: {last_epoch} | Loss: {train_loss:.3f} | RMSE: {train_rmse:.3f} | R²: {train_r2:.3f}')
    print(f'Final Val   Epoch: {last_epoch} | Loss: {val_loss:.3f}   | RMSE: {val_rmse:.3f}   | R²: {val_r2:.3f}')
    print('---------------------------------------------------------------------')

    # handling val_loss when early stopping is disabled
    if not implement_earlyStopping:
        best_val_loss = min(val_losses)

    else:
        best_val_loss = early_stopping.best_loss

    # saving model information and losses
    model_info['hyperparameters'] = {
        'lr': lr,
        'filters': filters,
        'kernel_size': kernel_size,
        'fc_units': fc_units
    }
    model_info['train_losses'] = train_losses
    model_info['val_losses'] = val_losses
    model_info['val_loss'] = best_val_loss

    return model, model_info


def run_and_tune_model(trial, train_loader, val_loader,
                       n_features, input_size, n_epochs,
                       padding='same', pooling='maxpool',
                       activation_func='relu',
                       implement_earlyStopping=False,
                       patience=10, start_EarlyStop_count_from_epoch=40):
    """
    Objective function for Optuna parameter tuning.

    This function is called for each trial during the parameter search.
    It defines the search space for parameters, trains a CNN model
    using the sampled parameters, and evaluates the model's performance
    on the validation set. The validation loss is returned to Optuna to guide
    the optimization process.

    :param trial: optuna.trial.Trial. A trial object provided by Optuna. It is used to
                  sample parameters and record the trial's results.
    :param train_loader: Train DataLoader.
    :param val_loader: Validation DataLoader.
    :param n_features: int.  Number of input channels in the image.
    :param input_size : int. Height/width of the square input image (e.g., 64 for 64x64).
    :param n_epochs: int. Number of training epochs.
    :param padding: str. Padding type for CNN layers ('same' or 'valid'). Defaults to 'same'.
    :param pooling: str. Pooling type for CNN layers ('maxpool' or 'avgpool'). Defaults to 'maxpool'.
    :param activation_func: str. Have to be either 'relu' or 'leakyrelu'. Defaults to 'relu'.
    :param implement_earlyStopping: boolean. Set to False to not initiate early stopping.
    :param patience: int. Number of epochs to wait before early stopping. Default to 10.
    :param start_EarlyStop_count_from_epoch: int. Epoch to start checking early stopping. Defaults to 40 to enable
                                             more early epoch before initializing early stopping.

    :return: The best validation loss (val_loss).
    """
    print(f'\nStarting trial number {trial.number}...\n')

    # sample hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # sample convolutional architecture
    num_layers = 2      # keeping number of convolutional layers fixed at 2 due to our specific input size
    filters = [trial.suggest_int(f'filters_layer_{i}', 32, 64, step=16) for i in range(num_layers)]
    kernel_size = [trial.suggest_int(f'kernel_size_layer_{i}', 3, 5, step=2) for i in range(num_layers)]

    # sample fully connected layer configuration
    num_fc_layers = trial.suggest_int('num_fc_layers', 1, 4)  # number of fully connected layers can be flexible from 1-3
    fc_units = [trial.suggest_int(f'fc_units_layer_{i}', 32, 256, step=32) for i in range(num_fc_layers)]
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)

    # training the model with the sampled parameters
    _, model_info = run_default_model(train_loader, val_loader,
                                      n_features=n_features, input_size=input_size,
                                      n_epochs=n_epochs, filters=filters,
                                      padding=padding, pooling=pooling,
                                      lr=lr, kernel_size=kernel_size, stride=1,
                                      activation_func=activation_func, fc_units=fc_units,
                                      weight_decay=weight_decay, dropout_rate=dropout_rate,
                                      implement_earlyStopping=implement_earlyStopping,
                                      patience=patience, start_EarlyStop_count_from_epoch=start_EarlyStop_count_from_epoch,
                                      verbose=True)

    # objective function
    best_val_loss = model_info['val_loss']  # for a specific trial
    train_loss_at_best_val = model_info['train_losses'][model_info['val_losses'].index(best_val_loss)]

    alpha = 0.3  # Weight for the penalty term to minimize the gap between train and validation loss
    objective_value = best_val_loss + alpha * abs(train_loss_at_best_val - best_val_loss)  # param tuning will minimize this value

    # storing additional information for later use
    best_epoch = model_info['val_losses'].index(best_val_loss)          # best epoch
    trial.set_user_attr('model_info', model_info)                       # saving model info for the trial
    trial.set_user_attr('best_epoch', best_epoch)                       # saving best epoch for the trial

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

    print(f'\nParameter importance plot saved from paramater tuning process')


def main(tile_dir_train, target_csv_train,
         tile_dir_val, target_csv_val, batch_size,
         n_features, input_size, n_epochs,
         model_save_path, model_info_save_path,
         padding='same', pooling='maxpool',
         activation_func='relu',
         default_params=None,
         implement_earlyStopping=False,
         patience=10, start_EarlyStop_count_from_epoch=40,
         tune_parameters=False, n_trials=50,
         plot_hyperparams_importance=False,
         hyperparam_importance_plot_path=None
         ):
    """
    Main function to either run the model with default parameters or perform parameter tuning.

    ****
    - when the model is set to tuning mode, set tune_parameters = True and default_params = None
    - when the model is in default mode, set tune_parameters = False and default_params = a dictionary of params

    :param tile_dir_train: Directory containing training set input tiles.
    :param target_csv_train: CSV file with training set target values.
    :param tile_dir_val: Directory containing validation set input tiles.
    :param target_csv_val: CSV file with validation set target values.
    :param batch_size: int. Batch size of DataLoader.
    :param n_features: int. Number of input channels in the image.
    :param input_size : int. Height/width of the square input image (e.g., 64 for 64x64).
    :param n_epochs: int. Number of training epochs.
    :param model_save_path: str. Path for saving the best model checkpoint.
    :param model_info_save_path: Filepath (pkl) to save the parameter-tuned or default model's params and
                                 losses.
    :param padding: str. Padding type for CNN layers ('same' or 'valid'). Defaults to 'same'.
    :param pooling: str. Pooling type for CNN layers ('maxpool' or 'avgpool'). Defaults to 'maxpool'.
    :param activation_func: str. Activation function ('relu' or 'leakyrelu'). Defaults to 'relu'.
    :param default_params: dict or None. Default parameters for running the model. Ignored when `tune_parameters` is True.
    :param implement_earlyStopping: boolean. Set to False to not initiate early stopping.
    :param patience: int. Number of epochs to wait before early stopping. Default to 10.
    :param start_EarlyStop_count_from_epoch: int. Epoch to start checking early stopping. Defaults to 40 to enable
                                             more early epoch before initializing early stopping.
    :param tune_parameters: bool. If True, perform parameter tuning using Optuna. Defaults to False.
    :param n_trials: int. Number of Optuna trials for parameter tuning. Defaults to 50.
    :param plot_hyperparams_importance: bool. Set to True to plot hypapameter importance plot during tuning
                                        parameters.
    :param hyperparam_importance_plot_path: str. Filepath to save hypeparam importance plot. Default set to None.


    :return: - The trained model.
             - A dictionary containing hyperparameters, training losses, and validation losses.
    """
    # creating storage directory
    makedirs([os.path.dirname(model_info_save_path)])

    # creating train and validation DataLoaders
    train_loader = DataLoaderCreator(tile_dir_train, target_csv_train,
                                     batch_size=batch_size,
                                     data_type='train').get_dataloader()

    val_loader = DataLoaderCreator(tile_dir_val, target_csv_val,
                                   batch_size=batch_size,
                                   data_type='validation').get_dataloader()

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
            n_features=n_features, input_size=input_size, n_epochs=n_epochs,
            padding=padding, pooling=pooling, activation_func=activation_func,
            ), n_trials=n_trials)

        # best parameters achieved from hyperparameter training
        best_params = study.best_trial.params
        best_epoch = study.best_trial.user_attrs["best_epoch"]

        # retraining the best model (optional)
        print('\nRetraining the best model...')
        trained_model, best_model_info = run_default_model(
                                         train_loader=train_loader,
                                         val_loader=val_loader,
                                         n_features=n_features,
                                         input_size=input_size,
                                         n_epochs=best_epoch,
                                         filters=[best_params[f'filters_layer_{i}'] for i in range(2)],  # Num conv layers fixed at 2
                                         kernel_size=[best_params[f'kernel_size_layer_{i}'] for i in range(2)],  # Num conv layers fixed at 2
                                         fc_units=[best_params[f'fc_units_layer_{i}'] for i in range(best_params['num_fc_layers'])],
                                         padding=padding,
                                         pooling=pooling,
                                         lr=best_params['lr'],
                                         activation_func=activation_func,
                                         weight_decay=best_params['weight_decay'],
                                         dropout_rate=best_params['dropout'],
                                         implement_earlyStopping=implement_earlyStopping,
                                         patience=patience,
                                         start_EarlyStop_count_from_epoch=start_EarlyStop_count_from_epoch,
                                         verbose=True)

        # save the best model's information and the best model
        with open(model_info_save_path, 'wb') as f:
            pickle.dump(best_model_info, f)

        torch.save(trained_model.state_dict(), model_save_path)
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

        print('\nRunning the model with default parameters:')
        print(default_params)

        # running the model with default hyperparameters and configuration parameters
        trained_model, model_info = run_default_model(
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    n_features=n_features,
                                    input_size=input_size,
                                    n_epochs=n_epochs,
                                    filters=default_params['filters'],
                                    kernel_size=default_params['kernel_size'],
                                    fc_units=default_params['fc_units'],
                                    padding=padding,
                                    pooling=pooling,
                                    lr=default_params['lr'],
                                    activation_func=activation_func,
                                    weight_decay=default_params['weight_decay'],
                                    dropout_rate=default_params['dropout'],
                                    implement_earlyStopping=implement_earlyStopping,
                                    patience=patience,
                                    start_EarlyStop_count_from_epoch=start_EarlyStop_count_from_epoch,
                                    verbose=True)

        # saving the model information and model
        with open(model_info_save_path, 'wb') as f:
            pickle.dump(model_info, f)

        torch.save(trained_model.state_dict(), model_save_path)
        print(f'\nModel saved at {model_save_path}')

        # printing model best parameters' summary
        print('\nModel default params:')
        print(default_params)
        print(f"\nEpochs ran - {len(model_info['val_losses'])}")
        print(f"\nLast val loss - {model_info['val_losses'][-1]:.3f}")

        return trained_model, model_info


def unstandardize_save_and_test(model, tile_dir, target_csv, mean_csv, std_csv,
                                output_csv, batch_size, data_type='test',
                                skip_processing=False):
    """
    Unstandardizes the trained model's prediction, saves the results in a csv, and does performance testing on the
    unstandardized data.

    :param model: The trained model object.
    :param tile_dir: Directory containing train/test/validation set input tiles.
    :param target_csv: CSV file with train/test/validation set target values.
    :param mean_csv: Filepath of csv with mean values of features and target data. Used in unstandardizing.
    :param std_csv: Filepath of csv with standard deviation values of features and target data. Used in unstandardizing.
    :param output_csv: Filepath of output csv that will hold the unstandardized results.
    :param batch_size: Batch size used in the DataLoader.
    :param data_type: Default set to 'test' as we generally apply model.eval() mode to do
                      model performance testing on any data by regarding them as 'test' data.
    :param skip_processing: Set to True to skip this step.

    :return: None.
    """
    if not skip_processing:
        # DataLoader
        DataLoader = DataLoaderCreator(tile_dir, target_csv,
                                       batch_size=batch_size,
                                       data_type=data_type).get_dataloader()

        # empty lists to store predictions and tileNo
        predictions, tile_no_list = [], []

        # setting model to evaluation mode
        # in evaluation mode, PyTorch removes all dropout layers
        model.eval()

        with torch.no_grad():  # disable gradient computation
            for features, targets, tileNo in DataLoader:
                features, targets = features.to(model.device), targets.to(model.device).view(-1, 1)

                # forward pass
                preds = model(features)

                # collect predictions and actuals for metrics
                predictions.extend(preds.cpu().numpy().flatten())
                tile_no_list.extend(tileNo.cpu().numpy().flatten())

        # making a new dataframe out of tile_no and standardized predictions
        model_output_df = pd.DataFrame({'tile_no': tile_no_list, 'standardized_pred': predictions})

        # unstandardizing predictions
        mean_df = pd.read_csv(mean_csv)
        std_df = pd.read_csv(std_csv)

        target_mean = mean_df.set_index('variable').loc['target', 'value']
        target_std = std_df.set_index('variable').loc['target', 'value']

        model_output_df['unstandardized_pred'] = (model_output_df['standardized_pred'] * target_std) \
                                                 + target_mean

        # loading dataframe with actual (actual pumping values) values of target
        original_df = pd.read_csv(target_csv)

        # merging the two dataframes
        final_df = model_output_df.merge(original_df, on='tile_no')

        # saving the csv
        makedirs([os.path.dirname(output_csv)])
        final_df.to_csv(output_csv, index=False)

        # checking scores for unstandardized model output and actual target data
        metrics_dict = calculate_metrics(predictions=final_df['unstandardized_pred'].tolist(),
                                         targets=final_df['target_value'].tolist())

        rmse = metrics_dict['RMSE']
        mae = metrics_dict['MAE']
        r2 = metrics_dict['R2']
        nrmse = metrics_dict['Normalized RMSE']

        return rmse, mae, r2, nrmse

    else:
        pass


def plot_learning_curve(train_loss, val_loss, plot_save_path):
    """
    Loads saved losses from a file, plots them, and saves the plot as an image.

    :param train_loss: Training losses as a series or list.
    :param val_loss: Validation losses as a series or list.
    :param plot_save_path: str. Path to save the plotted loss image.
    """
    # plotting losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (mean squared error)')
    plt.legend()

    # saving the plot as an image file
    plt.savefig(plot_save_path)
    plt.close()
    print(f'\nLoss plot saved...')


def plot_shap_values(trained_model, tile_dir, target_csv, batch_size,
                     plot_save_path, feature_names, data_type='test',
                     skip_processing=False):
    """
    Plot input variables importance plot based on SHAP values.

    **Note**
    The feature name list must be in the same order of input features (follow the dataset order of
    dr01_tile_westUS.py file's datasets_dict()). The code will then automatically sort the
    feature names based on their importance from SHAP values.

    :param trained_model: Trained model object.
    :param tile_dir: Directory containing train/test/validation set input tiles.
    :param target_csv: CSV file with train/test/validation set target values.
    :param batch_size: Batch size used in the DataLoader.
    :param plot_save_path: Filepath to save the plot.
    :param feature_names: List of representative feature names.
                         They must be in the same order of input features (follow the dataset order of
                         dr01_tile_westUS.py file's datasets_dict()). The code will then automatically sort the
                         feature names based on their importance from SHAP values.
    :param data_type: Either of 'train'/'test'/'validation'.
    :param skip_processing: Set to True to skip making this plot.

    :return: None.
    """
    if not skip_processing:
        # loading data
        print('\n___________________________________________________________________________')
        print(f'\nplotting SHAP feature importance...')
        dataloader = DataLoaderCreator(tile_dir, target_csv,
                                       batch_size=batch_size,
                                       data_type=data_type).get_dataloader()

        # setting model to evaluation mode
        trained_model.eval()

        # extracting a batch of data for SHAP computation
        batch = next(iter(dataloader))
        features, _, _ = batch
        features = features.to(trained_model.device)

        # using SHAP GradientExplainer designed for PyTorch/TensorFlow (DeepExplainer doesn't work for some reasons)
        explainer = shap.GradientExplainer(trained_model, features)
        shap_values = explainer.shap_values(features)

        # converting to numpy
        shap_values = np.array(shap_values)
        shap_values = np.squeeze(shap_values)  # removing the extra dimensions of size 1

        print(f'features shape: {features.shape}')
        print(f'shape values shape: {shap_values.shape}')

        # Computing the absolute mean SHAP values across spatial dimensions (height & width)
        # Shape: (num of samples in subset/batch, num features)
        mean_shap_values_per_feature = np.mean(np.abs(shap_values), axis=(2, 3))

        # averaging across all samples of the subset/batch
        mean_shap_values = np.mean(mean_shap_values_per_feature, axis=0)  # Shape: (num features, )

        # sorting in descending order
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        sorted_mean_shap_values = mean_shap_values[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # plotting
        plt.figure(figsize=(8, 5))
        plt.barh(sorted_feature_names, sorted_mean_shap_values, color='crimson')

        # # Add text labels for each bar
        # for i, v in enumerate(mean_shap_values):
        #     plt.text(v + 0.02, i, f'+{v:.2f}', color='crimson', fontweight='bold', va='center')

        plt.xlabel('Mean absolute SHAP value')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()  # To have the most important feature on top
        plt.savefig(plot_save_path, dpi=200, bbox_inches='tight')
        plt.close()

    else:
        pass




