# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import pickle
import numpy as np
from glob import glob

import pandas as pd
import rasterio as rio
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DataLoaderCreator():
    """
    A dataloader class to batchify tiles of input features and a single target value per tile for the model.
    """

    def __init__(self, tile_dir, target_csv, batch_size=64):
        """
        Initialize the DataLoader to batch the data.

        :param tile_dir (str): Directory containing .tif files for tiles.
        :param target_csv (csv): Target csv. Must have a tile_no anf value columns.
                                 The tile_no column represents the corresponding tile no and value
                                 represents the value to train/validate/test on.
        :param batch_size (int): Batch size for the DataLoader.
        """
        print('Initializing DataLoader to batch the data...\n')

        # reading target and input datasets.
        # we have to make sure that target values and tiles are matching (tiles have corresponding target values).
        # so, creating a dictionary mapping tile_no (from the file name) to its full file path for quick lookup.
        # then, sorting tiles to match the order of tile_no_list from the target CSV.
        # finally, reading the sorted tiles as arrays
        target_df = pd.read_csv(target_csv)
        target_values = target_df['standardized_value'].tolist()
        tile_no_list = target_df['tile_no'].tolist()

        tiles = glob(os.path.join(tile_dir, '*.tif'))
        tile_dict = {os.path.basename(tile).split('_')[-1].replace('.tif', ''): tile for tile in tiles}
        tiles_sorted = [tile_dict[str(tile_no)] for tile_no in tile_no_list if str(tile_no) in tile_dict]

        features_arrs = [rio.open(tt).read() for tt in tiles_sorted]  # storing multi-band features as array

        print('Same sorting order for feature tiles and target values ensured...\n')

        # creating numpy arrays for features and target
        features_np = np.stack(features_arrs)  # dimensions are - num of image * num features * height * width
        target_np = np.asarray(target_values)

        # converting to pyTorch tensor
        self.features_tensor = torch.tensor(features_np, dtype=torch.float32)
        self.target_tensor = torch.tensor(target_np, dtype=torch.float32)

        # checking and printing the shapes of the tensors before batching
        print('Features Tensor Shape before batching:', self.features_tensor.shape)
        print('Target Tensor Shape before batching:', self.target_tensor.shape, '\n')

        # creating a TensorDataset
        self.dataset = TensorDataset(self.features_tensor, self.target_tensor)

        # creating the DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,
                                     shuffle=True)  # shuffle True is randomizing the the data

        # checking and printing the shapes of the tensors after batching
        for (feature_batch, target_batch) in self.dataloader:
            print('\n Features Tensor Shape after batching:', feature_batch.shape)
            print('Target Tensor Shape after batching:', target_batch.shape, '\n')
            break

    def get_dataloader(self):
        return self.dataloader


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the model with the lowest validation loss.
    """

    def __init__(self, save_path, patience=5, delta=0.01):
        """
        Initializes the early stopping mechanism.

        :param save_path: str. Path for saving the best model checkpoint.
        :param patience: int. Number of epochs to wait before stopping if no improvement.
                         Default is 5.
        :param delta: float. Minimum change in the monitored quantity to qualify as an improvement.
                      Default is 0.01.
        """

        self.patience = patience  # Number of epochs to wait for improvement
        self.delta = delta  # Minimum change to qualify as improvement
        self.counter = 0  # Count epochs with no improvement
        self.best_loss = None  # Best loss seen so far
        self.early_stop = False  # Whether to stop training
        self.path = save_path

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

            torch.save(model.state_dict(), self.path)

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
                 activation_func='relu', pooling='maxpool', fc_layers=None):
        """
        Initializes a CNN with specified architecture.

        :param n_features (int): Number of channels in the input image.
        :param input_size (int): Height/width of the square input image (e.g., 64 for 64x64 images).
        :param filters (list): Number of filters in each convolutional layer.
        :param kernels (list): Kernel size for each convolutional layer.
        :param stride (int): Stride for convolutional layers. Defaults to 1.
        :param padding (str or int): Padding type ('same' or 'valid'). 'valid' resembles to 0 padding. Can also be
                                     integer values like 1 or 2.
        :param activation_func (str): Type of activation function ('relu', 'leakyrelu').
        :param pooling (str): Pooling option ('maxpool', 'avgpool').
        :param fc_layers (list): Number of units in each fully connected layer. Defaults to [128] if set to None.
        """

        # initialize the parent class (nn.Module) to properly set up the model.
        # This ensures that layers and parameters are registered, and functionality like
        # .parameters(), .state_dict(), and device management works correctly.
        super().__init__()

        # device
        self.device = 'cuda'  # running the code on GPU
        print(f'Model running on {self.device}....')

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
            self.conv_pools.append(pooling)  # downsampling

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

        for fc_unit in fc_layers:
            self.fc_layers.append(nn.Linear(input_size, fc_unit))
            self.fc_layer.append(self.activation)

            # output of each hidden layer step will be input of next hidden layer
            input_size = fc_unit

        # final output layer
        # Final output size set to 1 as each input tile (variable set) will give prediction over 1 pixel
        self.output_layer = nn.Linear(input_size, 1)

        # weight initialization
        self.initialize_weights()

        # transfers the model to 'cuda' (GPU)
        self.to(self.device)

    def _calculate_flattened_size(self, input_size, filters, kernels, stride, padding, pooling_stride=2):
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


def train(model, train_loader, optimizer, verbose=True):
    """
    Trains the model for one epoch.

    :param model: The model to train.
    :param train_loader: DataLoader for training data. The dataset should already be standardized/normalized.
    :param optimizer: Optimizer to update the model's parameters.
    :param verbose: If True, prints batch-wise loss during training.

    :return: Average training loss for the epoch.
    """
    device = 'cuda'  # device is GPU by default

    # set the model to training mode
    model.train()
    running_loss = 0.0

    # mse function
    mse_func = torch.nn.MSELoss()

    for batch_idx, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device).view(-1, 1)

        # forward pass
        predictions = model(features)
        loss = mse_func(predictions, targets)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate loss for the epoch
        runn
        froming_loss += loss.item()

        # print every 10 batches
        if verbose and batch_idx % 10 == 0:
            print(f'Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}')

    # average loss for the epoch
    avg_loss = running_loss / len(train_loader)

    return avg_loss


def validate(model, val_loader, verbose=True):
    """
    Validates the model for one epoch.

    :param model: The model to validate.
    :param val_loader: DataLoader for validation data. The dataset should already be standardized/normalized.
    :param verbose: If True, prints batch-wise loss during validation.

    :return: Average validation loss for the epoch.
    """
    device = 'cuda'  # device is GPU by default

    # Set the model to evaluation mode
    model.eval()
    running_loss = 0.0

    # mse function
    mse_func = torch.nn.MSELoss()

    with torch.no_grad():  # disable gradient computation
        for batch_idx, (features, targets) in enumerate(val_loader):
            features, targets = features.to(device), targets.to(device).view(-1, 1)

            # forward pass
            predictions = model(features)
            loss = mse_func(predictions, targets)

            # accumulate loss for the epoch
            running_loss += loss.item()

            # print every 10 batches
            if verbose and batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx + 1}/{len(val_loader)} - Loss: {loss.item():.4f}")

    # average loss for the epoch
    avg_loss = running_loss / len(val_loader)

    return avg_loss


def train_validate(model, train_loader, val_loader, n_epochs, optimizer,
                   model_save_path, loss_save_path,
                   patience=5, start_EarlyStop_count_from_epoch=0,
                   verbose=True):
    """
    Handles the training and validation loop across multiple epochs.

    :param model: The model to train and validate.
    :param train_loader: DataLoader for training data. The dataset should already be standardized/normalized.
    :param val_loader: DataLoader for validation data. The dataset should already be standardized/normalized.
    :param n_epochs: Number of epochs for training.
    :param optimizer: Optimizer to update the model's parameters.
    :param model_save_path: Model save path. Model is saved while checking the early stopping criteria.
    :param loss_save_path: Path to save the training and validation losses.
    :param patience: int. Number of epochs to wait before stopping if no improvement.
                     Default is 5.
    :param start_EarlyStop_count_from_epoch: int. Epoch from which early_stop tracking will start for validation_loss.
                                             Default set to 0 to start tracking from the very beginning.
    :param verbose: If True, prints epoch-wise losses.

    :return: Lists of training and validation losses across all epochs.
    """
    # initializing emtpy list to track train and validation losses across all epochs
    train_losses = []
    val_losses = []

    # initializing early stopping
    early_stopping = EarlyStopping(save_path=model_save_path, patience=patience, delta=0.01)

    # looping for each epoch
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # training the model for one epoch
        train_loss = train(model, train_loader, optimizer, verbose=verbose)
        train_losses.append(train_loss)

        # validating the model after the epoch
        val_loss = validate(model, val_loader, verbose=verbose)
        val_losses.append(val_loss)

        # checking for early stopping
        if epoch >= start_EarlyStop_count_from_epoch:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print('Training stopped early')
                break

        # printing epoch summary
        if verbose:
            print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # saving losses
    losses = {'train_losses': train_losses, 'val_losses': val_losses}
    with open(loss_save_path, 'wb') as f:
        pickle.dump(losses, f)

    return train_losses, val_losses

# def load_losses_and_print(loss_save_path, plot_save_path):
#     """
#     Loads saved losses from a file, plots them, and saves the plot as an image.
#
#     :param loss_save_path: str. Path to the saved pickle file containing losses.
#     :param plot_save_path: str. Path to save the plotted loss image.
#     """
#     # loading the losses saved during model training
#     with open(loss_save_path, 'rb') as f:
#         losses = pickle.load(f)
#
#     train_losses = losses['train_losses']
#     val_losses = losses['val_losses']
#
#     # plotting losses
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='Training Loss', marker='o')
#     plt.plot(val_losses, label='Validation Loss', marker='x')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#
#     # saving the plot as an image file
#     plt.savefig(plot_save_path)
#     plt.close()
#     print(f'Loss plot saved')

# standardize and normalize code (including de-stanardization and de-normalization)
# need a function for loading model
# need a function for model evaluation (gives the training, validation, and testing performancne)
# integrate de-standardization option for target data
