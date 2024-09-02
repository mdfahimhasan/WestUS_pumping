import os
import numpy as np
from glob import glob
import rasterio as rio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# # next TO-DO
# take care of 2012 GCVI, NDVI, NDMI
# standardization
# nanMSEloss


class DataLoaderCreator():
    """
    A dataloader class to bactchify features and target for the model.
    """

    def __init__(self, tile_dir, batch_size=64):
        print('Initializing DataLoader to batch the data...')

        # reading all datasets as numpy array
        tiles = glob(os.path.join(tile_dir, '*.tif'))
        tile_arrs = [rio.open(tt).read() for tt in tiles]

        # separating features and target
        features = []
        target = []
        for arr in tile_arrs:
            target.append(arr[0])  # First band is target (netGW_Irr)
            features.append(arr[1:])  # Second to last bands are features

        # converting lists to numpy arrays
        features_np = np.stack(features)  # dimensions are - num of image * num features * height * width
        target_np = np.stack(target)  # dimensions are - num of image * height * width  (here num features = 1)

        # convert to pyTorch tensor
        self.features_tensor = torch.tensor(features_np, dtype=torch.float32)
        self.target_tensor = torch.tensor(target_np, dtype=torch.float32)

        # Check and print the shapes of the tensors before batching
        print('Features Tensor Shape before batching:', self.features_tensor.shape)
        print('Target Tensor Shape before batching:', self.target_tensor.shape)

        # Create a TensorDataset
        self.dataset = TensorDataset(self.features_tensor, self.target_tensor)

        # Create the DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Check and print the shapes of the tensors after batching
        for (feature_batch, target_batch) in self.dataloader:
            print('\n Features Tensor Shape after batching:', feature_batch.shape)
            print('Target Tensor Shape after batching:', target_batch.shape, '\n')
            break

    def get_dataloader(self):
        return self.dataloader


tile_dir = '../../Data_main/rasters/multibands/tiles'
dataloader_creator = DataLoaderCreator(tile_dir, batch_size=64)
dataloader = dataloader_creator.get_dataloader()


class CNN_regression(torch.nn.Module):
    """
    A Convolutional Neural Network (CNN) Class for regression tasks.
    """

    def __init__(self, n_features, filters, kernels, stride=1, padding='same', activation_func='relu'):
        """
        Initializes a CNN with specified architecture.

        :param n_features: int. Number of channels in the input image.
        :param filters: list. Number of filters in each convolutional layer.
        :param kernels: list. Kernel size for each convolutional layer.
        :param activation_func: str. Type of activation function ('relu', 'leakyrelu').
        """
        super(CNN_regression, self).__init__()

        self.device = 'cuda'  # running the code on GPU
        print(f'Model running on {self.device}....')

        # dictionary of available activation and pooling options
        self.activations = {'relu': nn.ReLU(),
                            'leakyrelu': nn.LeakyReLU(negative_slope=0.01)}  # neg slope of leakyRelu is by default

        # setting some initial properties of the model based on user input
        activation = self.activations.get(activation_func, nn.ReLU())  # Default to ReLU if not specified
        stride = 1 if padding == 'same' else stride  # by default stride =1 for padding = 'same'; adding statement to show it
        in_channels = n_features  # at the beginning the in_channels will be equal to the number of features in the stacked data

        # the CNN steps goes like - conv -> BN -> activation -> pooling
        # note that, the current model uses padding = 'same' and AdaptiveAvgPool2d to maintain original input dimension
        # throughout the network
        self.layers = nn.ModuleList()  # will hold the model components

        for out_channels, kernel_size in zip(filters, kernels):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(activation)

            #  *** nn.AdaptiveAvgPool2d *** here an adaptive average pooling layer is placed which comes from the forward function (due to changing input size)

            in_channels = out_channels  # at the end of each block, the in_channels of the next block is set as equal to the out_channel of the previous block

        # Final layer. out_channels and kernel size is set to 1 are used for channel reduction and linear transformation across channels
        # while preserving the original dimension of the image
        self.final_conv = nn.Conv2d(in_channels=filters[-1], out_channels=1, kernel_size=1, stride=1, padding=0)

        # weight initialization
        self.initialize_weights()

        # transfers the model to 'cuda' if device='cuda'
        self.to(self.device)

    def initialize_weights(self):
        """
        Initializes weight for the Neural Network model. For 'relu' and 'leakyrelu', initialization method has been
        set to 'kaiming_normal' (he_normal) as a popular choice.

        resources about xavier and kaiming initialization -
        https://pouannes.github.io/blog/initialization/#:~:text=The%20only%20difference%20is%20that,find%20it%20simpler%20and%20clearer.

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(self.activations['relu'], nn.ReLU):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(self.activations['leakyrelu'], nn.LeakyReLU):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network, maintains original input dimensions.
        """
        for layer in self.layers:
            x = layer(x)

            ## ***********************adjust
            output_size = (x.size(2), x.size(3))  # Extracting current feature map size (H, W)
            ## ***********************adjust
            x = nn.AdaptiveAvgPool2d(output_size)(x)  # Applying adaptive pooling dynamically

        x = self.final_conv(x)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}({self.layers}, final_conv={self.final_conv})'
