import os
import numpy as np
from glob import glob
import rasterio as rio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# # next TO-DO
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


class UNet_regression(nn.Module):
    """
    Initializes a U-Net like CNN with specified architecture for regression task.

    UNet structure detail: 1. https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
                           2. https://www.geeksforgeeks.org/u-net-architecture-explained/
    """

    def __init__(self, n_features, filters, kernels, stride=1, padding='same', activation_func='relu',
                 pooling='maxpool'):
        """
        Initializes a CNN with specified architecture.

        :param n_features: int. Number of channels in the input image, including the NaN mask channel at the end of each image.
        :param filters: list. Number of filters in each convolutional layer.
        :param kernels: list. Kernel size for each convolutional layer.
        :param activation_func: str. Type of activation function ('relu', 'leakyrelu').
        :param pooling: str. Pooling option ('maxpool', 'avgpool')
        """
        super(UNet_regression, self).__init__()

        # device
        self.device = 'cuda'  # running the code on GPU
        print(f'Model running on {self.device}....')

        # activation
        self.activations = {'relu': nn.ReLU(),
                            'leakyrelu': nn.LeakyReLU(negative_slope=0.01)}  # neg slope of leakyRelu is by default

        activation = self.activations.get(activation_func, nn.ReLU())  # Default to ReLU if not specified

        # polling
        self.poolings = {'maxpool': nn.MaxPool2d(2),
                         'avgpool': nn.AvgPool2d(2)}
        pooling = self.poolings.get(pooling, nn.MaxPool2d(2))  # by default MaxPool2d if not specified

        # stride
        stride = 1 if padding == 'same' else stride  # by default stride = 1 for padding = 'same'; adding statement to show it

        # input channels
        # n_features includes an additional channel for the NaN mask, which informs the network about the validity of input data throughout processing.
        in_channels = n_features

        # the UNet (CNN type model) has a encoder-decoder structure
        # encoder steps: conv -> BN -> activation -> pooling
        self.encoder_layers = nn.ModuleList()  # will hold the model components

        self.encoder_pools = nn.ModuleList()  # separate list for pooling layers

        for out_channels, kernel_size in zip(filters, kernels):
            self.encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.encoder_layers.append(nn.BatchNorm2d(out_channels))
            self.encoder_layers.append(activation)
            self.encoder_pools.append(pooling)  # downsampling

            in_channels = out_channels  # at the end of each block, the in_channels of the next block is set as equal to the out_channel of the previous block

        # skip connections will link encoder outputs to corresponding decoder layers #
        # skip connections feed high-resolution features from encoder to decoder for better reconstruction during upsampling

        # decoder steps: conv -> BN -> upsample
        self.decoder_layers = nn.ModuleList()
        reversed_filters = filters[::-1]  # reversing the filters for decoder
        for i, (out_channels, kernel_size) in enumerate(zip(reversed_filters, kernels[::-1])):
            self.decoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.decoder_layers.append(nn.BatchNorm2d(out_channels))
            self.decoder_layers.append(activation)
            if i < len(reversed_filters) - 1:   # apply upsampling except on the last decoder layer
                self.decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))  # double the spatial dimensions to reverse downsampling

            in_channels = out_channels  # at the end of each block, the in_channels of the next block is set as equal to the out_channel

        # final output layer
        self.final_conv = nn.Conv2d(in_channels=reversed_filters[-1], out_channels=1, kernel_size=1, stride=1, padding=0)

        # weight initialization
        self.initialize_weights()

        # transfers the model to 'cuda' (GPU)
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
        Forward pass through the network, includes skip connections.
        """
        # # # # # # # # # # # # # # # # # # # # # # # # nan data handling # # # # # # # # # # # # # # # # # # # # # # #
        # This model integrates a NaN mask as the final channel of each input tensor, which identifies invalid or missing data areas.
        # First, the mask (1 - valid values, 0 - nan values) goes into the model with other input features.
        # Finally, it is applied before the final convolution layer to guarantee that only valid data contributes to the
        # model's predictions, thereby enhancing the reliability and accuracy of the results.
        # later, we also estimate loss without incorporating the nan pixels to ensure that the model doesn't learn from those values.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # extracting the nan mask (the last channel of each input) will be applied before final convolution
        nan_mask = x[:, -1:, :, :]  # Extract the mask

        # initiating empty list to store outputs for skip connections
        encoder_outputs = []

        # encoder
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):  # save output just after activation but before pooling
                encoder_outputs.append(x)
            if i < len(self.encoder_pools):  # apply pooling after saving the output
                x = self.encoder_pools[i](x)

        # decoder
        encoder_outputs = encoder_outputs[::-1]  # reverse to use first saved encoder output last in decoding
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if isinstance(layer, nn.Upsample):  # add skip connection
                # The encoder outputs have been reversed, so using the pop(0) to get the corresponding layer's encoder_output/
                # pop(0) removes the first element in each decoder step, so we can keep using it to get the relevant encoder_output/
                # Then, using torch.cat() to concatenate along the channel dimension.
                # This helps reintroducing the details that might have lost during the downsampling in the encoder phase.
                skip_connection = encoder_outputs.pop(0)
                x = torch.cat((x, skip_connection), dim=1)

        # apply the NaN mask before the final convolution to zero out invalid areas,
        # ensuring that the model's predictions are solely based on valid data points.
        x = x * nan_mask

        # final output layer
        x = self.final_conv(x)

        return x

    def __repr__(self):
        return f'{self.__class__.__name__}(encoder_layers={self.encoder_layers}, decoder_layers={self.decoder_layers},' \
               f'final_conv={self.final_conv})'
