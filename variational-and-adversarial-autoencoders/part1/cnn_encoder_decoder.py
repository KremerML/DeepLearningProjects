import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network
        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 MNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        act_fn = nn.GELU()
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2), # 28x28 => 14x14
            act_fn,
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 14x14 => 7x7
            act_fn,
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 7x7 => 4x4
            act_fn,
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*num_filters, z_dim)
        )
        self.fc_mean = nn.Linear(z_dim, z_dim)
        self.fc_var = nn.Linear(z_dim, z_dim)
        

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] of type long with values between 0 and 15.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        x = x.float() / 15 * 2.0 - 1.0  # Move images between -1 and 1
        
        enc1 = self.net(x)
        mean = self.fc_mean(enc1)
        log_std = self.fc_var(enc1)

        
        return mean, log_std


class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 16, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.
        Inputs:
            num_input_channels - Number of channels of the image to
                                 reconstruct. For a 4-bit MNIST, this parameter is 16
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """

        super().__init__()
        
        self.num_input_channels = num_input_channels
        act_fn = nn.GELU()
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2*16*num_filters),
            act_fn
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*num_filters, 2*num_filters, kernel_size=3, output_padding=0, padding=1, stride=2), # 4x4 => 7x7
            act_fn,
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(2*num_filters, num_filters, kernel_size=3, output_padding=1, padding=1, stride=2), # 7x7 => 14x14
            act_fn,
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2) # 14x14 => 28x28
            #nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
        
    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a softmax applied on it.
                Shape: [B,num_input_channels,28,28]
        """

        
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
