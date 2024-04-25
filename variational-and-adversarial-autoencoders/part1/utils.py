import torch
from torchvision.utils import make_grid, save_image
import numpy as np
import math


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"

    epsilon = torch.randn_like(mean)
    z = epsilon * std + mean
    return z

def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    KLD_element = (2*log_std).exp() + mean**2 - 1 - (2*log_std)
    KLD = 0.5 * torch.sum(KLD_element, dim=-1)
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    bpd = elbo * torch.tensor(math.log(math.e, 2)) / torch.tensor((math.prod(img_shape[1:])))
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """
    values = torch.arange(0.5/grid_size, 1, 1/grid_size)

    p_z = torch.distributions.Normal(0, 1) 
    z_vals = p_z.icdf(values)
    
    z_batch = torch.cartesian_prod(z_vals, z_vals)

    output = decoder.forward(z_batch)
    prob = torch.nn.functional.softmax(output, dim=1)

    imgs_shape = prob.shape

    prob = prob.permute(0, 2, 3, 1)
    prob = prob.reshape((np.prod(list(prob.shape[:-1])), prob.shape[-1]))

    imgs = torch.multinomial(prob, 1)
    imgs = imgs.reshape((imgs_shape[0],1,imgs_shape[2],imgs_shape[3])).float()

    img_grid = make_grid(imgs, nrow=20, normalize=True, value_range=(0, 1), pad_value=0.5)

    return img_grid

