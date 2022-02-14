from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np


def get_generator(latent_size: int) -> nn.Module:
    """
    Returns the generator network.
    :param latent_size: (int) Size of the latent input vector
    :return: (nn.Module) Simple feed forward neural network with three layers,
    """
    return nn.Sequential(nn.Linear(latent_size, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.Tanh(),
                         nn.Linear(256, 2, bias=True))


def get_discriminator(use_spectral_norm: bool) -> nn.Module:
    """
    Returns the discriminator network.
    :param use_spectral_norm: (bool) If true spectral norm is utilized
    :return: (nn.Module) Simple feed forward neural network with three layers and probability output.
    """
    if use_spectral_norm:
        return nn.Sequential(spectral_norm(nn.Linear(2, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 1, bias=True)))
    return nn.Sequential(nn.Linear(2, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 1, bias=True))


def get_data(samples: Optional[int] = 400, variance: Optional[float] = 0.05) -> torch.Tensor:
    """
    Function generates a 2d ring of 8 Gaussians
    :param samples: (Optional[int]) Number of samples including in the resulting dataset. Must be a multiple of 8.
    :param variance: (Optional[float]) Variance of the gaussian
    :return: (torch.Tensor) generated data
    """
    assert samples % 8 == 0 and samples > 0, "Number of samples must be a multiple of 8 and bigger than 0"
    # Init angels of the means
    angels = torch.cumsum((2 * np.pi / 8) * torch.ones((8)), dim=0)
    # Convert angles to 2D coordinates
    means = torch.stack([torch.cos(angels), torch.sin(angels)], dim=0)
    # Generate data
    data = torch.empty((2, samples))
    counter = 0
    for gaussian in range(means.shape[1]):
        for sample in range(int(samples / 8)):
            data[:, counter] = torch.normal(means[:, gaussian], variance)
            counter += 1
    # Reshape data
    data = data.T
    # Shuffle data
    data = data[torch.randperm(data.shape[0])]
    # Convert numpy array to tensor
    return data.float()
