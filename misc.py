import torch
import torch.nn as nn
import numpy as np


def get_generator() -> nn.Module:
    """
    Returns the generator network
    :return: (nn.Module) Simple feed forward neural network with three layers,
    """
    return nn.Sequential(nn.Linear(2, 10), nn.LeakyReLU(), nn.Linear(10, 10), nn.LeakyReLU(), nn.Linear(10, 2))


def get_discriminator() -> nn.Module:
    """
    Returns the discriminator network.
    :return: (nn.Module) Simple feed forward neural network with three layers and probability output.
    """
    return nn.Sequential(nn.Linear(2, 10), nn.LeakyReLU(), nn.Linear(10, 10), nn.LeakyReLU(), nn.Linear(10, 1),
                         nn.Sigmoid())


def get_data(samples: int = 400, variance: float = 0.05) -> torch.tensor:
    """
    Function generates a 2d ring of 8 gaussians
    :param samples: (int) Number of samples including in the resulting dataset. Must be a multiple of 8.
    :param variance: (float) Variance of the gaussian
    :return: (Torch Tensor) generated data
    """
    assert samples % 8 == 0 and samples > 0, "Number of samples must be a multiple of 8 and bigger than 0"
    # Init angels of the means
    angels = np.cumsum((2 * np.pi / 8) * np.ones((8)))
    # Convert angles to 2D coordinates
    means = np.array([np.cos(angels), np.sin(angels)])
    # Generate data
    data = np.empty((2, samples))
    counter = 0
    for gaussian in range(means.shape[1]):
        for sample in range(int(samples / 8)):
            data[:, counter] = np.random.normal(means[:, gaussian], variance)
            counter += 1
    # Reshape data for use in neural network
    data = data.T
    # Shuffle data
    np.random.shuffle(data)
    # Convert numpy array to tensor
    return torch.from_numpy(data).float()
