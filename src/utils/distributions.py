# distributions.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 05.03.24


import torch
import numpy as np
from normflows.distributions import BaseDistribution


class JointDistribution(BaseDistribution):
    """
    Defines a Normalizing Flow distribution that is a joint distribution of multiple independent distributions.

    Parameters
    ----------
    distributions: list,
        A list of distributions.
    dimensions: list,
        A list of dimensions of the distributions.

    """
    def __init__(self, distributions: list[BaseDistribution], dimensions: list[int]):

        super().__init__()
        self.distributions = distributions
        self.dimensions = [0] + dimensions


    def forward(self, num_samples):
        """
        Sample from the joint distribution.

        Parameters
        ----------
        num_samples: int,
            The number of samples to draw from the distribution.

        Returns
        -------
        samples: torch.Tensor,
            The samples drawn from the distribution.
        """
        return torch.cat([d.forward(num_samples) for d in self.distributions], dim=1)

    def log_prob(self, x: torch.Tensor):
        """
        Get the log probability of a batch of samples.

        Parameters
        ----------
        x: torch.Tensor,
            The batch of samples.

        Returns
        -------
        torch.Tensor,
            The log probability of the samples.

        """
        return sum(d.log_prob(x[:, self.dimensions[i]:self.dimensions[i+1]]) for i, d in enumerate(self.distributions))