# distributions.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 05.03.24


import torch
import numpy as np
from normflows.distributions import BaseDistribution


class JointDistribution(BaseDistribution):
    def __init__(self, distributions, dimensions):
        super().__init__()
        self.distributions = distributions
        self.dimensions = [0] + dimensions


    def forward(self, num_samples, context=None):
        return torch.cat([d.forward(num_samples) for d in self.distributions], dim=1)

    def log_prob(self, x):
        return sum(d.log_prob(x[:, self.dimensions[i]:self.dimensions[i+1]]) for i, d in enumerate(self.distributions))