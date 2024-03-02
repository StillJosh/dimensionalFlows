# utils.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24


import torch


def rejection_sampling_2d(p, num_samples=1000, min_x=-1, max_x=1, min_y=-1, max_y=1, M=None):
    """
    Rejection sampling algorithm.

    Parameters
    ----------
    p: function,
        The target distribution.

    Returns
    -------
    x: float,
        A sample from the target distribution.
    """
    square_size = (max_x - min_x) * (max_y - min_y)

    if M is None:
        # Estimate M by finding the maximum of p over a grid
        grid_size = 100
        xx, yy = torch.meshgrid(torch.linspace(min_x, max_x, grid_size), torch.linspace(min_y, max_y, grid_size))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        p_max = torch.max(p(zz))
        M = 1.1 * square_size * p_max

    res = torch.tensor([])
    while len(res) < num_samples:
        u = torch.rand(128)
        x = torch.rand(128, 2) * torch.tensor([max_x - min_x, max_y - min_y]) + torch.tensor([min_x, min_y])

        accept = u < p(x) / (M * torch.ones(128)/square_size)
        res = torch.cat([res, x[accept]], 0)

    return res[:num_samples]
