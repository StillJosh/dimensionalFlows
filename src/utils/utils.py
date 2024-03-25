# utils.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24


import torch
from typing import Optional


def rejection_sampling_2d(p: callable, num_samples: int = 1000, x_range: tuple = (-1, 1), y_range: tuple = (-1, 1),
                          big_m: Optional[float] = None) -> torch.Tensor:
    """
    Sample from a 2D distribution using rejection sampling.

    Parameters
    ----------
    p: function,
        The target distribution.
    num_samples: int,
        The number of samples to draw.
    x_range: tuple,
        The minimum and maximum x values to sample from.
    y_range: tuple,
        The minimum and maximum y values to sample from.
    big_m: float,
        An upper bound for the quotient of the target distribution and the proposal distribution.

    Returns
    -------
    x: torch.Tensor,
        A sample from the target distribution.
    """
    min_x, max_x = x_range
    min_y, max_y = y_range

    square_size = (max_x - min_x) * (max_y - min_y)

    # If big_m is None, estimate big_m by finding the maximum of p over a grid
    if big_m is None:
        grid_size = 100
        xx, yy = torch.meshgrid(torch.linspace(min_x, max_x, grid_size), torch.linspace(min_y, max_y, grid_size))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        p_max = torch.max(p(zz))
        big_m = 1.1 * square_size * p_max

    # Sample with rejection algorithm until num_samples samples are accepted
    res = torch.tensor([])
    while len(res) < num_samples:
        u = torch.rand(128)
        x = torch.rand(128, 2) * torch.tensor([max_x - min_x, max_y - min_y]) + torch.tensor([min_x, min_y])

        accept = u < p(x) / (big_m * torch.ones(128) / square_size)
        res = torch.cat([res, x[accept]], 0)

    return res[:num_samples]


def epoch_scheduler(dl, nfm, epoch, config):
    """
    Adjust the number of flows and the dimensionality of the dataset based on the current epoch.

    Parameters
    ----------
    dl: torch.utils.data.DataLoader,
        The dataloader.
    nfm: nf.NormalizingFlow,
        The normalizing flow model.
    epoch: int,
        The current epoch.
    """

    if hasattr(nfm, 'current_flow'):
        for epoch_switch in config['epoch_switch']:
            if epoch <= epoch_switch:
                nfm.current_flow = config['epoch_switch'].index(epoch_switch) + 1
                dl.dataset.return_dim = nfm.flow_dims[nfm.current_flow - 1]
                return

        nfm.current_flow = len(nfm.flow_dims)
        dl.dataset.return_dim = nfm.flow_dims[-1]

