# distribution_plotting.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24

import matplotlib.pyplot as plt
import torch
from normflows.core import NormalizingFlow
from normflows.distributions import PriorDistribution
import seaborn as sns

# Plot target distribution
def plot_progress_2d(target: PriorDistribution, nfm: NormalizingFlow, device: torch.device, x_range: tuple = (-3, 3),
                     y_range: tuple = (-3, 3)) -> (plt.Figure, plt.Axes):
    """
    Plots the normalizing flow distribution as a heatmap and the target distribution as contour lines.

    Parameters
    ----------
    target: PriorDistribution,
        The target distribution.
    nfm: NormalizingFlow,
        The normalizing flow model.
    device: torch.device,
        The device on which the model is trained.
    x_range: tuple,
        The minimum and maximum x-values of the plot.
    y_range: tuple,
        The minimum and maximum y-values of the plot.

    Returns
    -------
    fig: plt.Figure,
        The figure of the plot.
    ax: plt.Axes,
        The axes of the plot.
    """

    min_x, max_x = x_range
    min_y, max_y = y_range

    # Define a grid to calculate the log probabilities on
    grid_size = 200
    xx, yy = torch.meshgrid(torch.linspace(min_x, max_x, grid_size), torch.linspace(min_y, max_y, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)

    # Calculate the probabilities of the target distribution
    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob_target = torch.exp(log_prob)

    # Calculate the probabilities of the normalizing flow distribution
    log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    # Plot both distributions
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.pcolormesh(xx, yy, prob.data.numpy())
    ax.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
    ax.set_aspect('equal', 'box')

    return fig, ax


def plot_progress_1d(target_samples: torch.tensor, nfm: NormalizingFlow, device: torch.device,
                     x_range: tuple = (-3, 3)) -> (plt.Figure, plt.Axes):
    """
    Plots the normalizing flow distribution as a histogram and the target distribution as a line plot.

    Parameters
    ----------
    target_samples: torch.tensor,
        The samples from the target distribution.
    nfm: NormalizingFlow,
        The normalizing flow model.
    device: torch.device,
        The device on which the model is trained.
    x_range: tuple,
        The minimum and maximum x-values of the plot.

    Returns
    -------
    fig: plt.Figure,
        The figure of the plot.
    ax: plt.Axes,
        The axes of the plot.
    """

    min_x, max_x = x_range

    # Define a grid to calculate the log probabilities on
    grid_size = 200
    x = torch.linspace(min_x, max_x, grid_size).reshape(-1, 1).to(device)

    # Calculate the probabilities of the normalizing flow distribution
    log_prob = nfm.log_prob(x).to('cpu')
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    # Plot both distributions
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.plot(x, prob.data.numpy())
    sns.histplot(target_samples[:3000], bins=100, stat='density', alpha=0.5, ax=ax)
    #ax.set_aspect('equal', 'box')

    return fig, ax
