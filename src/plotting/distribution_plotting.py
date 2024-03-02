# distribution_plotting.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24

import torch
import matplotlib.pyplot as plt


# Plot target distribution
def plot_progress_2d(target, nfm, device, min_x=-3, max_x=3, min_y=-3, max_y=3):

    grid_size = 200
    xx, yy = torch.meshgrid(torch.linspace(min_x, max_x, grid_size), torch.linspace(min_y, max_y, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)
    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob_target = torch.exp(log_prob)

    # Plot initial posterior distribution
    log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.pcolormesh(xx, yy, prob.data.numpy())
    ax.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
    ax.set_aspect('equal', 'box')

    return fig, ax


