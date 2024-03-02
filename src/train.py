# train.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24


# Real NVP

import yaml
from pathlib import Path
import os

import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb

from src.utils.config_loader import get_architecture_from_config
from src.utils.logger import logger
from src.utils import utils
from src.plotting import distribution_plotting as dsplot

#os.environ["WANDB_MODE"] = "disabled"


def train(config):

    run = wandb.init(project='DimensionalFlows', config=config)
    logger.info('Start Run')


    # Use Apple M1 Chip if available, else check if GPU is available, else train on cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # For debugging purposes, you can force the device to be 'cpu'
    device = torch.device("cpu")

    logger.info(f'Using device: {device}')

    
    # Define flows
    K = 16
    torch.manual_seed(0)

    latent_size = 2
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set target and q0
    target = nf.distributions.TwoModes(2, 0.1)
    q0 = nf.distributions.DiagGaussian(2)

    x_train = utils.rejection_sampling_2d(lambda x: torch.exp(target.log_prob(x)), 1000, -3, 3, -2, 2)

    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)

    # Move model on GPU if available
    nfm = nfm.to(device)
    x_train = x_train.to(device)
    #nfm = nfm.double()

    nfm, criterion, optimizer, scheduler = get_architecture_from_config(nfm, config)

    for it in tqdm(range(config['max_iter'])):
        loss = run_epoch(nfm, x_train, optimizer, config)
        run.log({'loss': loss})

        if (it + 1) % config['show_iter'] == 0:
            fig, ax = dsplot.plot_progress_2d(target, nfm, device)
            run.log({'chart': wandb.Image(fig)})


def run_epoch(nfm, x_train, optimizer, config):

    optimizer.zero_grad()
    if config['annealing']:
        loss = nfm.forward_kld(x_train)
    else:
        loss = nfm.reverse_alpha_div(config['num_samples'], dreg=True, alpha=1)

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()


    return loss.to('cpu').data.numpy()



if __name__ == '__main__':
    # Load the run configurations from the config file
    config = yaml.load(open(Path(__file__).parent.joinpath('train_config.yaml'), 'r'),
                       Loader=yaml.FullLoader)
    train(config)
