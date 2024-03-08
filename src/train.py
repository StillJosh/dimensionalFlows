# train.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24


# Real NVP

from pathlib import Path

import normflows as nf
from normflows.core import NormalizingFlow
import torch
import yaml
from tqdm import tqdm

import wandb

from src.models.flows import norm_flow
from src.utils.distributions import JointDistribution
from src.plotting import distribution_plotting as dsplot
from src.utils import utils
from src.utils.config_loader import get_architecture_from_config, get_prior_from_config
from src.utils.logger import logger

# Disable wandb for debugging
#os.environ["WANDB_MODE"] = "disabled"


def train():
    """ Load the model and dataset and train the normalizing flow. """

    # Use Apple M1 Chip if available, else check if GPU is available, else train on cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # For debugging purposes, you can force the device to be 'cpu'
    device = torch.device("cpu")

    logger.info(f'Using device: {device}')

    # Set target and base distribution
    target = nf.distributions.TwoModes(2, 0.1)
    q0 = get_prior_from_config(config, run)

    # Sample from the target distribution
    x_train = utils.rejection_sampling_2d(lambda x: torch.exp(target.log_prob(x)), 1000, -3, 3, -2, 2)

    # Define the normalizing flow model
    nfm = norm_flow(q0, 2, K=16)

    # Move model on GPU if available
    nfm = nfm.to(device)
    x_train = x_train.to(device)


    optimizer, scheduler = get_architecture_from_config(nfm, config)

    # Iterate over the specified epochs, log the loss and regularly plot the current distribution as a contour plot
    for it in tqdm(range(config['max_iter'])):
        loss = run_epoch(nfm, x_train, optimizer, config)
        run.log({'loss': loss})

        if (it + 1) % 1 == 0:  # config['show_iter'] == 0:
            if x_train.shape[1] == 1:
                fig, ax = dsplot.plot_progress_1d(x_train, nfm, device)
                run.log({'chart': wandb.Image(fig)})

            elif x_train.shape[1] == 2:
                fig, ax = dsplot.plot_progress_2d(target, nfm, device)
                run.log({'chart': wandb.Image(fig)})

    torch.save(nfm, 'model.pth')
    run.log_model('model.pth')


def run_epoch(nfm: NormalizingFlow, x_train: torch.tensor, optimizer: torch.optim.Optimizer) -> float:
    """
    Run a single epoch of training.

    Parameters
    ----------
    nfm: NormalizingFlow,
        The normalizing flow model.
    x_train: torch.tensor,
        The training data.
    optimizer: torch.optim.Optimizer,
        The optimizer.

    Returns
    -------
    float,
        The loss of the model in this epoch.
    """


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
    torch.manual_seed(0)

    # Load the run configurations from the config file
    config = yaml.load(open(Path(__file__).parent.joinpath('train_config.yaml'), 'r'),
                       Loader=yaml.FullLoader)

    # Start the wandb run
    run = wandb.init(project='DimensionalFlows', config=config)
    logger.info('Start Run')

    train()
