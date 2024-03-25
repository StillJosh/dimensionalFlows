# train.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24


# Real NVP
import sys

sys.path.append('.')

from pathlib import Path

from normflows.core import NormalizingFlow
import torch
import yaml
from tqdm import tqdm

import wandb

from src.plotting import distribution_plotting as dsplot
from src.utils.config_loader import get_architecture_from_config, get_model_from_config
import src.utils.utils as utils
from src import datasets as ds
from src.utils.logger import logger

from torch.utils.data import DataLoader
from time import time

# Disable wandb for debugging
#os.environ["WANDB_MODE"] = "disabled"


def train():
    """ Load the model and dataset and train the normalizing flow. """

    # Define the normalizing flow model
    nfm = get_model_from_config(config)

    # Move model on GPU if available
    nfm = nfm.to(device)

    # Load the dataset and create dataloader
    dataset = getattr(ds, config['dataset']['name'])(**config['dataset']['params'])
    dl_train = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    if hasattr(nfm, 'flow_dims'):
        utils.reduce_dimension(dataset, nfm, config)

    optimizer, scheduler = get_architecture_from_config(nfm, config)

    start_time = time()
    # Iterate over the specified epochs, log the loss and regularly plot the current distribution as a contour plot
    for it in tqdm(range(config['epochs'])):

        # Validation Loss
        utils.epoch_scheduler(dl_train, nfm, config['epochs'], config)
        dl_train.dataset.phase = 'val'
        loss = run_epoch(nfm, dl_train, optimizer, phase='val')
        run.log({'val_loss': loss}, step=it )
        run.log({'val_loss': loss, 'time': time() - start_time}, step=it)

        # Training Loss
        utils.epoch_scheduler(dl_train, nfm, it, config)
        dl_train.dataset.phase = 'train'
        loss = run_epoch(nfm, dl_train, optimizer, phase='train')
        scheduler.step(loss)
        run.log({'train_loss': loss}, step=it)


        # Plot status of the model
        if (dl_train.dataset.return_dim <= 2) and ((it % config['show_iter'] == 0) or (it-1 in config['epoch_switch'])):
            fig, ax = dsplot.plot_progress(dl_train, nfm, device)
            run.log({'chart': wandb.Image(fig)}, step=it)


    torch.save(nfm, 'model.pth')
    run.log_model('model.pth')


def run_epoch(nfm: NormalizingFlow, dl_train: DataLoader, optimizer: torch.optim.Optimizer, phase='train') -> float:
    """
    Run a single epoch of training.

    Parameters
    ----------
    nfm: NormalizingFlow,
        The normalizing flow model.
    dl_train: DataLoader,
        The training data.
    optimizer: torch.optim.Optimizer,
        The optimizer.

    Returns
    -------
    float,
        The loss of the model in this epoch.
    """

    epoch_loss = 0
    for x_train in dl_train:
        x_train = x_train.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            loss = nfm.forward_kld(x_train)

        if phase == 'train':
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            #nf.utils.update_lipschitz(nfm, 50)

        epoch_loss += loss.item()

    return epoch_loss / len(dl_train)


if __name__ == '__main__':
    torch.manual_seed(42)

    # Use Apple M1 Chip if available, else check if GPU is available, else train on cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # For debugging purposes, you can force the device to be 'cpu'
    device = torch.device("cpu")

    logger.info(f'Using device: {device}')


    # Load the run configurations from the config file
    config = yaml.load(open(Path(__file__).parent.joinpath('train_config.yaml'), 'r'),
                       Loader=yaml.FullLoader)

    # Start the wandb run
    run = wandb.init(project='DimensionalFlows', config=config)
    wandb.define_metric('time')
    wandb.define_metric('val_loss', step_metric='time')
    logger.info('Start Run')

    train()
