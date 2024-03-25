# config_loader.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24

from typing import Optional

import normflows as nf
import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.dimensionalFlow import DimensionalFlow
from src.utils.distributions import JointDistribution
from src.models import flows as fl



def replace_empty_with_none(config: dict, key: str) -> str or None:
    """
    Returns None if a config entry is not available.

    Parameters
    ----------
    config: dict,
        The config dictionary.
    key: str,
        The key to look for in the config dictionary.

    Returns
    -------
    value: str or None,
        The value of the key in the config dictionary or None if the key is not available.
    """

    try:
        value = config[key]
    except KeyError:
        value = None

    return value


def get_prior_from_config(config: dict, run: Optional[wandb.run]) -> nf.distributions.BaseDistribution:
    """
    Returns the prior distribution specified in the config file.

    Parameters
    ----------
    config: dict,
        The config dictionary.
    run: wandb.run,
        The current wandb run.

    Returns
    -------
    q0: nf.distributions.BaseDistribution,
        The prior distribution.
    """

    prior_config = config['prior']
    if prior_config['name'] == 'gaussian':
        q0 = nf.distributions.DiagGaussian(prior_config['dim'], trainable=prior_config['trainable'])

    # Load prior as a pretrained flow from wandb
    elif prior_config['name'] == 'pretrained':
        artifact = run.use_artifact('jay-son/DimensionalFlows/' + prior_config['path'], type='model')
        artifact_dir = artifact.download()
        q0 = JointDistribution([torch.load(artifact_dir + '/model.pth').eval(), nf.distributions.DiagGaussian(1)],
                               [1, 2])

    else:
        raise ValueError(f"Prior {prior_config['name']} not supported.")

    return q0


def get_architecture_from_config(model, config):
    """
    Returns the model architecture specified in the config file. Specifically, it returns the optimizer and scheduler.

    Parameters
    ----------
    model: torch.nn.Module,
        The model to be trained.
    config: dict,
        The config dictionary.

    Returns
    -------
    optimizer: torch.optim.Optimizer,
        The optimizer.
    scheduler: torch.optim.lr_scheduler,
        The scheduler. None if no scheduler is specified.
    """

    ################################## Define Model ##################################


    ################################## Define Loss ##################################


    ################################ Define Optimizer ################################

    optimizer_config = config['optimizer']
    if optimizer_config['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config['lr'],
                                     weight_decay=optimizer_config['weight_decay'])
    elif optimizer_config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_config['lr'],
                                    weight_decay=optimizer_config['weight_decay'],
                                    momentum=0.9)
    else:
        raise ValueError(f"Optimizer {optimizer_config['name']} not supported.")

    ################################## Define Scheduler ###############################

    config_scheduler = config['scheduler']
    if config_scheduler['name'] is None:
        scheduler = None
    elif config_scheduler['name'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config_scheduler['lr_factor'],
                                      patience=config_scheduler['lr_patience'],
                                      min_lr=config_scheduler['min_lr'])
    else:
        raise ValueError(f"Scheduler {config_scheduler['name']} not supported.")

    return optimizer, scheduler


def get_model_from_config(config: dict) -> nf.flows.base.Flow:
    """
    Returns the normalizing flow model specified in the config file.

    Parameters
    ----------
    config: dict,
        The config dictionary.
    run: wandb.run,
        The current wandb run.

    Returns
    -------
    nfm: nf.models.NormalizingFlow,
        The normalizing flow model.
    """

    model_config = config['model']
    if len(model_config) > 1:
        flows = [getattr(fl, flow['name'])(nf.distributions.DiagGaussian(flow['dim'], trainable=False), **flow['params']) for flow
                 in model_config.values()]

        flow_dims = [flow['dim'] for flow in model_config.values()]
        nfm = DimensionalFlow(flows, flow_dims)

    else:
        flow_config = list(model_config.values())[0]
        nfm = getattr(fl, flow_config['name'])(nf.distributions.DiagGaussian(flow_config['dim'], trainable=False),
                                                      **flow_config['params'])

    return nfm
