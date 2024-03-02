# config_loader.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 02.03.24


# config_loader.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 26.11.23


# utils.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 19.11.23

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def replace_empty_with_none(config, key):
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


def get_architecture_from_config(model, config):
    """
    Returns the model architecture specified in the config file. Specifically, it returns the model, criterion,
    optimizer and scheduler.

    Parameters
    ----------
    model: torch.nn.Module,
        The model to be trained.
    config: dict,
        The config dictionary.

    Returns
    -------
    model: torch.nn.Module,
        The model.
    criterion: torch.nn.Module,
        The loss function for training.
    optimizer: torch.optim.Optimizer,
        The optimizer.
    scheduler: torch.optim.lr_scheduler,
        The scheduler. None if no scheduler is specified.
    """

    ################################## Define Model ##################################


    ################################ Load Pretrained ################################


    ################################## Define Loss ##################################

    if config['criterion'] == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion {config['criterion']} not supported.")

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
                                      verbose=True)
    else:
        raise ValueError(f"Scheduler {config_scheduler['name']} not supported.")

    return model, criterion, optimizer, scheduler
