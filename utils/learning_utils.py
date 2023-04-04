import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def weighted_ce_loss(output, target, weight=None):
    """
    :param output:
    :param target:
    :return:
    """
    grid_loss_values = target * torch.log(output + 1e-5)
    if weight is None:
        return -torch.mean(grid_loss_values)
    expanded = weight.expand_as(target)
    return -torch.mean(expanded * grid_loss_values)


def weighted_dice_loss(output, target, weight=None):
    """
    Return weighted Dice loss for the HD branch
    :param output:
    :param target:
    :return:
    """
    grid_loss_values = (2 * target * output + 0.01) / (target + output + 0.01)
    if weight is None:
        return -torch.mean(grid_loss_values)
    expanded = weight.expand_as(target)
    return -torch.mean(expanded * grid_loss_values)


def get_split_dataloaders(dataset,
                          split_train=0.7,
                          split_valid=0.85,
                          **kwargs):
    n = len(dataset)

    np.random.seed(0)
    torch.manual_seed(0)

    train_index, valid_index = int(split_train * n), int(split_valid * n)
    indices = list(range(n))

    train_indices = indices[:train_index]
    valid_indices = indices[train_index:valid_index]
    test_indices = indices[valid_index:]

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(dataset=train_set, worker_init_fn=np.random.seed, **kwargs)
    valid_loader = DataLoader(dataset=valid_set, **kwargs)
    test_loader = DataLoader(dataset=test_set, **kwargs)
    return train_loader, valid_loader, test_loader