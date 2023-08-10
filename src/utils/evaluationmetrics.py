# numeric imports
import torch
import numpy as np


def mean_squared_error(
        true : torch.Tensor, 
        pred : torch.Tensor) -> torch.Tensor:
    """ MSE """
    # assert tensors are 2D, and have the same shape
    assert len(true.shape) == 2, "true tensor must be 2D, got {}D".format(len(true.shape))
    assert len(pred.shape) == 2, "pred tensor must be 2D, got {}D".format(len(pred.shape))
    assert true.shape == pred.shape, "true and pred must have the same shape, got {} and {}".format(true.shape, pred.shape)
    # compute mse
    mse = torch.mean((true - pred)**2)
    return mse

def mean_absolute_error(
        true : torch.Tensor,
        pred : torch.Tensor) -> torch.Tensor:
    """ MAE """
    # assert tensors are 2D, and have the same shape
    assert len(true.shape) == 2, "true tensor must be 2D, got {}D".format(len(true.shape))
    assert len(pred.shape) == 2, "pred tensor must be 2D, got {}D".format(len(pred.shape))
    assert true.shape == pred.shape, "true and pred must have the same shape, got {} and {}".format(true.shape, pred.shape)
    # compute mae
    mae = torch.mean(torch.abs(true - pred))
    return mae

def root_mean_squared_error(
        true : torch.Tensor,
        pred : torch.Tensor) -> torch.Tensor:
    """ RMSE """
    # assert tensors are 2D, and have the same shape
    assert len(true.shape) == 2, "true tensor must be 2D, got {}D".format(len(true.shape))
    assert len(pred.shape) == 2, "pred tensor must be 2D, got {}D".format(len(pred.shape))
    assert true.shape == pred.shape, "true and pred must have the same shape, got {} and {}".format(true.shape, pred.shape)
    # compute rmse
    rmse = torch.sqrt(torch.mean((true - pred)**2))
    return rmse

def r_squared(
        true : torch.Tensor,
        pred : torch.Tensor) -> torch.Tensor:
    """ R^2 """
    # assert tensors are 2D, and have the same shape
    assert len(true.shape) == 2, "true tensor must be 2D, got {}D".format(len(true.shape))
    assert len(pred.shape) == 2, "pred tensor must be 2D, got {}D".format(len(pred.shape))
    assert true.shape == pred.shape, "true and pred must have the same shape, got {} and {}".format(true.shape, pred.shape)
    # compute r^2
    RSS = torch.sum((true - pred)**2)
    TSS = torch.sum((true - torch.mean(true))**2)
    r2 = 1 - (RSS/TSS)
    return r2

    


