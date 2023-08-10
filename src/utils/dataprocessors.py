import torch

def min_max_scaling(tensor : torch.Tensor, min = None, max = None):
    """
    Min-max scaling for a given tensor

    Arguments:
        tensor (torch.Tensor)   : tensor to scale
        min (float)             : minimum value of the tensor
        max (float)             : maximum value of the tensor

    Returns:
        tensor (torch.Tensor)   : scaled tensor
        min (float)             : minimum value of the tensor
        max (float)             : maximum value of the tensor
    """
    # get min and max
    if min is None:
        min = torch.min(tensor)
    if max is None:
        max = torch.max(tensor)
    # scale tensor
    tensor = (tensor - min) / (max - min)
    return tensor, min, max

def min_max_inverse(tensor : torch.Tensor, min : float, max : float):
    """
    Inverse min-max scaling for a given tensor

    Arguments:
        tensor (torch.Tensor)   : tensor to scale
        min (float)             : minimum value of the tensor
        max (float)             : maximum value of the tensor

    Returns:
        tensor (torch.Tensor)   : scaled tensor
    """
    tensor = tensor * (max - min) + min
    return tensor

def z_scaling(tensor,):
    """
    Z-scaling for a given tensor

    Arguments:
        tensor (torch.Tensor)   : tensor to scale

    Returns:
        tensor (torch.Tensor)   : scaled tensor
        mean (float)            : mean of the tensor
        std (float)             : standard deviation of the tensor
    """
    # get mean and std
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    # scale tensor
    tensor = (tensor - mean) / std
    return tensor, mean, std

def z_inverse(tensor : torch.Tensor, mean : float, std : float):
    """
    Inverse Z-scaling for a given tensor

    Arguments:
        tensor (torch.Tensor)   : tensor to scale
        mean (float)            : mean of the tensor
        std (float)             : standard deviation of the tensor

    Returns:
        tensor (torch.Tensor)   : scaled tensor
    """
    tensor = tensor * std + mean
    return tensor