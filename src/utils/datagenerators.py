# numeric imports
import torch 
import numpy as np
# typing imports
from typing import Callable, Tuple


def gen_1d(
        fun : Callable[[torch.Tensor], torch.Tensor], 
        leftlim : float, 
        rightlim : float,
        nobs : int,
        randomspacing : bool = False) -> torch.Tensor:
    """ 
    Generate 1d data for a given function 

    Arguments:
        fun (Callable[[torch.Tensor], torch.Tensor])    : function to generate data from
        leftlim (float)                                 : left limit of the data
        rightlim (float)                                : right limit of the data
        nobs (int)                                      : number of observations
        randomspacing (bool)                            : spacing between observations (True -> random, False -> evenly spaced)

    Returns:
        data (torch.Tensor)                             : 1d data points (X)
        torch.Tensor (nobs x 1)                         : 1d data (y)
    """
    if randomspacing:
        # generate data
        domain = torch.rand(nobs) * (rightlim - leftlim) + leftlim 
    else:
        domain = torch.linspace(leftlim, rightlim, nobs)
    data = fun(domain)
    return domain, data


def gen_2d(
        func : Callable[[np.ndarray, np.ndarray], np.ndarray],
        x1lims : Tuple[float, float],
        x2lims : Tuple[float, float],
        nobs : int,
        randomspacing : bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Generate 2d data for a given function 

    Arguments:
        func (Callable[[np.Tensor], torch.Tensor])      : numpy function to generate data from
        x1lims (Tuple[float, float])                    : left and right limits of the x1 data
        x2lims (Tuple[float, float])                    : left and right limits of the x2 data
        nobs (int)                                      : number of observations
        randomspacing (bool)                            : spacing between observations (True -> random, False -> evenly spaced)

    Returns:
        data (torch.Tensor)                             : 1d data points
        torch.Tensor (nobs x 1)                         : 1d data
    """
    # get data limts
    x1_min, x1_max = x1lims
    x2_min, x2_max = x2lims
    # generate input data
    if randomspacing:
        # generate data
        domain_x1 = np.random.random(nobs) * (x1_max - x1_min) + x1_min
        domain_x2 = np.random.random(nobs) * (x2_max - x2_min) + x2_min
        
    else:
        domain_x1 = np.linspace(x1_min, x1_max, nobs)
        domain_x2 = np.linspace(x2_min, x2_max, nobs)
    # make meshgrid
    X1, X2 = np.meshgrid(domain_x1, domain_x2)
    # generate data
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    y = func(X[:, 0], X[:, 1])
    return X, y