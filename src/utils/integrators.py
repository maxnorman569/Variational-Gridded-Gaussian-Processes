# numeric imports
import torch
import numpy as np
# integration imports
import scipy.integrate as integrate
# typing imports
from typing import Callable


def integrate_1d(
        function : Callable[[np.ndarray], np.ndarray],
        mesh : np.ndarray,) -> np.ndarray:
    """ 
    Integrate a function over a 1D mesh. (i.e returns the area under the curve between the knots in the mesh)

    Arguments:
        function (Callable[[np.ndarray], np.ndarray])   : The function to integrate.
        mesh (np.ndarray)                               : The mesh to integrate over.

    Returns:
        np.ndarray                                      : The integrated function over the mesh. (length = mesh.shape[0] - 1)
        np.ndarray                                      : The error of the integration. (length = mesh.shape[0] - 1)
    """
    areas = []
    errors = []
    for i in range(1, len(mesh)):
        result, error = integrate.quad(function, mesh[i-1], mesh[i])
        areas.append(result)
        errors.append(error)
    return np.array(areas), np.array(errors)

    

