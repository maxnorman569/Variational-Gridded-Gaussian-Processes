import torch
import math
from abc import ABC, abstractmethod

class FourierBasis(ABC):
    """ Constructs a Fourier Basis for Variational Fourier Features """

    def __init__(self, n_frequencies: int, a: float, b: float, lengthscale: float):
        self.M = n_frequencies  # the number of omegas to use
        self.a = a
        self.b = b
        self.lengthscale = lengthscale
        self.omegas = (2 * torch.pi) * torch.arange(self.M + 1) / (b - a)

    def _domain_maks(self, X: torch.Tensor) -> torch.Tensor:
        """ Returns a bool mask for input points of X lying in the domain [a, b] """
        # identify the points in the domain
        domain_mask = torch.logical_and(X >= self.a, X < self.b)
        return domain_mask

    @abstractmethod
    def _real_outside_domain(self, x_outside: torch.Tensor) -> torch.Tensor:
        """ Evaluates the real part of the Fourier Basis at a point outside the domain [a, b] """
        pass

    @abstractmethod
    def _imaginary_outside_domain(self, x_outside: torch.Tensor) -> torch.Tensor:
        pass

    def _real_inside_domain(self, x_inside: torch.Tensor) -> torch.Tensor:
        """ Evaluates the real part of the Fourier Basis at a point inside the domain [a, b] """
        # evaluate the basis at x
        cosines = torch.cos(self.omegas * (x_inside - self.a))
        return cosines

    def _imaginary_inside_domain(self, x_inside: torch.Tensor) -> torch.Tensor:
        """ Evaluates the imaginary part of the Fourier Basis at a point inside the domain [a, b] """
        # evaluate the basis at x
        sines = torch.sin(self.omegas[1:] * (x_inside - self.a))
        return sines

    def _inside_domain(self, x_inside: torch.Tensor) -> torch.Tensor:
        """ Evaluates the basis for points inside the domain [a, b] """
        # evaluate the basis at x
        real_basis = self._real_inside_domain(x_inside)
        imaginary_basis = self._imaginary_inside_domain(x_inside)
        basis_at_x = torch.cat((real_basis, imaginary_basis))
        return basis_at_x

    def _outside_domain(self, x_outside: torch.Tensor) -> torch.Tensor:
        """ Evaluates the basis for points outside the domain [a, b] """
        # evaluate the basis at x
        real_basis = self._real_outside_domain(x_outside)
        imaginary_basis = self._imaginary_outside_domain(x_outside)
        basis_at_x = torch.cat((real_basis, imaginary_basis))
        return basis_at_x

    def __call__(self, X: float) -> torch.Tensor:
        """ Evaluates the Fourier Basis at a point x """
        domain_mask = self._domain_maks(X)
        basis_at_x = []
        for i, x in enumerate(X):
            # evaluate the basis at x
            if domain_mask[i]:
                basis_at_x.append(self._inside_domain(x))
            else:
                basis_at_x.append(self._outside_domain(x))
        return torch.vstack((basis_at_x)).T


class FourierBasisMatern12(FourierBasis):
    """ Constructs a Fourier Basis for Variational Fourier Features for a Matern 1/2 Kernel """

    def __init__(self, n_frequencies: int, a: float, b: float, lengthscale: float):
        super().__init__(n_frequencies, a, b, lengthscale)
        self.lmbda = 1 / lengthscale

    def _real_outside_domain(self, x_outside: torch.Tensor) -> torch.Tensor:
        """ Evaluates the real part of the Fourier Basis at a point outside the domain [a, b] """
        # construct real basis for x outside domain
        r = min(abs(x_outside - self.a), abs(x_outside - self.b))
        phi_m = torch.exp(-self.lmbda * r)
        real_phi = torch.ones(self.M + 1) * phi_m
        return real_phi

    def _imaginary_outside_domain(self, x_outside: torch.Tensor) -> torch.Tensor:
        """ Evaluates the imaginary part of the Fourier Basis at points outside the domain [a, b] """
        return torch.zeros(self.M)
    

class FourierBasisMatern32(FourierBasis):
    """ Constructs a Fourier Basis for Variational Fourier Features for a Matern 3/2 Kernel """

    def __init__(self, n_frequencies : int, a : float, b : float, lengthscale : float):
        super().__init__(n_frequencies, a, b, lengthscale) 
        self.lmbda = math.sqrt(3) / lengthscale

    def _real_outside_domain(self, x_outside : torch.Tensor) -> torch.Tensor:
        """ Evaluates the real part of the Fourier Basis at a point outside the domain [a, b] """   
        # construct the real basis for x outside domain
        r = min(abs(x_outside - self.a), abs(x_outside - self.b))
        phi_m = (1 + self.lmbda * r) * torch.exp(-self.lmbda * r)
        real_phi = torch.ones(self.M + 1) * phi_m
        return real_phi
    
    def _imaginary_outside_domain(self, x_outside : torch.Tensor) -> torch.Tensor:
        """ Evaluates the imaginary part of the Fourier Basis at points outside the domain [a, b] """
        # construct the imaginary basis for x outside domain
        s = 1 if x_outside < self.a else -1
        r = min(abs(x_outside - self.a), abs(x_outside - self.b))
        imaginary_phi = s * r * self.omegas[1:] * torch.exp(-self.lmbda * r)
        return imaginary_phi


class FourierBasisMatern52(FourierBasis):
    """ Constructs a Fourier Basis for Variational Fourier Features for a Matern 3/2 Kernel """

    def __init__(self, n_frequencies : int, a : float, b : float, lengthscale : float):
        super().__init__(n_frequencies, a, b, lengthscale) 
        self.lmbda = math.sqrt(5) / lengthscale

    def _real_outside_domain(self, x_outside : torch.Tensor) -> torch.Tensor:
        """ Evaluates the real part of the Fourier Basis at a point outside the domain [a, b] """   
        # construct the real basis for x outside domain
        r = min(abs(x_outside - self.a), abs(x_outside - self.b))
        real_phi = (1 + self.lmbda * r + (1/2) * ((self.lmbda ** 2) + (self.omegas ** 2) * (r ** 2))) * torch.exp(-self.lmbda * r)
        return real_phi
    
    def _imaginary_outside_domain(self, x_outside : torch.Tensor) -> torch.Tensor:
        """ Evaluates the imaginary part of the Fourier Basis at points outside the domain [a, b] """
        # construct the imaginary basis for x outside domain
        s = 1 if x_outside < self.a else -1
        r = min(abs(x_outside - self.a), abs(x_outside - self.b))
        imaginary_phi = s * r * self.omegas[1:] * (1 + self.lmbda * r) * torch.exp(-self.lmbda * r)
        return imaginary_phi

    