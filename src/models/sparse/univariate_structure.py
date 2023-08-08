# numeric imports
import torch
import numpy as np
# GP imports
import gpytorch
from src.basis.fourier import FourierBasis, FourierBasisMatern12, FourierBasisMatern32, FourierBasisMatern52
from src.basis.bspline import B0SplineBasis, B1SplineBasis
# misc imports
from abc import ABC, abstractmethod
import linear_operator.operators as operators
# typing imports
from typing import Tuple


class SparseGP(gpytorch.Module, ABC):
    """ 
    Base class for Sparse GP models in 1-dimensional input space 

    Need to set the following attributes in child class:
        - self.kernel (gpytorch.kernels.Kernel) : kernel function
 
    Need to set the following methods in child class:
        - self._Kuf(X : torch.Tensor) -> torch.Tensor : computes the covariance matrix between the inducing points and the training points
        - self._Kuu() -> torch.Tensor : computes the covariance matrix between the inducing points
    """
    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, ) -> 'SparseGP':
        """ 
        constructs a Sparse GP model 
        
        Arguments:
            X (torch.Tensor)    : (n x 1) training inputs
            y (torch.Tensor)    : (n x 1) training targets
        """
        super().__init__()
        # data
        self.train_inputs = (X,)
        self.train_targets = y
        # model components
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.mean = gpytorch.means.ZeroMean()
        # TODO: KERNEL TO BE SET IN CHILD CLASS

    def non_informative_initialise(self, 
                                lmbda: float, 
                                kappa: float) -> None:
        """
        Initialises the model by setting the hyperparameters.

        Arguments:
            x_std (torch.Tensor)    : standard deviation of the input data
            y_std (torch.Tensor)    : standard deviation of the output data
            lmbda (float)           : lengthscale hyperparameter (usually in range [1, 10], where lmbda ~ 1 favours linear functions and lmbda ~ 10 gives more flexibility)
            kappa (float)           : noise hyperparameter (we expect kappa to be in the range of [2, 100], where kappa = sqrt(signal variance / noise variance)) i.e how squiggly the function is vs the variance of the residuals. Therefore large kappas mean high signal variance (very squiggly) relative to low residual variance (passing through points)
        
        Returns:
            None
        """
        # training data
        X = self.train_inputs[0]
        y = self.train_targets
        # initialise hyperparameters
        self.kernel.outputscale = y.var()
        self.likelihood.noise = self.kernel.outputscale / (kappa ** 2)
        self.kernel.base_kernel.lengthscale[0]  = (X.std() / lmbda)
    
    def informative_initialise(self,
                            prior_amplitude : float,
                            lmbda: float,) -> None:
        """
        Initialises the model hyperparameters based on prior knolwedge of the plausible function amplitudes.

        Arguments:
            prior_amplitude (float) : amplitude of the function prior
            lmbda (float)           : lengthscale hyperparameter (lambda ~ 1 -> favours linear function, lambda ~ 10 -> favours highly non-linear functions)

        Returns:
            None
        """
        # training data
        X = self.train_inputs[0]
        y = self.train_targets
        # initialise hyperparameters
        self.kernel.outputscale = (torch.tensor(prior_amplitude) / 2) ** 2
        self.likelihood.noise = y.var() - self.kernel.outputscale
        self.kernel.base_kernel.lengthscale[0]  = (X.std() / lmbda)

    @abstractmethod
    def _Kuu(self,) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing features: [Kuu]_{ij} = Cov[u_i, u_j]
        """
        pass
    
    @abstractmethod
    def _Kuf(self, 
            x : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the Kuf matrix between the inducing features and the latent function: Kuf]_{ij} = Cov[u_i, f(x_j)]
        """
        pass

    def _sigma(self,) -> torch.Tensor:
        """ 
        Computes [Kuu + noisesigma^{-2} Kuf Kuf^T] 
        
        Arguments:
            None

        Returns:
            sigma (torch.tensor)    : n x n matrix, [Kuu + noisesigma^{-2} Kuf Kuf^T] 
        """
        X = self.train_inputs[0]
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # compute matrices
        Kuu = self._Kuu()
        Kuf = self._Kuf(X)
        return Kuu + (Kuf @ Kuf.T) / noisesigma
        
    def _conditional_mu(self, 
                        x_star : torch.Tensor) -> torch.Tensor:
        """ computes the conditional mean for the GP posterior over the test inputs x_star
        - cond_mu = noisesigma^{-2} ku()^T sigma^{-1} Kuf y 
        
        Arguments:
            x_star (torch.tensor)   : n^* x 1, test inputs to predict at

        Returns:
            cond_mu (torch.tensor)  : n^* x n^*, conditional covariance of posterior of test inputs
        """
        # training inputs
        X = self.train_inputs[0]
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # compute matrices
        Kuf = self._Kuf(X)
        Kuf_star = self._Kuf(x_star)
        sigma = gpytorch.lazify(self._sigma())
        # compute the conditional mean
        cond_mu = (Kuf_star.T @ sigma.inv_matmul(Kuf) @ self.train_targets) / noisesigma
        return cond_mu

    def _conditional_cov(self, 
                         x_star : torch.Tensor) -> torch.Tensor:
        """ 
        computes the conditional covariance for the GP posterior over the test inputs x_star
         - cond_cov = k(,) + ku()^T sigma^{-1} ku() - ku()^t Kuu^{-1} ku() 

        Arguments:
            x_star (torch.tensor)   : test inputs to predict at

        Returns:
            cond_cov (torch.tensor) : conditional covariance of posterior of test inputs
        """
        # compute matrices
        Kuu = gpytorch.lazify(self._Kuu())
        Kuf_star = self._Kuf(x_star)
        sigma = gpytorch.lazify(self._sigma())
        # compute the terms in the conditional covariance
        term1 = self.kernel(x_star).evaluate()
        term2 = Kuf_star.T @ sigma.inv_matmul(Kuf_star)
        term3 = Kuf_star.T @ Kuu.inv_matmul(Kuf_star)
        # compute the conditional covariance
        cond_cov = term1 + term2 - term3
        return cond_cov
    
    def prior(self,
              x_star : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the prior distribution N(f^* | prior_mu, prior_cov) for the test inputs

        Arguments:
            x_star (torch.tensor)   : the test inputs

        Returns:
            prior (MultivariateNormal) : prior over the test inputs
        """
        prior_mu = self.mean(x_star)
        prior_cov = self.kernel(x_star).evaluate()
        return gpytorch.distributions.MultivariateNormal(prior_mu, prior_cov)
    
    def posterior(self, 
                x_star : torch.Tensor) -> torch.Tensor:
        """ 
        computes the approximate posterior distribution N(f^* | cond_mu, cond_cov) for the test inputs given a Gaussian variational distribution
        - cond_mu - given by self._conditional_mu()
        - cond_cov - given by self._conditional_cov()

        Arguments:
            x_star (torch.tensor)   : the test inputs

        Returns:
            posterior (MultivariateNormal) : approximate posterior over the test inputs
        """
        # training inputs
        X = self.train_inputs[0]
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # compute matrices
        Kuu = gpytorch.lazify(self._Kuu())
        Kuf = self._Kuf(X)
        Kuf_star = self._Kuf(x_star)
        sigma = gpytorch.lazify(self._sigma())
        # conditional mean
        cond_mu = (Kuf_star.T @ sigma.inv_matmul(Kuf) @ self.train_targets) / noisesigma
        # conditional covariance
        term1 = self.kernel(x_star).evaluate()
        term2 = Kuf_star.T @ sigma.inv_matmul(Kuf_star)
        term3 = Kuf_star.T @ Kuu.inv_matmul(Kuf_star)
        cond_cov = gpytorch.lazify(term1 + term2 - term3)
        # compute the approximate posterior
        posterior = gpytorch.distributions.MultivariateNormal(cond_mu, cond_cov)
        return posterior
    
    def posterior_predictive(self, 
                            x_star : torch.Tensor) -> torch.Tensor:
        """ 
        computes the posterior predictive distribution for the test inputs given a Gaussian variational distribution
        
        Arguments:
            x_star (torch.tensor)   : the test inputs

        Returns:
            posterior_predictive    : Posterior predictive over the test inputs
        """
        # get posterior
        posterior = self.posterior(x_star)
        # pass it through the likelihood
        posterior_predictive = self.likelihood(posterior)
        return posterior_predictive
    
    def _elbo(self,) -> torch.Tensor:
        """ 
        computest the Evidence Lower Bound (ELBO) for a Gaussian variational distribution, with Gaussian Likelihood, at optimal m, and S

        Arguments:
            None

        Returns:
            elbo (torch.tensor)     : the elbow 
        """
        # training data
        X = self.train_inputs[0]
        y = self.train_targets.squeeze()
        # get noise sigma
        noise_sigma = self.likelihood.noise[0]
        # compute matrices
        Kuu = gpytorch.lazify(self._Kuu())
        Kuf = self._Kuf(X)
        Kff = self.kernel(X).evaluate()
        # 'approximate prior' term
        approx_prior = Kuf.T @ Kuu.inv_matmul(Kuf)
        # evidence term
        evidence_mean = self.mean(X)
        evidence_covariance = approx_prior + (noise_sigma * torch.eye(X.size(0)))
        evidence_term = gpytorch.distributions.MultivariateNormal(evidence_mean, evidence_covariance).log_prob(y)
        # trace term
        trace_term = torch.trace(Kff - approx_prior) / (2 * noise_sigma)
        # elbo
        elbo = evidence_term - trace_term
        return elbo
    

####################################################################################################
#                                                                                                  #
#                                            SVGP                                                  #
#                                                                                                  #
####################################################################################################

# Parent class for SVGP
class SVGP(SparseGP):

    """ Sparse Variational Gaussian Process (SVGP) """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 Z : torch.Tensor) -> 'SVGP':
        """
        Arguments:
            train_x (torch.tensor)      : (n x 1) training inputs 
            train_y (torch.tensor)      : (n x 1) training targets
            inducing_z (torch.tensor)   : (m x 1) inducing points
        """
        super().__init__(X, y)
        # register inducing points
        self.register_parameter("Z", torch.nn.Parameter(torch.zeros(Z.shape)))
        self.initialize(Z=Z)
        # TODO: kernel to be set in child

    def _Kuu(self,) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing features
        - [Kuu]_{ij} = Cov[u_i, u_j]

        Arguments:
            None

        Returns:
            Kuu (torch.tensor)  : n x n matrix of inducing feature covariances
        """
        # compute Kuu for each dimension
        Kuu = self.kernel(self.Z).evaluate()
        return Kuu
    
    def _Kuf(self, 
            x : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the Kuf matrix between the inducing features and the latent function
        - [Kuf]_{ij} = Cov[u_i, f(x_j)]

        Arguments:
            x (torch.tensor)    : indicies for function evaluations

        Returns:
            Kuu (torch.tensor)  : n x n matrix of inducing feature covariances
        """
        Kuf = self.kernel(self.Z, x).evaluate()
        return Kuf
    

# Child classes for SVGP
class Matern12SVGP(SVGP):
    """ SVGP with Matern 1/2 kernel """
    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 Z : torch.Tensor):
        super().__init__(X, y, Z)
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2))


# Child classes for SVGP
class Matern32SVGP(SVGP):
    """ SVGP with Matern 3/2 kernel """
    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 Z : torch.Tensor):
        super().__init__(X, y, Z)
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=3/2))

    
# Child classes for SVGP
class Matern52SVGP(SVGP):
    """ SVGP with Matern 5/2 kernel """
    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 Z : torch.Tensor):
        super().__init__(X, y, Z)
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=5/2))


####################################################################################################
#                                                                                                  #
#                                            VFF                                                   #
#                                                                                                  #
####################################################################################################

# Parent class for VFFGP
class VFFGP(SparseGP, ABC):

    """ Variational Fourier Features Gaussian Process (VFFGP) """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 nfrequencies : int, 
                 dim1lims : Tuple[float, float]) -> 'VFFGP':
        """ 
        Arguments:
            train_x (torch.tensor)          : (n x 1) training inputs 
            train_y (torch.tensor)          : (n x 1) training targets
            nfrequencies (int)              : number of frequencies to use in the Fourier basis
            dim1lims (Tuple[float, float])  : lower and upper bounds of the input space (min, max)
        """
        super().__init__(X, y)
        # parameters
        self.nfrequencies = nfrequencies
        self.alim = dim1lims[0]
        self.blim = dim1lims[1]
        # TODO: kernel to be set in Child
        # TODO: set the basis 

    @abstractmethod
    def spectral_density(self, ) -> torch.Tensor:
        """
        Computes the spectra density corersponding to the Matérn [1, 3, 5]/2 covariances

        Arguments:
            omega (torch.Tensor)    : frequency
            sigma (float)           : amplitude hyperparameter
            lengthscale (float)     : lengthscale hyperparameter (lmbda = sqrt(3) / original lengthscale)

        Returns:
            (torch.Tensor)          : spectral density
        """
        pass
    

# Child classes for VFFGP
class Matern12VFFGP(VFFGP):
    
    """ VFFGP with Matern 1/2 kernel. """

    def __init__(self, 
                X : torch.Tensor, 
                y : torch.Tensor, 
                nfrequencies : int, 
                dim1lims : Tuple[float, float]) -> 'VFFGP':
        super().__init__(X, y, nfrequencies, dim1lims)
        self.omegas = FourierBasisMatern12(self.nfrequencies, self.alim, self.blim, 1.).omegas
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2))

    def spectral_density(self, ) -> torch.Tensor:
        """
        Computes the spectra corersponding to the Matérn 1/2 covariances

        Arguments:
            omega (torch.Tensor)    : frequency
            scalesigma (float)      : amplitude hyperparameter
            lengthscale (float)     : lengthscale hyperparameter

        Returns:
            (torch.Tensor)          : spectral density
        """
        # omegas
        omegas = self.omegas
        # hyperparameters
        scalesigma = self.kernel.outputscale
        lengthscale = self.kernel.base_kernel.lengthscale
        # get lamnda
        lmbda = 1 / lengthscale.squeeze()
        # compute spectral density
        numerator = 2 * scalesigma * lmbda
        denominator = (lmbda ** 2) + (omegas ** 2)
        spectral_density = numerator / denominator
        return spectral_density
    
    def _alpha(self, omegas) -> torch.Tensor:
        """
        Computes alpha half of the Kuu representation for the Matérn 1/2 covarainces

        Arguments:
            omegas (torch.Tensor)   : frequency (! omegas[0] = 0 !)
            scalesigma (float)      : amplitude hyperparameter
            lengthscale (float)     : lengthscale hyperparameter
            a (float)               : lower bound of the input space
            b (float)               : upper bound of the input space

        Returns:
            (torch.Tensor)          : alpha
        """
        a = self.alim
        b = self.blim
        # check that omegas[0] = 0
        assert omegas[0] == 0, "The first element of omegas must be 0"
        # compute the inverse spectral density
        S_inv = 1 / self.spectral_density()
        # compute the alpha half
        alpha = ((b - a) / 2) * torch.cat([2 * S_inv[0][None], S_inv[1:], S_inv[1:]])
        return alpha
    
    def _beta(self, omegas) -> torch.Tensor:
        """
        Computes the beta half of the Kuu representation for the Matérn 1/2 covarainces

        Arguments:
            omega (torch.Tensor)    : frequency
            sigma (float)           : amplitude hyperparameter

        Returns:
            (torch.Tensor)          : beta
        """
        scalesigma = self.kernel.outputscale.sqrt()
        # compute the sigma half
        sigma_half = torch.ones(len(omegas)) / scalesigma
        # compute the zero half
        zero_half = torch.zeros(len(omegas) - 1)
        # compute beta
        beta = torch.cat((sigma_half, zero_half))
        return beta
    
    def _Kuu(self,) -> torch.Tensor:
        """
        Computes the Kuu using the representation given by (62) in the VFF paper for the Matérn 1/2 covarainces

        Arguments:
            omegas (torch.Tensor)   : frequency (! omegas[0] = 0 !)
            scalesigma (float)      : amplitude hyperparameter
            lengthscale (float)     : lengthscale hyperparameter
            a (float)               : lower bound of the input space
            b (float)               : upper bound of the input space

        Returns:
            (torch.Tensor)          : Kuu
        """
        # compute alpha and beta
        alpha = self._alpha(self.omegas)
        beta = self._beta(self.omegas).unsqueeze(-1)
        return operators.DiagLinearOperator(alpha).add_low_rank(beta).to_dense().to(torch.float64) # TODO: this is also wrong, shouldn't have to cast!
    
    def _Kuf(self, 
                    x : float,) -> torch.Tensor:
        """ 
        Returns the cross-covariance between the domains 

        Arguments:
            fourier_basis (FourierBasis)    : Fourier Basis
            x (float)                       : point in the domain to evaluate the cross-covariance at

        Returns:
            (torch.Tensor)                  : cross-covariance
        """
        lengthscale = self.kernel.base_kernel.lengthscale.squeeze()
        basis = FourierBasisMatern12(self.nfrequencies, self.alim, self.blim, lengthscale)
        return basis(x).to(torch.float64) # TODO: this is wrong, should return a float64 tensor (shouldntt bave to cast here)
    
    

####################################################################################################
#                                                                                                  #
#                                            ASVGP                                                 #
#                                                                                                  #
####################################################################################################

# Parent class for ASVGP
class ASVGP(SparseGP, ABC):

    """ Actually Sparse Variational Gaussian Process (ASVGP) """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 nknots : int, 
                 dim1lims : Tuple[float, float]) -> 'ASVGP':
        """
        Arguments:
            train_x (torch.tensor)          : (n x 1) training inputs 
            train_y (torch.tensor)          : (n x 1) training targets
            nknots (int)                    : number of knots
            dim1lims (Tuple[float, float])  : lower and upper bounds of the input space (min, max)
        """
        super().__init__(X, y)
        # parameters
        self.nknots = nknots
        self.alim = dim1lims[0]
        self.blim = dim1lims[1]
        self.mesh = torch.linspace(self.alim, self.blim, nknots)
        self.delta = self.mesh[1] - self.mesh[0]
        # TODO: set kernel in child

    @abstractmethod
    def rkhs_inner_product(self, band : int,) -> torch.Tensor:
        """ Computes the RKHS inner product for the B-spline basis functions """
        pass


# Child classes for ASVGP
class Matern12B1SplineASVGP(ASVGP):

    """ ASVGP with B1-spline bases and Matern 1/2 kernel. """

    def __init__(self, 
                X : torch.Tensor, 
                y : torch.Tensor, 
                nknots : int, 
                dim1lims : Tuple[float, float]) -> 'ASVGP':
        """
        Arguments:
            train_x (torch.tensor)          : (n x 1) training inputs 
            train_y (torch.tensor)          : (n x 1) training targets
            nknots (int)                    : number of knots
            dim1lims (Tuple[float, float])  : lower and upper bounds of the input space (min, max)
        """
        super().__init__(X, y, nknots, dim1lims)
        self.basis = B1SplineBasis(self.mesh)
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2))

    def rkhs_inner_product(self, band: int) -> torch.Tensor:
        """ 
        Computes the RKHS inner product for the B-spline of order 1 

        Arguments:
            band : int, the band of the matrix to return (0 for main diagonal, 1 for first upper and lower diagonal) 
        """
        # assert band is 0 or 1 for B1-spline
        assert band in [0, 1], "band must be 0 or 1 for B-spline of order 1"
        # get hyperparameters
        scalesigma = self.kernel.outputscale
        lengthscale = self.kernel.base_kernel.lengthscale[0]
        n_basis_functions = self.basis.n_basis_functions
        # compute inner products
        if band == 0:
            # compute integral terms
            int1 = torch.ones(n_basis_functions) * (2. / self.delta)
            int2 = torch.ones(n_basis_functions) * ((2. / 3.) * self.delta)
            # boundary conditions
            bound_cond = (self.basis(self.alim) ** 2).flatten() + (self.basis(self.blim) ** 2).flatten()
            # inner product
            inner_prod = (int1 / (2. * scalesigma)) + (int2 / (2. * lengthscale * scalesigma)) + (bound_cond / (2. * scalesigma))
            return torch.diag_embed(inner_prod.flatten(), offset=0)
        else:
            # compute integrals
            int1 = torch.ones(n_basis_functions - 1)  / -self.delta
            int2 = torch.ones(n_basis_functions - 1) * (self.delta / 6.)
            # boundary conditions
            # boundary conditions = 0!
            # inner product
            inner_prod = (int1 / (2. * scalesigma)) + (int2 / (2. * lengthscale * scalesigma))
            return torch.diag_embed(inner_prod, offset=1) + torch.diag_embed(inner_prod, offset=-1)
        
    def compute_l2_inner_product(self):
        m = self.basis.n_basis_functions
        delta = self.basis.delta
        first_row = torch.nn.functional.pad(torch.as_tensor([2 / 3 * delta, 1 / 6 * delta]), (0, m - 2))
        boundary_correction = -torch.as_tensor([1 / 3 * delta, *[0.] * (m - 2), 1 / 3 * delta])
        return operators.ToeplitzLinearOperator(first_row).add_diagonal(boundary_correction)

    def compute_l2_grad_inner_product(self):
        m = self.basis.n_basis_functions
        delta = self.basis.delta
        first_row = torch.nn.functional.pad(torch.as_tensor([2 / delta, -1 / delta]), (0, m - 2))
        boundary_correction = -torch.as_tensor([1 / delta, *[0.] * (m - 2), 1 / delta])

        return operators.ToeplitzLinearOperator(first_row).add_diagonal(boundary_correction)

    def compute_boundary_condition(self):
        m = self.basis.n_basis_functions
        boundary_correction = torch.zeros(m)
        boundary_correction[[0, -1]] = 1.
        return operators.DiagLinearOperator(boundary_correction)
        
    def _Kuf(self, x) -> torch.Tensor:
        return self.basis(x)

    def _Kuu(self,) -> torch.Tensor:
        # # inner products
        # phi_mm = self.rkhs_inner_product(band=0)
        # phi_mmp1 = self.rkhs_inner_product(band=1)
        # # make banded matrix
        # Kuu = phi_mm + phi_mmp1

        # get hyperparameters
        scalesigma = self.kernel.outputscale
        lengthscale = self.kernel.base_kernel.lengthscale[0]
        # compute matrices
        A = self.compute_l2_inner_product()
        B = self.compute_l2_grad_inner_product()
        BC = self.compute_boundary_condition()
        return (
                    A.mul(lengthscale) +
                    B.mul(1 / lengthscale) +
                    BC
            ).mul(1 / (2 * scalesigma))
    

####################################################################################################
#                                                                                                  #
#                                            GRIDDED                                               #
#                                                                                                  #
####################################################################################################

# Parent class for GriddedGP
class GriddedGP(SparseGP):

    """ GriddedGP """

    def __init__(self, 
                X : torch.Tensor, 
                y : torch.Tensor, 
                nknots : int, 
                dim1lims : Tuple[float, float]) -> 'GriddedGP':
        """
        Arguments:
            train_x (torch.tensor)          : (n x 1) training inputs 
            train_y (torch.tensor)          : (n x 1) training targets
            nknots (int)                    : number of knots
            dim1lims (Tuple[float, float])  : lower and upper bounds of the input space (min, max)
        """
        super().__init__(X, y)
                # parameters
        self.nknots = nknots
        self.alim = dim1lims[0]
        self.blim = dim1lims[1]
        self.mesh = torch.linspace(self.alim, self.blim, nknots)
        self.delta = self.mesh[1] - self.mesh[0]
        # TODO: set kernel in child

    def q_v(self,):
        """ 
        Computes the q(v) = N(v|m, S) the posterior over the inducing features v
        - m = scalesigma^{-2} Kuu @ sigma^{-1} @ Kuf @ y
        - S = Kuu @ sigma^{-1} @ Kuu

        Arguments:
            None

        Returns:
            MultivariateNormal      : posterior over the inducing features v
        """
        X = self.train_inputs[0]
        y = self.train_targets
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # compute matrices
        Kuu = self._Kuu()
        Kuf = self._Kuf(X)
        sigma = gpytorch.lazify(self._sigma())
        # compute optimal mu
        optimal_mu = (Kuu @ sigma.inv_matmul(Kuf) @ y) / noisesigma
        # compute optimal sigma
        optimal_sigma = Kuu @ sigma.inv_matmul(Kuu)
        return gpytorch.distributions.MultivariateNormal(optimal_mu, optimal_sigma)
    

# Child classes for GriddedGP
class Matern12B0SplineGriddedGP(GriddedGP):
    """ Gridded GP with Matern 1/2 kernel and B0 spline basis functions. """
    def __init__(self, 
            X : torch.Tensor, 
            y : torch.Tensor, 
            nknots : int, 
            dim1lims : Tuple[float, float]) -> 'GriddedGP':
        """
        Arguments:
            train_x (torch.tensor)          : (n x 1) training inputs 
            train_y (torch.tensor)          : (n x 1) training targets
            nknots (int)                    : number of knots
            dim1lims (Tuple[float, float])  : lower and upper bounds of the input space (min, max)
        """
        super().__init__(X, y, nknots, dim1lims)
        self.basis = B0SplineBasis(self.mesh)
        self.n_splines = self.basis.m
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2))

    def _Kuf(self,
            x : torch.Tensor ) -> torch.Tensor:
        """ 
        Computes the Kuf matrix between the inducing variable and the latent function at x' (i.e Cov[v_i, f(x')]) according to the following cases:
        - Case 1: x' > right limit of v_i (i.e x' > b_i, x' above domain of v_i)
            -> Cov[v_i, f(x')] = - (exp{ -|x' - a| / l} - exp{ -|x' - b| / l} )
        - Case 2: x' < left limit of v_i (i.e x' < a_i, x' below domain of v_i)
            -> Cov[v_i, f(x')] = (exp{ -|x' - a| / l} - exp{ -|x' - b| / l} )
        - Case 3: x' \in [a_i, b_i] (i.e x' inside domain of v_i)
            -> Cov[v_i, f(x')] = 2 - (exp{ -|x' - a| / l} + exp{ -|x' - b| / l} )

        The implementation uses the following trick:
        1. Case 2 = (-1) x Case 1 -> therefore we can use an indicator function to mask the cases
        2. indicator function = 0 if x' \in [a_i, b_i], therefore the entries where x lies are 0. and we can fill them masking again with the indicator
        
        Arguments:
            basis (SplineBasis)     : the spline basis
            scalesigma (float)      : the scale parameter of the kernel
            lengthscale (float)     : the lengthscale parameter of the kernel
            x (torch.Tensor)        : the input tensor of shape

        Returns:
            Kuf (torch.Tensor)      : the Kuf matrix of shape (m, n)
        """
        # get parameters
        mesh = self.mesh
        m = self.n_splines
        scalesigma = self.kernel.outputscale
        lengthscale = self.kernel.base_kernel.lengthscale[0]
        k = torch.arange(m)
        x = x if len(x.shape) == 2 else x.unsqueeze(-1)
        # compute the indicator (-1 if x' < b, 0 if x' \in [a, b], 1 if x' > a)
        indicator = -torch.sign(torch.searchsorted(mesh, x[:, 0], right=False) - k[:, None] - 1)
        # compute the exponents
        # exp_1 = exp{ -|x - a| / l } = exp{ -|x - mesh[:-1]| / l}
        exp_1 = lengthscale * torch.exp(
            -torch.abs(x[:, 0] - mesh[:-1, None]) / lengthscale
            )
        # exp_2 = exp{ -|x - b| / l } = exp{ -|x - mesh[1:]| / l}
        exp_2 = lengthscale * torch.exp(
            -torch.abs(x[:, 0] - mesh[1:, None]) / lengthscale
            )
        # non-overlapping case
        Kuf = indicator * (exp_1 - exp_2)
        # overlapping case
        Kuf[indicator == 0] = 2 * lengthscale - (exp_1 + exp_2)[indicator == 0]
        Kuf *= scalesigma
        return Kuf
    
    def _Kuu(self, 
             ) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing variables (i.e Cov[v_i, v_j]) 
        
        The implementation uses the following trick:
        1. Cov[v_i, v_j] = Cov[v_i, v_{i + k}] for k = [0, 1, 2, ..., m]
        2. Using 1 we can re-write the covariances as a Toeplitz matrix with first row given by
            -> first_row = exp{ -(k - 1) * delta / l} + exp{ -(k + 1) * delta / l} - 2 * exp{ -k * delta / l}
        3. THe diagonal is given by:
            -> first_row[0] = 2 * (exp{ -delta / l} + (delta / l) - 1)

        Arguments:
            basis (SplineBasis)     : the spline basis
            scalesigma (float)      : the scale parameter of the kernel
            lengthscale (float)     : the lengthscale parameter of the kernel

        Returns:
            Kuu (torch.Tensor)      : the Kuu matrix of shape (m, m)
        """
        # get parameters
        m = self.n_splines
        delta = self.delta
        scalesigma = self.kernel.outputscale
        lengthscale = self.kernel.base_kernel.lengthscale.squeeze()
        k = torch.arange(m)
        # compute first row of Kuu 
        first_row = (
            torch.exp((-(k - 1) * delta) / lengthscale) + 
            torch.exp((-(k + 1) * delta) / lengthscale) - 
            2 * torch.exp((-k * delta) / lengthscale)
            )
        # add diagonal
        first_row[0] = 2 * (torch.exp(-delta / lengthscale) + (delta / lengthscale) - 1) 
        Kuu = operators.ToeplitzLinearOperator(first_row).to_dense()
        Kuu *= (lengthscale ** 2 * scalesigma)
        return Kuu
