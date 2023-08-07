# numeric imports
import torch
# GP imports
import gpytorch
# basis imports
from src.basis.fourier import FourierBasis, FourierBasisMatern12
from src.basis.bspline import SplineBasis, B0SplineBasis,B1SplineBasis
# misc imports
import linear_operator.operators as operators
from abc import ABC, abstractmethod
from typing import Tuple


# NOTE: ALL MODELS ARE MATERN 12 KERNELS
class KroneckerStructure(gpytorch.Module, ABC):
    """ 
    Parent class for Gaussian Process Regression in 2D with Kronecker structure, (inter domain) inducing features
    """
    def __init__(self,
                 X : torch.Tensor,
                 y : torch.Tensor,
                 ) -> 'KroneckerStructure':
        super().__init__()
        self.train_inputs = (X,)
        self.train_targets = y
        # model components
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.mean = gpytorch.means.ZeroMean()
        # kernel
        self.kernel_1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2, active_dims=[0]))
        self.kernel_2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2, active_dims=[1]))
        self.kernel = self.kernel_1 * self.kernel_2

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
        X1 = X[:, 0]
        X2 = X[:, 1]
        y = self.train_targets
        # initialise hyperparameters for dim 1
        self.kernel_1.outputscale = y.var()
        self.kernel_1.base_kernel.lengthscale[0]  = (X1.std() / lmbda)
        # initialise hyperparameters for dim 2
        self.kernel_2.outputscale = y.var()
        self.kernel_2.base_kernel.lengthscale[0]  = (X2.std() / lmbda)
        # intialise noise
        self.likelihood.noise = ((self.kernel_1.outputscale + self.kernel_2.outputscale) / 2) / (kappa ** 2)

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
        X1 = X[:, 0]
        X2 = X[:, 1]
        y = self.train_targets
        # initialise hyperparameters for dim 1
        self.kernel_1.outputscale = (torch.tensor(prior_amplitude) / 2) ** 2
        self.kernel_1.base_kernel.lengthscale[0]  = (X1.std() / lmbda)
        # initialise hyperparameters for dim 2
        self.kernel_2.outputscale = (torch.tensor(prior_amplitude) / 2) ** 2
        self.kernel_2.base_kernel.lengthscale[0]  = (X2.std() / lmbda)
        # intialise noise
        self.likelihood.noise = y.var() - ((self.kernel_1.outputscale + self.kernel_2.outputscale) / 2)

    def prior(self, 
              x : torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        computes the prior for a set of inputs x

        Arguments:
            x (torch.tensor)        : the inputs

        Returns:
            (MultivariateNormal)    : the GP prior over the inputs x
        """
        mean = self.mean(x)
        covar = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    @abstractmethod
    def _Kuu(self,) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing features
        - [Kuu]_{ij} = Cov[u_i, u_j]

        Arguments:
            None

        Returns:
            Kuu (torch.tensor)  : n x n matrix of inducing feature covariances
        """
        pass
    
    @abstractmethod
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
        y = self.train_targets
        # get noise sigma
        noise_sigma = self.likelihood.noise[0]
        # compute matrices
        Kuu = gpytorch.lazify(self._Kuu())
        Kuf = self._Kuf(self.train_inputs[0])
        Kff = self.kernel(self.train_inputs[0]).evaluate()
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

class Matern12SVGP(KroneckerStructure):

    """ SVGP class for 2D data using Kronecker structure with Matern 1/2 kernel """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 Z : torch.Tensor) -> KroneckerStructure:
        """
        Arguments:
            train_x (torch.tensor)      : (n x 2) training inputs 
            train_y (torch.tensor)      : (n x 1) training targets
            inducing_z (torch.tensor)   : (m x 2) inducing points
        """
        super().__init__(X, y)
        # register inducing points
        self.register_parameter("Z", torch.nn.Parameter(torch.zeros(Z.shape)))
        self.initialize(Z=Z)

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
        Kuu_1 = self.kernel_1(self.Z).evaluate()
        Kuu_2 = self.kernel_2(self.Z).evaluate()
        # compute the kronecker product
        Kuu = torch.kron(Kuu_1, Kuu_2)
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
        full_Z = torch.cartesian_prod(self.Z[:,0], self.Z[:,1])
        Kuf = self.kernel(full_Z, x).evaluate()
        return Kuf
    

####################################################################################################
#                                                                                                  #
#                                            VFF                                                   #
#                                                                                                  #
####################################################################################################

class Matern12VFFGP(KroneckerStructure):

    """ VFFGP class for 2D data using Kronecker structure with Matern 1/2 kernel """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 nfrequencies : int, 
                 dim1lims : Tuple[float, float] , 
                 dim2lims : Tuple[float, float]) -> KroneckerStructure:
        """ 
        Arguments:
            train_x (torch.tensor)          : (n x 2) training inputs
            train_y (torch.tensor)          : (n x 1) training targets
            nfrequencies (int)              : number of frequencies to use in the Fourier feature approximation
            dim1lims (Tuple[float, float])  : limits for the first dimension (dim1_min, dim1_max)
            dim2lims (Tuple[float, float])  : limits for the second dimension (dim2_min, dim2_max)
        """
        super().__init__(X, y)
        # register limits
        self.nfrequencies = nfrequencies
        self.dim1lims = dim1lims
        self.dim2lims = dim2lims
        # register omegas
        self.omegas_1 = FourierBasisMatern12(self.nfrequencies, self.dim1lims[0], self.dim1lims[1], 1.).omegas
        self.omegas_2 = FourierBasisMatern12(self.nfrequencies, self.dim2lims[0], self.dim2lims[1], 1.).omegas

    def spectral_density(self, 
                         dimomegas : FourierBasis,
                         dimkernel : gpytorch.kernels,) -> torch.Tensor:
        """
        Computes the spectra corersponding to the Matérn 1/2 covariances

        Arguments:
            dimbasis (FourierBasis)         : Fourier basis for the dimension
            dimkernel (gpytorch.kernels)    : kernel for the dimension 

        Returns:
            (torch.Tensor)                  : spectral density
        """
        # omegas
        omegas = dimomegas
        # hyperparameters
        scalesigma = dimkernel.outputscale
        lengthscale = dimkernel.base_kernel.lengthscale
        # get lamnda
        lmbda = 1 / lengthscale.squeeze()
        # compute spectral density
        numerator = 2 * scalesigma * lmbda
        denominator = (lmbda ** 2) + (omegas ** 2)
        spectral_density = numerator / denominator
        return spectral_density
    
    def _alpha(self, 
            dimlims : Tuple[float, float],
            dimomegas : FourierBasis,
            dimkernel : gpytorch.kernels,) -> torch.Tensor:
        """
        Computes alpha half of the Kuu representation for the Matérn 1/2 covarainces

        Arguments:
            dimbasis (FourierBasis)         : Fourier basis for the dimension
            dimkernel (gpytorch.kernels)    : kernel for the dimension

        Returns:
            (torch.Tensor)          : alpha
        """
        a, b = dimlims
        omegas = dimomegas
        # check that omegas[0] = 0
        assert omegas[0] == 0, "The first element of omegas must be 0"
        # compute the inverse spectral density
        S_inv = 1 / self.spectral_density(dimomegas, dimkernel)
        # compute the alpha half
        alpha = ((b - a) / 2) * torch.cat([2 * S_inv[0][None], S_inv[1:], S_inv[1:]])
        return alpha
    
    def _beta(self, 
            dimomegas : FourierBasis,
            dimkernel : gpytorch.kernels,) -> torch.Tensor:
        """
        Computes the beta half of the Kuu representation for the Matérn 1/2 covarainces

        Arguments:
            dimbasis (FourierBasis)         : Fourier basis for the dimension
            dimkernel (gpytorch.kernels)    : kernel for the dimension

        Returns:
            (torch.Tensor)                  : beta
        """
        scalesigma = dimkernel.outputscale.sqrt()
        # compute the sigma half
        sigma_half = torch.ones(len(dimomegas)) / scalesigma
        # compute the zero half
        zero_half = torch.zeros(len(dimomegas) - 1)
        # compute beta
        beta = torch.cat((sigma_half, zero_half))
        return beta
    
    def _Kuu_along_dim(self,
            dimlims : Tuple[float, float],
            dimomegas : FourierBasis,
            dimkernel : gpytorch.kernels,) -> torch.Tensor:
        """
        Computes the Kuu matrix for the Matérn 1/2 covarainces along a dimension 

        Arguments:
            dimbasis (FourierBasis)         : Fourier basis for the dimension
            dimkernel (gpytorch.kernels)    : kernel for the dimension

        Returns:
            (torch.Tensor)                  : Kuu
        """
        alpha = self._alpha(dimlims, dimomegas, dimkernel)
        beta = self._beta(dimomegas, dimkernel).unsqueeze(-1)
        return operators.DiagLinearOperator(alpha).add_low_rank(beta).to_dense().to(torch.float64) # TODO: this is also wrong, shouldn't have to cast!
    
    def _Kuf_along_dim(self, 
                    dimlims : Tuple[float, float],
                    dimkernel : FourierBasis,
                    x : float,) -> torch.Tensor:
        """ 
        Computes the Kuf matrix for the Matérn 1/2 covarainces along a dimension

        Arguments:
            dimbasis (FourierBasis)         : Fourier basis for the dimension
            x (float)                       : input along the dimension

        Returns:
            (torch.Tensor)                  : Kuf
        """
        a, b = dimlims
        lengthscale = dimkernel.base_kernel.lengthscale.squeeze()
        basis = FourierBasisMatern12(self.nfrequencies, a, b, lengthscale)
        return basis(x).to(torch.float64) # TODO: this is wrong, should return a float64 tensor (shouldntt bave to cast here)
    
    def _Kuu(self,) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing features
        - [Kuu]_{ij} = Cov[u_i, u_j]

        Arguments:
            None

        Returns:
            Kuu (torch.tensor)  : m x m matrix of inducing feature covariances
        """
        Kuu_1 =  self._Kuu_along_dim(self.dim1lims, self.omegas_1, self.kernel_1)
        Kuu_2 =  self._Kuu_along_dim(self.dim2lims, self.omegas_2, self.kernel_2)
        Kuu = torch.kron(Kuu_1, Kuu_2)
        return Kuu
    
    def _Kuf(self, 
            x : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the Kuf matrix between the inducing features and the latent function
        - [Kuf]_{ij} = Cov[u_i, f(x_j)]

        Arguments:
            x (torch.tensor)    : indicies for function evaluations

        Returns:
            Kuf (torch.tensor)  : m x n matrix of inducing feature covariances
        """
        Kuf_1 = self._Kuf_along_dim(self.dim1lims, self.kernel_1, x[:, 0])
        Kuf_2 = self._Kuf_along_dim(self.dim2lims, self.kernel_2, x[:, 1])
        Kuf = torch.stack([k1 * k2 for k2 in Kuf_1 for k1 in Kuf_2], dim = 0)
        return Kuf



####################################################################################################
#                                                                                                  #
#                                            ASVGP                                                 #
#                                                                                                  #
#################################################################################################### 

class Matern12B1SplineASVGP(KroneckerStructure):

    """ VFFGP class for 2D data using Kronecker structure with Matérn 1/2 covariances and B1-spline basis """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 nknots : int, 
                 dim1lims : Tuple[float, float] , 
                 dim2lims : Tuple[float, float]) -> KroneckerStructure:
        """ 
        Arguments:
            X (torch.tensor)                : (n x 2) training inputs
            y (torch.tensor)                : (n x 1) training targets
            nfrequencies (int)              : number of frequencies to use in the Fourier feature approximation
            dim1lims (Tuple[float, float])  : limits for the first dimension (dim1_min, dim1_max)
            dim2lims (Tuple[float, float])  : limits for the second dimension (dim2_min, dim2_max)
        """
        super().__init__(X, y)
        # register limits
        self.nknots = nknots
        self.dim1lims = dim1lims
        self.dim2lims = dim2lims
        self.mesh_1 = torch.linspace(self.dim1lims[0], self.dim1lims[1], nknots)
        self.mesh_2 = torch.linspace(self.dim2lims[0], self.dim2lims[1], nknots)
        self.delta_1 = self.mesh_1[1] - self.mesh_1[0]
        self.delta_2 = self.mesh_2[1] - self.mesh_2[0]
        self.delta = self.delta_1
        # basis
        self.basis_1 = B1SplineBasis(self.mesh_1)
        self.basis_2 = B1SplineBasis(self.mesh_2)

    def rkhs_inner_product(self,):
        print('depreciated')
        return None
    
    def compute_l2_inner_product(self, 
                                 dimbasis : SplineBasis) -> torch.tensor:
        """ """
        m = dimbasis.n_basis_functions
        delta = dimbasis.delta
        first_row = torch.nn.functional.pad(torch.as_tensor([2 / 3 * delta, 1 / 6 * delta]), (0, m - 2))
        boundary_correction = -torch.as_tensor([1 / 3 * delta, *[0.] * (m - 2), 1 / 3 * delta])
        return operators.ToeplitzLinearOperator(first_row).add_diagonal(boundary_correction).to_dense()

    def compute_l2_grad_inner_product(self,
                                      dimbasis : SplineBasis) -> torch.tensor:
        """"""
        m = dimbasis.n_basis_functions
        delta = dimbasis.delta
        first_row = torch.nn.functional.pad(torch.as_tensor([2 / delta, -1 / delta]), (0, m - 2))
        boundary_correction = -torch.as_tensor([1 / delta, *[0.] * (m - 2), 1 / delta])
        return operators.ToeplitzLinearOperator(first_row).add_diagonal(boundary_correction).to_dense()

    def compute_boundary_condition(self,
                                    dimbasis : SplineBasis) -> torch.tensor:
        m = dimbasis.n_basis_functions
        boundary_correction = torch.zeros(m)
        boundary_correction[[0, -1]] = 1.
        return operators.DiagLinearOperator(boundary_correction).to_dense()
        
    def _Kuu_along_dim(self,
                    dimbasis : FourierBasis,
                    dimkernel : gpytorch.kernels,) -> torch.Tensor:
        """
        Computes the Kuu matrix for the Matérn 1/2 covarainces along a dimension 

        Arguments:
            dimbasis (FourierBasis)         : basis for the dimension
            dimkernel (gpytorch.kernels)    : kernel for the dimension

        Returns:
            (torch.Tensor)                  : Kuu matrix for the dimension
        """
        #  # inner products
        # phi_mm = self.rkhs_inner_product(dimbasis, dimkernel, band=0)
        # phi_mmp1 = self.rkhs_inner_product(dimbasis, dimkernel, band=1)
        # # make banded matrix
        # Kuu = phi_mm + phi_mmp1
        # get hyperparameters
        scalesigma = dimkernel.outputscale
        lengthscale = dimkernel.base_kernel.lengthscale.squeeze()
        # compute matrices
        A = self.compute_l2_inner_product(dimbasis)
        B = self.compute_l2_grad_inner_product(dimbasis)
        BC = self.compute_boundary_condition(dimbasis)
        return (
                    A.mul(lengthscale) +
                    B.mul(1 / lengthscale) +
                    BC
            ).mul(1 / (2 * scalesigma))
    
    def _Kuf_along_dim(self,
                    dimbasis : FourierBasis,
                    x : torch.Tensor) -> torch.Tensor:
        """
        Computes the Kuf matrix for the Matérn 1/2 covarainces along a dimension 

        Arguments:
            dimbasis (FourierBasis)         : basis for the dimension

        Returns:
            (torch.Tensor)                  : Kuu matrix for the dimension
        """
        return dimbasis(x)
    
    def _Kuu(self,) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing features
        - [Kuu]_{ij} = Cov[u_i, u_j]

        Arguments:
            None

        Returns:
            Kuu (torch.tensor)  : m x m matrix of inducing feature covariances
        """
        Kuu_1 = self._Kuu_along_dim(self.basis_1, self.kernel_1)
        Kuu_2 = self._Kuu_along_dim(self.basis_2, self.kernel_2)
        # compute the kronecker product
        Kuu = torch.kron(Kuu_1, Kuu_2)
        return Kuu.to(torch.float64) # TODO: fix this hacky fix

    def _Kuf(self, 
             x : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the Kuf matrix between the inducing features and the latent function
        - [Kuf]_{ij} = Cov[u_i, f(x_j)]

        Arguments:
            x (torch.tensor)    : indicies for function evaluations

        Returns:
            Kuf (torch.tensor)  : m x n matrix of inducing feature covariances
        """
        Kuf_1 = self._Kuf_along_dim(self.basis_1, x[:, 0])
        Kuf_2 = self._Kuf_along_dim(self.basis_2, x[:, 1])
        Kuf = torch.stack([k1 * k2 for k2 in Kuf_1 for k1 in Kuf_2], dim = 0)
        return Kuf.to(torch.float64) # TODO: fix this hacky fix


####################################################################################################
#                                                                                                  #
#                                            GRIDDED                                               #
#                                                                                                  #
####################################################################################################

class Matern12B0SplineGriddedGP(KroneckerStructure):

    """ VFFGP class for 2D data using Kronecker structure """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 nknots : int, 
                 dim1lims : Tuple[float, float] , 
                 dim2lims : Tuple[float, float]) -> KroneckerStructure:
        """ 
        Arguments:
            X (torch.tensor)                : (n x 2) training inputs
            y (torch.tensor)                : (n x 1) training targets
            nfrequencies (int)              : number of frequencies to use in the Fourier feature approximation
            dim1lims (Tuple[float, float])  : limits for the first dimension (dim1_min, dim1_max)
            dim2lims (Tuple[float, float])  : limits for the second dimension (dim2_min, dim2_max)
        """
        super().__init__(X, y)
                # register limits
        self.nknots = nknots
        self.dim1lims = dim1lims
        self.dim2lims = dim2lims
        self.mesh_1 = torch.linspace(self.dim1lims[0], self.dim1lims[1], nknots)
        self.mesh_2 = torch.linspace(self.dim2lims[0], self.dim2lims[1], nknots)
        self.delta_1 = self.mesh_1[1] - self.mesh_1[0]
        self.delta_2 = self.mesh_2[1] - self.mesh_2[0]
        # basis
        self.basis_1 = B0SplineBasis(self.mesh_1)
        self.basis_2 = B0SplineBasis(self.mesh_2)
    
    def _Kuu_along_dim(self, 
                    dimbasis : torch.Tensor,
                    dimkernel : gpytorch.kernels,) -> torch.Tensor:
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
        m = dimbasis.m
        delta = dimbasis.delta
        scalesigma = dimkernel.outputscale
        lengthscale = dimkernel.base_kernel.lengthscale[0]
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
    
    def _Kuf_along_dim(self,
                    dimbasis : torch.Tensor,
                    dimkernel : gpytorch.kernels,
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
        mesh = dimbasis.mesh
        m = dimbasis.m
        scalesigma = dimkernel.outputscale
        lengthscale = dimkernel.base_kernel.lengthscale[0]
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
    
    def _Kuu(self,) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing features
        - [Kuu]_{ij} = Cov[u_i, u_j]

        Arguments:
            None

        Returns:
            Kuu (torch.tensor)  : m x m matrix of inducing feature covariances
        """
        Kuu_1 = self._Kuu_along_dim(self.basis_1, self.kernel_1)
        Kuu_2 = self._Kuu_along_dim(self.basis_2, self.kernel_2)
        Kuu  = torch.kron(Kuu_1, Kuu_2)
        return Kuu
    
    def _Kuf(self,
             x : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the Kuf matrix between the inducing features and the latent function
        - [Kuf]_{ij} = Cov[u_i, f(x_j)]

        Arguments:
            x (torch.tensor)    : indicies for function evaluations

        Returns:
            Kuf (torch.tensor)  : m x n matrix of inducing feature covariances
        """
        Kuf_1 = self._Kuf_along_dim(self.basis_1, self.kernel_1, x[:, 0])
        Kuf_2 = self._Kuf_along_dim(self.basis_2, self.kernel_2, x[:, 1])
        Kuf = torch.stack([k1 * k2 for k2 in Kuf_1 for k1 in Kuf_2], dim = 0)
        return Kuf
    
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


