# numeric imports
import torch
import numpy 
# gp imports
import gpytorch
from src.models.exact.univariate_structure import Matern12GP
from src.models.sparse.univariate_structure import SparseGP, Matern12VFFGP
from src.basis.bspline import SplineBasis, B0SplineBasis, B1SplineBasis
from src.basis.fourier import FourierBasisMatern12
# operator imports
import linear_operator.operators as operators
# typing imports
from typing import Tuple


####################################################################################################
#                                                                                                  #
#                                            EXACTGP                                               #
#                                                                                                  #
####################################################################################################

class GriddedMatern12ExactGP(Matern12GP):

    def __init__(self,
                train_x: torch.Tensor, 
                train_y: torch.Tensor,
                n_b0_splines : int,
                gridlims : Tuple[float, float],
                likelihood = gpytorch.likelihoods.GaussianLikelihood()):

        super().__init__(train_x, train_y, likelihood)
        self.dimlims = gridlims
        self.n_b0_splines = n_b0_splines
        self.b0_mesh_1 = torch.linspace(self.dimlims[0], self.dimlims[1], self.n_b0_splines + 1)
        self.b0_delta_1 = self.b0_mesh_1[1] - self.b0_mesh_1[0]
        self.b0_basis_1 = B0SplineBasis(self.b0_mesh_1)

    def _Kxf(self,
             x : torch.Tensor) -> torch.Tensor:
        """ """
        return self.kernel(self.train_x, x).evaluate()
    
    def _Kxx(self) -> torch.Tensor:
        """ """
        return self.kernel(self.train_x, self.train_x).evaluate()
    
    def _sigma(self,) -> torch.Tensor:
        """ 
        Computes [Kxx + noisesigma^{2} In] 
        
        Arguments:
            None

        Returns:
            sigma (torch.tensor)    : n x n matrix, [Kuu + noisesigma^{-2} Kuf Kuf^T] 
        """
        X = self.train_inputs[0]
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # compute matrices
        Kxx = self._Kxx()
        return Kxx + noisesigma * torch.eye(X.shape[0])

    def _Kvx(self,
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
        mesh = self.b0_basis_1.mesh
        m = self.b0_basis_1.n_basis_functions
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
    
    def _Kvv(self, 
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
        m = self.b0_basis_1.n_basis_functions
        delta = self.b0_basis_1.delta
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
    
    def q_v(self,):
        # compute matrices
        Kxx = gpytorch.lazify(self._Kxx())
        Kvx = self._Kvx(self.train_inputs[0])
        Kvv = self._Kvv()
        sigma = gpytorch.lazify(self._sigma())
        p_f_y_cov = gpytorch.lazify(Kxx.evaluate() - Kxx.evaluate() @ sigma.inv_matmul(Kxx.evaluate()))
        # compute mean
        mean = Kvx @ sigma.inv_matmul(self.train_targets)
        cov = Kvv - (Kvx @ Kxx.inv_matmul(Kvx.T)) + Kvx @ p_f_y_cov.inv_matmul(Kvx.T)
        return gpytorch.distributions.MultivariateNormal(mean, cov)
    

####################################################################################################
#                                                                                                  #
#                                            SVGP                                                  #
#                                                                                                  #
####################################################################################################    

class GriddedMatern12SVGP(SparseGP):

    def __init__(self, 
                X : torch.Tensor, 
                y : torch.Tensor, 
                Z : torch.Tensor,
                n_b0_splines : int,
                gridlims : Tuple[float, float],):
        """ """
        super().__init__(X, y)
        # register inducing points
        self.register_parameter("Z", torch.nn.Parameter(torch.zeros(Z.shape)))
        self.initialize(Z=Z)
        # gridded componenets
        self.gridlims = gridlims
        self.n_b0_splines = n_b0_splines
        self.b0_mesh_1 = torch.linspace(self.gridlims[0], self.gridlims[1], self.n_b0_splines + 1)
        self.b0_delta_1 = self.b0_mesh_1[1] - self.b0_mesh_1[0]
        self.b0_basis_1 = B0SplineBasis(self.b0_mesh_1)
        # kernel
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2))

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

    def _Kvf(self,
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
        mesh = self.b0_basis_1.mesh
        m = self.b0_basis_1.n_basis_functions
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
    
    def _Kvv(self, 
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
        m = self.b0_basis_1.n_basis_functions
        delta = self.b0_basis_1.delta
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
    
    def q_v(self,):
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # compute matrices
        Kvf = self._Kvf(self.Z)
        Kvv = self._Kvv()
        Kuf = self._Kuf(self.train_inputs[0])
        Kuu = gpytorch.lazify(self._Kuu())
        sigma = gpytorch.lazify(self._sigma())
        # compute conditional mean
        m_v = (Kvf @ sigma.inv_matmul(Kuf) @ self.train_targets) / noisesigma
        # compute conditional covariance
        S_v = Kvv - (Kvf @ Kuu.inv_matmul(Kvf.T)) + (Kvf @ sigma.inv_matmul(Kvf.T))
        return gpytorch.distributions.MultivariateNormal(m_v, S_v)
    

####################################################################################################
#                                                                                                  #
#                                            VFFGP                                                 #
#                                                                                                  #
####################################################################################################    

class GriddedMatern12VFFGP(Matern12VFFGP):

    """ VFFGP class for 2D data using Kronecker structure with Matern 1/2 kernel """

    def __init__(self, 
                X : torch.Tensor, 
                y : torch.Tensor, 
                # basis part
                nfrequencies : int, 
                n_b0_splines : int, 
                # limits
                vfflims : Tuple[float, float] , 
                gridlims : Tuple[float, float]):
        super().__init__(X, y, nfrequencies, vfflims)
        # register limits
        self.n_b0_splines = n_b0_splines
        self.nknots = self.n_b0_splines + 1 # for B0-splines
        self.griddim1lims = gridlims
        self.vffdim1lims = vfflims
        self.b0_mesh = torch.linspace(self.griddim1lims[0], self.griddim1lims[1], self.nknots)
        # basis
        self.b0_basis = B0SplineBasis(self.b0_mesh)

    @staticmethod
    def _Kvu_0(
        B0basis : B0SplineBasis) -> torch.Tensor:
        Kvu_0 = torch.ones(B0basis.m, dtype=torch.float64) * B0basis.delta
        return Kvu_0.unsqueeze(-1)
    
    @staticmethod
    def _Kvu_cos(
        B0basis : B0SplineBasis, 
        VFFbasis : FourierBasisMatern12) -> torch.Tensor:
        """ """
        # get the omegas
        omegas = VFFbasis.omegas
        a = VFFbasis.a
        beta_half = torch.sin(omegas[1:] * (B0basis.mesh[1:] - a)[:, None]) # Sin(w_j (beta - a))
        alpha_half = torch.sin(omegas[1:] * (B0basis.mesh[:-1] - a)[:, None]) # SIn(w_j (alpha - a))
        Kvu_cos = (beta_half - alpha_half) / omegas[1:] # (Sin(w_j (beta - a)) - Sin(w_j (alpha - a))) / w_j
        return Kvu_cos
    
    @staticmethod
    def _Kvu_sin(
        B0basis : B0SplineBasis, 
        VFFbasis : FourierBasisMatern12) -> torch.Tensor:
        """ """
        # get the omegas
        omegas = VFFbasis.omegas
        b = VFFbasis.a
        beta_half = torch.cos(omegas[1:] * (B0basis.mesh[1:] - b)[:, None]) # Cos(w_j (beta - a))
        alpha_half = torch.cos(omegas[1:] * (B0basis.mesh[:-1] - b)[:, None]) # Sin(w_j (alpha - a))
        Kvu_sin = -(beta_half - alpha_half) / omegas[1:]
        return Kvu_sin

    def _Kvu(self,) -> torch.Tensor:
        """  """
        # hyperparemters
        lengthscale_1 = self.kernel.base_kernel.lengthscale.squeeze()
        # basis
        vffbasis = FourierBasisMatern12(self.nfrequencies, self.vffdim1lims[0], self.vffdim1lims[1], lengthscale_1)
        # compute matrices
        Kvu_0 = self._Kvu_0(self.b0_basis)
        Kvu_cos = self._Kvu_cos(self.b0_basis, vffbasis)
        Kvu_sin = self._Kvu_sin(self.b0_basis, vffbasis)
        Kvu = torch.cat([Kvu_0, Kvu_cos, Kvu_sin], dim=1)
        return Kvu

    def _Kvv(self,) -> torch.Tensor:
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
        m = self.b0_basis.m
        delta = self.b0_basis.delta
        scalesigma = self.kernel.outputscale
        lengthscale = self.kernel.base_kernel.lengthscale[0]
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
        Kuu *= ((lengthscale ** 2) * scalesigma)
        return Kuu

    def q_u(self,) -> gpytorch.distributions.MultivariateNormal:
        """ 
        Computes the q(u) = N(u|m, S) the posterior over the inducing features v
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
    
    def p_v_u(self,):
        """ computes p(v|u) """
        # compute matrics
        Kvu = self._Kvu()
        Kuu = gpytorch.lazify(self._Kuu())
        Kvv = self._Kvv()
        u = self.q_u().mean
        # compute conditional mean
        m_v_u = Kvu @ Kuu.inv_matmul(u)
        # compute conditional covariance
        S_v_u = Kvv - Kvu @ Kuu.inv_matmul(Kvu.T)   
        return gpytorch.distributions.MultivariateNormal(m_v_u, S_v_u)

    def q_v(self,):
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # compute matrices
        Kvu = self._Kvu()
        Kvv = self._Kvv()
        Kuf = self._Kuf(self.train_inputs[0])
        Kuu = gpytorch.lazify(self._Kuu())
        sigma = gpytorch.lazify(self._sigma())
        # compute conditional mean
        m_v = (Kvu @ sigma.inv_matmul(Kuf) @ self.train_targets) / noisesigma
        # compute conditional covariance
        S_v = Kvv - (Kvu @ Kuu.inv_matmul(Kvu.T)) + (Kvu @ sigma.inv_matmul(Kvu.T))
        return gpytorch.distributions.MultivariateNormal(m_v, S_v)
       

####################################################################################################
#                                                                                                  #
#                                            ASVGP                                                 #
#                                                                                                  #
#################################################################################################### 

class GriddedMatern12ASVGP(SparseGP):    

    """ VFFGP class for 2D data using Kronecker structure with Matern 1/2 kernel """

    def __init__(self, 
                 # data
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 # B-spline part
                 n_b0_splines : int,
                 n_b1_splines : int, # NOTE: this is the number of B1-splines in one B0 Spline
                 # vff part
                 dimlims : Tuple[float, float]):
        """  """
        super().__init__(X, y)
        # grid limits
        self.dimlims = dimlims
        # - MESH - #
        # b0 spline
        self.n_b0_splines = n_b0_splines
        self.b0_mesh_1 = torch.linspace(dimlims[0], dimlims[1], self.n_b0_splines + 1)
        self.b0_delta_1 = self.b0_mesh_1[1] - self.b0_mesh_1[0]
        # paded mesh
        self.padding = 1
        self.b0_mesh_padded_1 = torch.cat(((self.b0_mesh_1[0] - self.b0_delta_1).unsqueeze(-1), self.b0_mesh_1, (self.b0_mesh_1[-1] + self.b0_delta_1).unsqueeze(-1)))
        # b1 spline
        self.n_b1_splines = n_b1_splines
        self.b1_mesh_1 = torch.stack([torch.linspace(self.b0_mesh_padded_1[i], self.b0_mesh_padded_1[i+1], self.n_b1_splines+2)[:-1] for i in range(self.n_b0_splines+ 2*(self.padding))], dim = 0).flatten()
        self.b1_mesh_1 = torch.cat((self.b1_mesh_1, self.b0_mesh_padded_1[-1].view(1)))
        # padd the meshes
        # - BASIS - #
        # B0 basis
        self.b0_basis_1 = B0SplineBasis(self.b0_mesh_1)
        # B1 basis
        self.b1_basis_1 = B1SplineBasis(self.b1_mesh_1)
        # kernel
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1/2))

    # - ASVGP PART - #
    def compute_l2_inner_product(self,) -> torch.tensor:
        """ """
        m = self.b1_basis_1.n_basis_functions
        delta = self.b1_basis_1.delta
        first_row = torch.nn.functional.pad(torch.as_tensor([2 / 3 * delta, 1 / 6 * delta]), (0, m - 2))
        boundary_correction = -torch.as_tensor([1 / 3 * delta, *[0.] * (m - 2), 1 / 3 * delta])
        return operators.ToeplitzLinearOperator(first_row).add_diagonal(boundary_correction).to_dense()

    def compute_l2_grad_inner_product(self,) -> torch.tensor:
        """"""
        m = self.b1_basis_1.n_basis_functions
        delta = self.b1_basis_1.delta
        first_row = torch.nn.functional.pad(torch.as_tensor([2 / delta, -1 / delta]), (0, m - 2))
        boundary_correction = -torch.as_tensor([1 / delta, *[0.] * (m - 2), 1 / delta])
        return operators.ToeplitzLinearOperator(first_row).add_diagonal(boundary_correction).to_dense()

    def compute_boundary_condition(self,) -> torch.tensor:
        m = self.b1_basis_1.n_basis_functions
        boundary_correction = torch.zeros(m)
        boundary_correction[[0, -1]] = 1.
        return operators.DiagLinearOperator(boundary_correction).to_dense()
        
    def _Kuu(self,) -> torch.Tensor:
        """
        Computes the Kuu matrix for the Matérn 1/2 covarainces along a dimension 

        Arguments:
            dimbasis (FourierBasis)         : basis for the dimension
            dimkernel (gpytorch.kernels)    : kernel for the dimension

        Returns:
            (torch.Tensor)                  : Kuu matrix for the dimension
        """
        # get hyperparameters
        scalesigma = self.kernel.outputscale
        lengthscale = self.kernel.base_kernel.lengthscale.squeeze()
        # compute matrices
        A = self.compute_l2_inner_product()
        B = self.compute_l2_grad_inner_product()
        BC = self.compute_boundary_condition()
        return (
                    A.mul(lengthscale) +
                    B.mul(1 / lengthscale) +
                    BC
            ).mul(1 / (2 * scalesigma)).to(torch.float64)
    
    def _Kuf(self, x : torch.Tensor) -> torch.Tensor:
        """
        Computes the Kuf matrix for the Matérn 1/2 covarainces along a dimension 

        Arguments:
            dimbasis (FourierBasis)         : basis for the dimension

        Returns:
            (torch.Tensor)                  : Kuu matrix for the dimension
        """
        return self.b1_basis_1(x)
    
    # - Gridded part - #
    def _Kvu(self,) -> torch.tensor:
        # compute the l2 inner product
        half_b1_basis_function_l2 = self.b1_basis_1.delta /2.
        full_b1_basis_function_l2 = self.b1_basis_1.delta 
        # cast across the intersecting basis functions
        A = half_b1_basis_function_l2.unsqueeze(-1)
        B = torch.ones(self.n_b1_splines) * full_b1_basis_function_l2
        C = half_b1_basis_function_l2.unsqueeze(-1)
        non_zero_segement = torch.cat((A, B, C))
        left_zero_segment = torch.zeros(self.n_b1_splines + 1)
        right_zero_segment = torch.zeros(self.b1_basis_1.n_basis_functions - (non_zero_segement.shape[0] + len(left_zero_segment)))
        first_row = torch.cat((left_zero_segment, non_zero_segement, right_zero_segment))
        Kvu = torch.vstack([torch.roll(first_row, (self.n_b1_splines+1)*i) for i in range(self.n_b0_splines)])
        return Kvu.to(torch.float64)


    def _Kvv(self,) -> torch.Tensor:
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
        m = self.b0_basis_1.m
        delta = self.b0_basis_1.delta
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
        Kuu *= ((lengthscale ** 2) * scalesigma)
        return Kuu.to(torch.float64)
        
    def q_u(self,) -> gpytorch.distributions.MultivariateNormal:
        """ 
        Computes the q(u) = N(u|m, S) the posterior over the inducing features v
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
    
    def p_v_u(self,):
        """ computes p(v|u) """
        # compute matrics
        Kvu = self._Kvu()
        Kuu = gpytorch.lazify(self._Kuu())
        Kvv = self._Kvv()
        u = self.q_u().mean
        # compute conditional mean
        m_v_u = Kvu @ Kuu.inv_matmul(u)
        # compute conditional covariance
        S_v_u = Kvv - Kvu @ Kuu.inv_matmul(Kvu.T)   
        return gpytorch.distributions.MultivariateNormal(m_v_u, S_v_u)

    def q_v(self,):
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # compute matrices
        Kvu = self._Kvu()
        Kvv = self._Kvv()
        Kuf = self._Kuf(self.train_inputs[0])
        Kuu = gpytorch.lazify(self._Kuu())
        sigma = gpytorch.lazify(self._sigma())
        # compute conditional mean
        m_v = (Kvu @ sigma.inv_matmul(Kuf) @ self.train_targets) / noisesigma
        # compute conditional covariance
        S_v = Kvv - (Kvu @ Kuu.inv_matmul(Kvu.T)) + (Kvu @ sigma.inv_matmul(Kvu.T))
        return gpytorch.distributions.MultivariateNormal(m_v, S_v)


####################################################################################################
#                                                                                                  #
#                                            GRIDDEDGP                                             #
#                                                                                                  #
####################################################################################################

class Matern12GriddedGP(SparseGP):

    """ GriddedGP """

    def __init__(self, 
                X : torch.Tensor, 
                y : torch.Tensor, 
                n_b0_splines : int, 
                gridlims : Tuple[float, float]) -> 'SparseGP':
        """
        Arguments:
            train_x (torch.tensor)          : (n x 1) training inputs 
            train_y (torch.tensor)          : (n x 1) training targets
            nknots (int)                    : number of knots
            dim1lims (Tuple[float, float])  : lower and upper bounds of the input space (min, max)
        """
        super().__init__(X, y)
        # parameters
        self.gridlims = gridlims
        self.n_b0_splines = n_b0_splines
        self.b0_mesh_1 = torch.linspace(self.gridlims[0], self.gridlims[1], self.n_b0_splines + 1)
        self.b0_basis = B0SplineBasis(self.b0_mesh_1)
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
        mesh = self.b0_mesh_1
        m = self.n_b0_splines
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
        m = self.n_b0_splines
        delta = self.b0_basis.delta
        scalesigma = self.kernel.outputscale
        lengthscale = self.kernel.base_kernel.lengthscale[0]
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

