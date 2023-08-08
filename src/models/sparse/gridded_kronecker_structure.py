# numeric imports
import torch
# gp imports
import gpytorch
from src.models.sparse.kronecker_structure import KroneckerStructure, Matern12SVGP, Matern12VFFGP
from src.basis.fourier import FourierBasis, FourierBasisMatern12
from src.basis.bspline import SplineBasis, B0SplineBasis, B1SplineBasis

from typing import Tuple
from src.basis.bspline import B0SplineBasis
import linear_operator.operators as operators




####################################################################################################
#                                                                                                  #
#                                            SVGP                                                  #
#                                                                                                  #
####################################################################################################

class GriddedMatern12SVGP(Matern12SVGP):

    """ class to get gridded predictions from an SVGP model """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 Z : torch.Tensor,
                 nsplines : int, 
                 dim1lims : Tuple[float, float] , 
                 dim2lims : Tuple[float, float]) -> KroneckerStructure:
        """ """
        super().__init__(X, y, Z)
        # register limits
        self.nsplines = nsplines
        self.nknots = self.nsplines + 1 # for B0-splines
        self.dim1lims = dim1lims
        self.dim2lims = dim2lims
        self.mesh_1 = torch.linspace(self.dim1lims[0], self.dim1lims[1], self.nknots)
        self.mesh_2 = torch.linspace(self.dim2lims[0], self.dim2lims[1], self.nknots)
        self.delta_1 = self.mesh_1[1] - self.mesh_1[0]
        self.delta_2 = self.mesh_2[1] - self.mesh_2[0]
        # basis
        self.basis_1 = B0SplineBasis(self.mesh_1)
        self.basis_2 = B0SplineBasis(self.mesh_2)
    
    #TODO: rename Kvf to Kvu
    @staticmethod
    def _Kvf_along_dim(
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
    
    def _Kvf(self,
        x : torch.Tensor) -> torch.Tensor:
        """ 
        Computes the Kuf matrix between the inducing features and the latent function
        - [Kuf]_{ij} = Cov[u_i, f(x_j)]

        Arguments:
            x (torch.tensor)    : indicies for function eval|uations

        Returns:
            Kuf (torch.tensor)  : m x n matrix of inducing feature covariances
        """
        
        Kuf_1 = self._Kvf_along_dim(self.basis_1, self.kernel_1, x[:, 0])
        Kuf_2 = self._Kvf_along_dim(self.basis_2, self.kernel_2, x[:, 1])
        Kuf = torch.stack([k1 * k2 for k2 in Kuf_1 for k1 in Kuf_2], dim = 0)
        return Kuf
    
    @staticmethod
    def _Kvv_along_dim( 
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
        Kuu *= (lengthscale * 2 * scalesigma)
        return Kuu
    
    def _Kvv(self,) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing features
        - [Kuu]_{ij} = Cov[u_i, u_j]

        Arguments:
            None

        Returns:
            Kuu (torch.tensor)  : m x m matrix of inducing feature covariances
        """
        Kuu_1 = self._Kvv_along_dim(self.basis_1, self.kernel_1)
        Kuu_2 = self._Kvv_along_dim(self.basis_2, self.kernel_2)
        Kuu  = torch.kron(Kuu_1, Kuu_2)
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
        # get Kron inducing features
        kron_Z = torch.cartesian_prod(self.Z[:,0], self.Z[:,1])
        # compute matrics
        Kvf = self._Kvf(kron_Z)
        Kuu = gpytorch.lazify(self._Kuu())
        Kvv = self._Kvv()
        u = self.q_u().mean
        # compute conditional mean
        m_v_u = Kvf @ Kuu.inv_matmul(u)
        # compute conditional covariance
        S_v_u = Kvv - Kvf @ Kuu.inv_matmul(Kvf.T)   
        return gpytorch.distributions.MultivariateNormal(m_v_u, S_v_u)

    def q_v(self,):
        # get noise sigma
        noisesigma = self.likelihood.noise[0]
        # get Kron inducing features
        kron_Z = torch.cartesian_prod(self.Z[:,0], self.Z[:,1])
        # compute matrices
        Kvf = self._Kvf(kron_Z)
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
#                                            VFF                                                   #
#                                                                                                  #
####################################################################################################

class GriddedMatern12VFFGP(Matern12VFFGP):

    """ VFFGP class for 2D data using Kronecker structure with Matern 1/2 kernel """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 # vff part
                 nfrequencies : int, 
                 vffdim1lims : Tuple[float, float] , 
                 vffdim2lims : Tuple[float, float],
                 # grid part
                 nsplines : int, 
                 griddim1lims : Tuple[float, float],
                 griddim2lims : Tuple[float, float]):
        super().__init__(X, y, nfrequencies, vffdim1lims, vffdim2lims)
        # register limits
        self.nsplines = nsplines
        self.nknots = self.nsplines + 1 # for B0-splines
        self.griddim1lims = griddim1lims
        self.griddim2lims = griddim2lims
        self.mesh_1 = torch.linspace(self.griddim1lims[0], self.griddim1lims[1], self.nknots)
        self.mesh_2 = torch.linspace(self.griddim2lims[0], self.griddim2lims[1], self.nknots)
        self.delta_1 = self.mesh_1[1] - self.mesh_1[0]
        self.delta_2 = self.mesh_2[1] - self.mesh_2[0]
        # basis
        self.B0basis_1 = B0SplineBasis(self.mesh_1)
        self.B0basis_2 = B0SplineBasis(self.mesh_2)

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

    def _Kvu_along_dim(self,
                    dimB0basis : B0SplineBasis, 
                    dimVFFbasis : FourierBasisMatern12):
        """  """
        Kvu_0 = self._Kvu_0(dimB0basis)
        Kvu_cos = self._Kvu_cos(dimB0basis, dimVFFbasis)
        Kvu_sin = self._Kvu_sin(dimB0basis, dimVFFbasis)
        Kvu = torch.cat([Kvu_0, Kvu_cos, Kvu_sin], dim=1)
        return Kvu
    
    def _Kvu(self,) -> torch.Tensor:
        """ """
        # hyperparemters
        lengthscale_1 = self.kernel_1.base_kernel.lengthscale[0]
        lengthscale_2 = self.kernel_2.base_kernel.lengthscale[0]
        # basis
        vffbasis_1 = FourierBasisMatern12(self.nfrequencies, self.dim1lims[0], self.dim1lims[1], lengthscale_1)
        vffbasis_2 = FourierBasisMatern12(self.nfrequencies, self.dim2lims[0], self.dim2lims[1], lengthscale_2)
        # compute Kvf
        Kvu_1 = self._Kvu_along_dim(self.B0basis_1, vffbasis_1)
        Kvu_2 = self._Kvu_along_dim(self.B0basis_2, vffbasis_2)
        # compute Kvu
        # Kvu = torch.stack([k1 * k2 for k2 in Kvu_1 for k1 in Kvu_2], dim = 0)
        Kvu = torch.kron(Kvu_1, Kvu_2)
        return Kvu

    @staticmethod
    def _Kvv_along_dim( 
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
        Kuu *= (lengthscale * 2 * scalesigma)
        return Kuu
    
    def _Kvv(self,) -> torch.Tensor:
        """ 
        Computes the Kuu matrix between the inducing features
        - [Kuu]_{ij} = Cov[u_i, u_j]

        Arguments:
            None

        Returns:
            Kuu (torch.tensor)  : m x m matrix of inducing feature covariances
        """
        Kuu_1 = self._Kvv_along_dim(self.B0basis_1, self.kernel_1)
        Kuu_2 = self._Kvv_along_dim(self.B0basis_2, self.kernel_2)
        Kuu  = torch.kron(Kuu_1, Kuu_2)
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
        print('Kvu shape: ', Kvu.shape)
        print('Kuu shape: ', Kuu.shape)
        print('Kvv shape: ', Kvv.shape)
        print('u shape: ', u.shape)
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
                 dim1lims : Tuple[float, float],
                 dim2lims : Tuple[float, float],):
        """  """
        super().__init__(X, y)
        # grid limits
        self.dim1lims = dim1lims
        self.dim2lims = dim2lims
        # - MESH - #
        # b0 spline
        self.n_b0_splines = n_b0_splines
        self.b0_mesh_1 = torch.linspace(dim1lims[0], dim1lims[1], self.n_b0_splines + 1)
        self.b0_mesh_2 = torch.linspace(dim2lims[0], dim2lims[1], self.n_b0_splines + 1)
        self.b0_delta_1 = self.b0_mesh_1[1] - self.b0_mesh_1[0]
        self.b0_delta_2 = self.b0_mesh_2[1] - self.b0_mesh_2[0]
        # paded mesh
        self.padding = 1
        self.b0_mesh_padded_1 = torch.cat(((self.b0_mesh_1[0] - self.b0_delta_1).unsqueeze(-1), self.b0_mesh_1, (self.b0_mesh_1[-1] + self.b0_delta_1).unsqueeze(-1)))
        self.b0_mesh_padded_2 = torch.cat(((self.b0_mesh_2[0] - self.b0_delta_2).unsqueeze(-1), self.b0_mesh_2, (self.b0_mesh_2[-1] + self.b0_delta_2).unsqueeze(-1)))
        # b1 spline
        self.n_b1_splines = n_b1_splines
        self.b1_mesh_1 = torch.stack([torch.linspace(self.b0_mesh_padded_1[i], self.b0_mesh_padded_1[i+1], self.n_b1_splines+2)[:-1] for i in range(self.n_b0_splines+ 2*(self.padding))], dim = 0).flatten()
        self.b1_mesh_1 = torch.cat((self.b1_mesh_1, self.b0_mesh_padded_1[-1].view(1)))
        self.b1_mesh_2 = torch.stack([torch.linspace(self.b0_mesh_padded_2[i], self.b0_mesh_padded_2[i+1], self.n_b1_splines+2)[:-1] for i in range(self.n_b0_splines+ 2*(self.padding))], dim = 0).flatten()
        self.b1_mesh_2 = torch.cat((self.b1_mesh_2, self.b0_mesh_padded_2[-1].view(1)))
        # padd the meshes
        # - BASIS - #
        # B0 basis
        self.b0_basis_1 = B0SplineBasis(self.b0_mesh_1)
        self.b0_basis_2 = B0SplineBasis(self.b0_mesh_2)
        # B1 basis
        self.b1_basis_1 = B1SplineBasis(self.b1_mesh_1)
        self.b1_basis_2 = B1SplineBasis(self.b1_mesh_2)
        # kernel
        self.kernel_1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1/2), active_dims=[0])
        self.kernel_2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1/2), active_dims=[1])
        self.kernel = self.kernel_1 * self.kernel_2

    # - ASVGP PART - #
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
                    dimbasis : SplineBasis,
                    dimkernel : gpytorch.kernels,) -> torch.Tensor:
        """
        Computes the Kuu matrix for the Matérn 1/2 covarainces along a dimension 

        Arguments:
            dimbasis (FourierBasis)         : basis for the dimension
            dimkernel (gpytorch.kernels)    : kernel for the dimension

        Returns:
            (torch.Tensor)                  : Kuu matrix for the dimension
        """
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
                    dimbasis : SplineBasis,
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
        Kuu_1 = self._Kuu_along_dim(self.b1_basis_1, self.kernel_1)
        Kuu_2 = self._Kuu_along_dim(self.b1_basis_2, self.kernel_2)
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
        Kuf_1 = self._Kuf_along_dim(self.b1_basis_1, x[:, 0])
        Kuf_2 = self._Kuf_along_dim(self.b1_basis_2, x[:, 1])
        Kuf = torch.stack([k1 * k2 for k2 in Kuf_1 for k1 in Kuf_2], dim = 0)
        return Kuf.to(torch.float64)
    
    # - Gridded part - #
    def _Kvu_along_dim(self,
                       dimbasis : SplineBasis, # B1 basis
                       ) -> torch.Tensor:
        # compute the l2 inner product
        half_b1_basis_function_l2 = dimbasis.delta /2.
        full_b1_basis_function_l2 = dimbasis.delta 
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
        
    def _Kvu(self, ) -> torch.Tensor:
        """ """
        Kvu_1 = self._Kvu_along_dim(self.b1_basis_1)
        Kvu_2 = self._Kvu_along_dim(self.b1_basis_2)
        Kvu = torch.kron(Kvu_1, Kvu_2)
        return Kvu.to(torch.float64)


    def _Kvv_along_dim(self,
                       dimbasis : SplineBasis, # B0 basis
                       dimkernel : gpytorch.kernels, #
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
        m = dimbasis.m
        delta = dimbasis.delta
        scalesigma = dimkernel.outputscale
        lengthscale = dimkernel.base_kernel.lengthscale.squeeze()
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
        Kuu *= (lengthscale * 2 * scalesigma)
        return Kuu.to(torch.float64)
    
    def _Kvv(self, ) -> torch.Tensor:
        """ """
        Kvv_1 = self._Kvv_along_dim(self.b0_basis_1, self.kernel_1)
        Kvv_2 = self._Kvv_along_dim(self.b0_basis_2, self.kernel_2)
        Kvv = torch.kron(Kvv_1, Kvv_2)
        return Kvv.to(torch.float64)
        
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
#                                            GRIDDED                                               #
#                                                                                                  #
####################################################################################################

class Matern12GriddedGP(KroneckerStructure):

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


