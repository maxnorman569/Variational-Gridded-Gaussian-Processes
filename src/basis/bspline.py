import torch

# - BASIS FUNCTIONS FOR B-SPLINES - #
class B0BasisFunction:
    """ Bm, 0(x) basis functions for B-spline of order 1 """
    def __init__(self,
                 m : int,
                 cm : float,
                 cm1) -> 'B0BasisFunction':
        """ Constructs the Bm,0(x) basis functions for a B-spline of order 1 """
        self.m = m
        self.cm = cm
        self.cm1 = cm1
        self.delta = cm1 - cm

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        """ Evaluates the Bm,0(x) basis functions at x according to the Cox-de Boor recursion """
        # get B_{m,0}(x) and b_{m+1,0}(x)
        B0 = torch.logical_and(x >= self.cm, x <= self.cm1) * 1
        return B0
    

class B1BasisFunction:
    """ Bm,1(x) basis functions for B-spline of order 1 """
    def __init__(self, 
                 m : int, 
                 vm : float, 
                 vm1 : float, 
                 vm2 : float) -> 'B1BasisFunction':
        """ Constructs the Bm,1(x) basis functions for a B-spline of order 1 """
        self.m = m
        self.vm = vm
        self.vm1 = vm1
        self.vm2 = vm2
        self.delta = vm1 - vm

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        """ Evaluates the Bm,1(x) basis functions at x according to the Cox-de Boor recursion """
        # get B_{m,0}(x) and b_{m+1,0}(x)
        B0 = torch.logical_and(x >= self.vm, x <= self.vm1) * 1
        B1 = torch.logical_and(x > self.vm1, x <= self.vm2) * 1
        return ((x - self.vm) / (self.vm1 - self.vm)) * B0 + ((self.vm2 - x) / (self.vm2 - self.vm1)) * B1
    

class B1LeftHalfBasisFunction: 
    """ Left half B0,1(x) basis functions for B-spline of order 1 """
    def __init__(self, 
                 m : int,
                 v0 : float, 
                 v1 : float,) -> 'B1BasisFunction':
        """ Constructs the left half B0,1(x) basis functions for a B-spline of order 1 """
        self.m = 0
        self.v0 = v0
        self.v1 = v1
        self.delta = v1 - v0

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        """ Evaluates the left half B0,1(x) basis functions at x according to the Cox-de Boor recursion """
        B1 = torch.logical_and(x >= self.v0, x < self.v1) * 1
        return ((self.v1 - x) / (self.v1 - self.v0)) * B1
    

class B1RightHalfBasisFunction:
    """ Right half BM,1(x) basis functions for B-spline of order 1 (M being the last basis function) """
    def __init__(self, 
                 m : int,
                 vm : float, 
                 vm1 : float,) -> 'B1BasisFunction':
        """ Constructs the right half BM,1(x) basis functions for a B-spline of order 1 """
        self.m = m
        self.vm = vm
        self.vm1 = vm1
    
    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        """ Evaluates the Bm,1(x) basis functions at x according to the Cox-de Boor recursion """
        B0 = torch.logical_and(x >= self.vm, x <= self.vm1) * 1
        return ((x - self.vm) / (self.vm1 - self.vm)) * B0


# - B-SPLINE BASIS - #
class SplineBasis:
    """ B-spline basis of order k """
    def __init__(self,
                mesh : torch.Tensor) -> 'SplineBasis':
        """ Constructs a B-spline of order k """
        # child needs to set self.order
        self.mesh = mesh
        self.m = mesh.size(0) - (self.order + 1 )
        self.delta = mesh[1] - mesh[0]
        # child needs to set self.basis

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        """ Evaluates the B-spline of order 1 at x """
        return torch.vstack([self.basis_functions[m](x) for m in range(len(self.basis_functions))])


class B0SplineBasis(SplineBasis):
    """ B-spline of order 0 """
    def __init__(self, mesh : torch.Tensor) -> SplineBasis:
        self.order = 0
        super().__init__(mesh)
        self.basis = [B0BasisFunction(m, self.mesh[m], self.mesh[m+1]) for m in range(self.m)]


class B1SplineBasis(SplineBasis):
    """ B-spline of order 1 """
    def __init__(self, mesh : torch.Tensor) -> 'B1SplineBasis':
        self.order = 1
        super().__init__(mesh)
        self.basis_functions = [B1LeftHalfBasisFunction(0, self.mesh[0], self.mesh[1])] + [B1BasisFunction(m+1, self.mesh[m], self.mesh[m+1], self.mesh[m+2]) for m in range(0, self.m)] + [B1RightHalfBasisFunction(self.m+2, self.mesh[-2], self.mesh[-1])]
        self.n_basis_functions = len(self.basis_functions)

    # # Computes the RKHS inner Product for the B-spline of order 1
    # # TODO: find a better place for this function
    # def rkhs_inner_product(self, 
    #                         band : int,
    #                         scalesigma : float, 
    #                         lengthscale : float) -> torch.Tensor:
    #     """ 
    #     Returns the matrices involved in the RKHS inner product (EXCLUDING HYPERPARAMETERS)
    #     Note: the hyperparemters featuer in the model methods _Kuu and _Kuf

    #     Arguments:
    #         band : int, the band of the matrix to return (0 for main diagonal, 1 for first upper and lower diagonal)
        
    #     """
    #     assert band in [0, 1], "band must be 0 or 1 for B-spline of order 1"

    #     # compute inner products
    #     if band == 0:
    #         # compute integral terms
    #         int1 = torch.ones(self.m) * (2. / self.delta)
    #         int2 = torch.ones(self.m) * ((2. / 3.) * self.delta)
    #         # boundary conditions
    #         bound_cond = torch.zeros(self.m)
    #         bound_cond[0] = 1.
    #         bound_cond[-1] = 1.
    #         # inner product
    #         inner_prod = (int1 / (2. * scalesigma)) + (int2 / (2. * lengthscale * scalesigma)) + (bound_cond / (2. * scalesigma))
    #         return torch.diag_embed(inner_prod.flatten(), offset=0)
    #     else:
    #         # compute integrals
    #         int1 = torch.ones(self.m - 1) * (-1. / self.delta)
    #         int2 = torch.ones(self.m - 1) * (self.delta / 6.)
    #         # boundary conditions
    #         # boundary conditions = 0!
    #         # inner product
    #         inner_prod = ((1. / (2. * scalesigma) ) * int1) + ((1 / (2. * lengthscale * scalesigma)) * int2)
    #         return torch.diag_embed(inner_prod, offset=1) + torch.diag_embed(inner_prod, offset=-1)