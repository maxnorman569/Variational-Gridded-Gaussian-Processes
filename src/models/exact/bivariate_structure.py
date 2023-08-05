# numerical import
import torch
# gpytorch imports
import gpytorch
from gpytorch.distributions import MultivariateNormal

# Note: The bivariate case is no different to the univariate case, this file is only added for consistency between the exact and sparese modules

class GP(gpytorch.models.ExactGP):
    """ Exact Gaussian Process Regression Model """

    def __init__(self, 
                 train_x : torch.Tensor, 
                 train_y : torch.Tensor,
                 likelihood : gpytorch.likelihoods) -> 'GP':
        """
        Exact Gaussian Process Regression Model.

        Arguments:
            train_x (torch.Tensor)              : (n x 1) training data input
            train_y (torch.Tensor)              : (n x 1) training data output
            likelihood (gpytorch.likelihoods)   : likelihood function

        Returns:
            None
        """
        super().__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.mean = gpytorch.means.ZeroMean()
        # kernel specified in child class

    def non_informative_initialise(self, 
                                lmbda: float, 
                                kappa: float) -> None:
        """
        Initialises the model by setting the hyperparameters.

        Arguments:
            x_std (torch.Tensor)    : standard deviation of the input data
            y_std (torch.Tensor)    : standard deviation of the output data
            lmbda (float)           : lengthscale hyperparameter
            kappa (float)           : noise hyperparameter (we expect kappa to be in the range of [2, 100])
        
        Returns:
            None
        """
        self.kernel.outputscale = self.train_y.var()
        self.likelihood.noise = self.mean.outputscale / (kappa ** 2)
        self.kernel.base_kernel.lengthscale = (self.train_x.std() / lmbda)
    
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
        self.kernel.outputscale = (torch.tensor(prior_amplitude) / 2) ** 2
        self.likelihood.noise = self.train_y.var() - self.mean.outputscale
        self.kernel.base_kernel.lengthscale = (self.train_x.std() / lmbda)

    def forward(self, 
                x : torch.Tensor) -> MultivariateNormal:
        """ 
        Takes input data and returns the predicted mean and covariance of the Gaussian process at those input points. 

        Arguments:
            x (torch.Tensor)    : input data

        Returns:
            MultivariateNormal  : multivariate normal conditioned on training data at input points
        """
        mean_x = self.mean(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def prior(self,
              x_star : torch.Tensor) -> MultivariateNormal:
        """
        Returns the prior distribution of the model over given test points.

        Arguments:
            x_star (torch.Tensor)   : test points

        Returns:
            MultivariateNormal      : prior distribution
        """
        return self.forward(x_star)
    
    def posterior(self,
                x_star : torch.Tensor) -> MultivariateNormal:
        """
        Returns the posterior distribution of the model over given test points.

        Arguments:
            x_star (torch.Tensor)   : test points

        Returns:
            MultivariateNormal      : posterior distribution
        """
        # set model and likelihood into eval mode
        self.eval()
        self.likelihood.eval()
        # make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = self(x_star)
        return f_pred

    def posterior_predictive(self,
                            x_star : torch.Tensor) -> MultivariateNormal:
        """
        Returns the posterior predictive distribution of the model over given test points.

        Arguments:
            x_star (torch.Tensor)   : test points

        Returns:
            MultivariateNormal      : posterior predictive distribution
        """
        # set model and likelihood into eval mode
        self.eval()
        self.likelihood.eval()
        # make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = self.likelihood(self(x_star))
        return f_pred
    

class Matern12GP(GP):
    """ Exact univariate Gaussian Process Regression Model with Matern 1/2 Kernel """
    def __init__(self, 
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor,
                 likelihood = gpytorch.likelihoods.GaussianLikelihood()) -> 'GP':
        super().__init__(train_x, train_y, likelihood)
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1/2))
        self.likelihood = likelihood


class Matern32GP(GP):
    """ Exact univariate Gaussian Process Regression Model with Matern 1/2 Kernel """
    def __init__(self, 
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor,
                 likelihood = gpytorch.likelihoods.GaussianLikelihood()) -> 'GP':
        super().__init__(train_x, train_y, likelihood)
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 3/2))
        self.likelihood = likelihood
    

class Matern52GP(GP):
    """ Exact univariate Gaussian Process Regression Model with Matern 1/2 Kernel """
    def __init__(self, 
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 likelihood = gpytorch.likelihoods.GaussianLikelihood()) -> 'GP':
        super().__init__(train_x, train_y, likelihood)
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 5/2))
        self.likelihood = likelihood