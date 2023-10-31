import numpy as np
from scipy.special import expit as sigmoid


class ScalarFunction:
    """Interface for a scalar function that can be used with an optimiser"""

    def __call__(self, x):
        """Evaluate the function on the datapoint x, returning a scalar value"""
        raise NotImplementedError('function has not been defined')

    def jacobian(self, x):
        """Evaluate the first derivative of the function at the datapoint x, returning a vector of derivatives"""
        raise NotImplementedError('jacobian has not been defined')

    def hessian(self, x):
        """
        Evaluate the second derivative of the function at the datapoint x, returning a matrix of second derivatives
        """
        raise NotImplementedError('hessian has not been defined')


class Rosenbrock(ScalarFunction):
    """This is the Rosenbrock function (look it up!)

    It is a simple polynomial equation but it is quite hard to find the exact minimum!

    """
    def __call__(self, x):
        x1, x2 = x.squeeze()
        return 100 * (x2 - x1**2)**2 + (1 - x1)**2

    def jacobian(self, x):
        x1, x2 = x.squeeze()

        # TODO: Replace this by the analytical first derivatives of Rosenbrock's function
        derivative = np.ones((2, 1))

        assert derivative.shape == (2, 1), 'jacobian must be a 2x1 vector'
        return derivative

    def hessian(self, x):
        x1, x2 = x.squeeze()

        # TODO: Replace this by the analytical second derivatives of Rosenbrock's function
        second_derivative = np.ones((2, 2))

        assert second_derivative.shape == (2, 2), 'hessian must be a 2x2 matrix'
        return second_derivative


class LogisticRegressionNLL(ScalarFunction):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dims = x.shape[0]

    def __call__(self, phi):
        small_number = 1e-15
        probability = sigmoid(phi.T @ self.x)
        L = self.y * np.log(probability + small_number) + (1 - self.y) * np.log(1 - probability + small_number)
        return -L.sum()

    def jacobian(self, phi):
        # TODO: Replace this by the analytical first derivatives
        derivative = np.ones((self.dims, 1))

        assert derivative.shape == (self.dims, 1), 'jacobian must be a {0}x1 vector'.format(self.dims)
        return derivative

    def hessian(self, phi):
        # TODO: Replace this by the analytical second derivatives
        second_derivative = np.ones((self.dims, self.dims))

        assert second_derivative.shape == (self.dims, self.dims), 'hessian must be a {0}x{0} matrix'.format(self.dims)
        return second_derivative


# initialise an instance of Rosenbrock for convenience since it doesn't require any initial parameters
rosenbrock = Rosenbrock()
