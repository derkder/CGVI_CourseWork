import numpy as np

from functions import ScalarFunction
from functions import Rosenbrock

class FiniteDifference(ScalarFunction):
    def __init__(self, h=1e-3):
        super(FiniteDifference, self).__init__()
        self.h = h

    def jacobian(self, x):
        # TO DO: Replace this with the finite difference approximation of the first derivatives.
        # Remember that this should be a vector of partial derivatives (e.g. a 2x1 vector for a 2D function)
        n = len(x)
        partial_derivatives = np.zeros((n, 1))

        for i in range(n):
            # add disturbance
            x_perturbed = x.copy()
            x_perturbed[i] += self.h

            # Calculate the finite difference for the i-th dimension
            partial_derivatives[i] = (self(x_perturbed) - self(x)) / self.h

        return partial_derivatives

    def hessian(self, x):
        raise NotImplementedError('finite difference has not been defined for hessian')


def finite_difference(function_type, *args, **kwargs):
    """Create an instance of a function with finite differences instead of analytical derivatives

    Examples:
        For a function that implements derivatives this uses finite differences instead. For a relatively smooth
        function you would expect the value to be comparable to the exact derivative.

        >>> rosenbrock = functions.Rosenbrock()
        >>> type(rosenbrock)
        functions.Rosenbrock
        >>> rosenbrock.jacobian(np.array([[1, 2]]).T)
        array([[-400],
               [ 200]])

        >>> rosenbrock = finite_difference(functions.Rosenbrock)
        >>> type(rosenbrock)
        numerical.RosenbrockFiniteDifference
        >>> rosenbrock.jacobian(np.array([[1, 2]]).T)
        array([[-400.0008006],
               [ 200.       ]])
    """
    name = '{}FiniteDifference'.format(function_type.__name__)
    finite_difference_function = type(name, (FiniteDifference, function_type), {})
    return finite_difference_function(*args, **kwargs)


class RosenbrockFiniteDifference(Rosenbrock, FiniteDifference):
    def jacobian(self, x):
        return FiniteDifference.jacobian(self, x)