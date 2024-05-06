from itertools import count
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import fmin


def line_search(start_position, update_direction, function, tolerance=1e-5):
    def line(offset):
        return function(start_position + offset * update_direction)

    offset = fmin(line, x0=0.0, xtol=tolerance, disp=False)
    return start_position + offset * update_direction


class Optimiser:
    def step(x, function):
        raise NotImplementedError()


class NewtonMethod(Optimiser):
    # 变量 x 代表的是参数向量 \(\theta\)，因为这里相当于是在求theta取什么的时候likelihood最小
    def step(x, function):
        # 那么就有疑惑了jacobian不是又要theta又要x吗，这里怎么只传了前者，因为x在self里（self.x）
        J = function.jacobian(x)
        H = function.hessian(x)
        
        # this is equivalent to -inv(H) @ J, but more numerically stable
        update_direction = -lstsq(H, J, rcond=-1)[0]

        return line_search(x, update_direction, function)


class SteepestDescent(Optimiser):
    def step(x, function):
        gradient = function.jacobian(x)
        #move toward the opposite direction of gradient
        update_direction = -gradient

        return line_search(x, update_direction, function)


def array_from_dict(d):
    return np.stack([d[i] for i in sorted(d)])


def optimise(start_position, tolerance, function, optimiser, max_iterations=None):
    # 这里的x是theta的意思，即x前的系数，这里learning step想要求得的东西
    x = {0: start_position}
    w = {0: function(start_position)}

    for i in count(start=0):
        x[i + 1] = optimiser.step(x[i], function)
        w[i + 1] = function(x[i + 1])

        if all(abs(x[i + 1] - x[i]) < tolerance) or (max_iterations is not None and i >= max_iterations):
            break
        print('Iteration {:4d}, Function {}'.format(i + 1, w[i + 1]))

    return array_from_dict(x), array_from_dict(w)
