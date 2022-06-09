from typing import Callable, Optional
import numpy as np


def simplex_projection(x: np.ndarray) -> np.ndarray:
    """Projects the vector x onto the unit simplex. Afterwards, it is guaranteed that:

        sum(x) == 1 & x >= 0

    We use the code presented in https://arxiv.org/abs/1101.6081,
    reference implementation https://github.com/lanha/SupResPALM/blob/master/reproject_simplex/projsplx.m

    It's better to look at the reference implementation than the paper.
    """
    assert len(x.shape) == 1
    n = x.size
    x_sorted = -np.sort(-x)

    found_t_hat = False
    current_sum = 0
    t_i = None
    for i in range(0, n-1):
        current_sum += x_sorted[i]
        t_i = (current_sum - 1) / (i + 1)
        if t_i >= x_sorted[i + 1]:
            found_t_hat = True
            break

    if not found_t_hat:
        t_i = (current_sum + x_sorted[n-1] - 1) / n

    assert t_i is not None
    result = np.maximum(x - t_i, 0)
    assert 1 - result.sum() < 0.01
    return result


class FixedStepGradientDescent():
    """
    A simplified implementation of gradient descent that tries to solve the following optimization problem:
        min_{x in Q} f(x)

    By the following strategy:
        x_{k+1} = x_k - alpha * gradient(f(x_k))

    Where alpha is a fixed step size.
    """

    def __init__(self, f: Callable[[np.ndarray], float], gradient: Callable[[np.ndarray], np.ndarray], projection: Optional[Callable[[np.ndarray], np.ndarray]], step_size: float) -> None:
        """
        Initializes the gradient descent method.

        f:
            function to minimize. Must take a vector as input and return a scalar, which is the value that we minimize.
        gradient:
            gradient function of f. Must take in a vector (the current value of x we are at) and return the gradient of f at x as a vector.
        projection:
            a projection function for the feasible set. It must hold that projection(x) in Q for all x in R^d.
            If None is given, uses identity.
        """
        self._f = f
        self._gradient = gradient
        self._projection = projection if projection is not None else lambda x: x
        self._step_size = step_size

    def step(self, x: np.ndarray, step_size: float) -> np.ndarray:
        """
        Returns the next step of the gradient descent.
        """
        return self._projection(x - step_size * self._gradient(x))

    def objective(self, x: np.ndarray) -> float:
        """
        Returns the value of the function at x.
        """
        return self._f(x)

    def minimize(self, x_0: np.ndarray, epsilon: float = 1e-6, max_iter: int = 1000) -> tuple[np.ndarray, float]:
        """
        Minimizes the function f starting from x_0. Runs until max_iter is reached or
        the difference in function value between two iterations is smaller than the given epsilon.

        Returns the minimizer and the value of the function at the minimizer.
        """
        x_i = x_0
        f_i = self._f(x_i)
        for i in range(max_iter):
            x_next = self.step(x_i, self._step_size)
            f_next = self._f(x_next)
            if np.abs(self._f(x_i) - self._f(x_next)) < epsilon:
                break
            else:
                x_i = x_next
                f_i = f_next
        return x_i, f_i
