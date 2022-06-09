from optimization import *


def test_gradient_descent():
    def f(x): return x**2
    def gradient(x): return 2 * x
    gd = FixedStepGradientDescent(f, gradient, None, 0.1)
    x_min, f_min = gd.minimize(np.array([1]))
    assert np.sum(x_min - 0) < 0.01
    assert f_min < 0.01


def test_simplex_projection():
    assert np.allclose(simplex_projection(
        np.array([1.3, 1.3, 1.3])), np.array([1/3, 1/3, 1/3]))
    assert np.allclose(simplex_projection(
        np.array([0.3, 0.5, 0.2])), np.array([0.3, 0.5, 0.2]))
