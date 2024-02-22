"""Test the python function from src."""

import numpy as np
import pytest

from src.autograd import ADiffFloat


def _sigmoid(x):
    if isinstance(x, ADiffFloat):
        return x.sigmoid()
    else:
        return 1.0 / (1 + np.exp(-x))


def _relu(x):
    if isinstance(x, ADiffFloat):
        return x.relu()
    else:
        return (x > 0.0) * x


def _difference_quotient(fun, pos, h):
    return (fun(pos + h) - fun(pos)) / h


def test_add() -> None:
    """Check an example with an addition."""
    x = ADiffFloat(1.0)
    y = ADiffFloat(-1.0)
    z = x + y
    z.backward()
    assert np.allclose(x.partial, 1.0) and np.allclose(y.partial, 1.0)


def test_mul() -> None:
    """Check an example with a multiplication."""
    x = ADiffFloat(1.0)
    y = ADiffFloat(-1.0)
    z = x * y
    z.backward()
    assert np.allclose(x.partial, -1.0) and np.allclose(y.partial, 1.0)


def test_auto_diff() -> None:
    """See if expressions mixing add and mul work properly."""
    x = ADiffFloat(2.0)
    y = ADiffFloat(3.0)
    z = x * (x + y) + y * y
    z.backward()
    assert np.allclose(z.value, 19.0)
    assert np.allclose(x.partial, 7.0)
    assert np.allclose(y.partial, 8.0)


@pytest.mark.parametrize("pos", (-1.0, 0.0, 1.0, 2.0))
def test_sigmoid(pos) -> None:
    """Check backprop through a sigmoid."""
    x = ADiffFloat(pos)
    act_x = x.sigmoid()
    act_x.backward()
    h = 1e-6
    diff_quot = _difference_quotient(_sigmoid, pos, h)
    assert np.allclose(x.partial, diff_quot, atol=h)


@pytest.mark.parametrize("pos", (-1.0, 0.0, 1.0, 2.0))
def test_sigmoid_activation(pos) -> None:
    """Check backprop through a sigmoid and additional ops."""
    h = 1e-6

    def _test_fun(x):
        return (_sigmoid(-1.0 + x) + 2.0) * 5.0

    adf = ADiffFloat(pos)
    res = _test_fun(adf)
    res.backward()

    diff_quot = _difference_quotient(_test_fun, pos, h)
    assert np.allclose(adf.partial, diff_quot, atol=h)


@pytest.mark.parametrize("pos", (-1.0, 0.0, 1.0, 2.0))
def test_relu_activation(pos) -> None:
    """Check backprop through a relu and additional ops."""
    h = 1e-6

    def _test_fun(x):
        return (_relu(x * (-5.0)) + 2.0) * 2.0

    adf = ADiffFloat(pos)
    res = _test_fun(adf)
    res.backward()

    diff_quot = _difference_quotient(_test_fun, pos, h)
    assert np.allclose(adf.partial, diff_quot, atol=h)
