"""This module should provide autograd functionality."""

import math


class ADiffFloat(object):
    """Implements an algorithimically differentiable floating point number class."""

    def _backward(self, seed=1.0):
        """Run a default backward function at the end of each path."""
        self.partial += seed

    def __init__(self, value: float) -> None:
        """Instantiate the differentiable float.

        We require to record the partial derivate as well as
        a backward function.

        Args:
            value (float): _description_
        """
        self.value = value
        self.partial = 0
        self.backward = self._backward

    def wrap_other(self, other):
        """Wrap a python-float as ADiffFloat."""
        if not isinstance(other, ADiffFloat):
            return ADiffFloat(other)
        else:
            return other

    def __add__(self, other):
        """Overloads the plus (+) operator."""
        other = self.wrap_other(other)
        result = ADiffFloat(self.value + other.value)

        def _backward(seed=1.0):
            self.backward(seed)
            other.backward(seed)

        result.backward = _backward

        return result

    def __mul__(self, other):
        """Overloads the multiplication (*) operator."""
        other = self.wrap_other(other)
        result = ADiffFloat(self.value * other.value)

        def _backward(seed=1.0):
            self.backward(other.value * seed)
            other.backward(self.value * seed)

        result.backward = _backward
        return result

    def relu(self):
        """Implement backprop through a relu."""
        result = ADiffFloat(self.value if self.value > 0 else 0.0)

        def _backward(seed=1.0):
            self.backward((self.value > 0) * seed)

        result.backward = _backward
        return result

    def sigmoid(self):
        """Implement backprop through a sigmoid function."""

        def _sig(x):
            return 1.0 / (1 + math.exp(-x))

        def _dsig(x):
            return _sig(x) * (1 - _sig(x))

        result = ADiffFloat(_sig(self.value))

        def _backward(seed=1.0):
            self.backward(_dsig(self.value) * seed)

        result.backward = _backward
        return result

    def __repr__(self):
        """Stop python from printing useless memory address values."""
        return f"ADiffFloat(Val: {self.value:2.2f}, Partial: {self.partial:2.2f})"

    def __neg__(self):
        """Multiplication with (1). Required for mse."""
        return self * -1

    def __radd__(self, other):
        """Allow right addition. Required for sum-operator support."""
        return self + other

    def __ladd__(self, other):
        """Allow left addition."""
        # Required for an easy life.
        return self + other

    def __sub__(self, other):
        """Support subtraction via negative addition."""
        return self + (-other)
