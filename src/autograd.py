"""This module should provide autograd functionality."""

# import math


class ADiffFloat(object):
    """Implements an algorithimically differentiable floating point number class."""

    def _backward(self, seed=1.0):
        """Run a default backward function at the end of each path."""
        # Paths end here. TODO: Sum up contributions.
        self.partial += 0.

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

        # TODO: Replace the line below.
        result = ADiffFloat(0.0)
        # TODO: Implement the addition of the two floats self and other.
        # Set the self.backbard function
        # of the result such that both summands
        # end up with derivative information.

        return result

    def __mul__(self, other):
        """Overloads the multiplication (*) operator."""
        other = self.wrap_other(other)

        # TODO: Replace the line below.
        result = ADiffFloat(0.0)
        # TODO: Implement multiplication of the two floats self and other.
        # Set the self.backbard function
        # of the result such that gradient information
        # is backpropagated into both factors.

        return result

    def relu(self):
        """Implement backprop through a relu."""
        # TODO: Implement a relu forward pass below.
        # remeber relu(x) equals x if x > 0 else 0.
        result = ADiffFloat(0.0)

        # TODO: Set the backward function.
        # Relus propagate gradient information if their value is > 0.

        return result

    def sigmoid(self):
        """Implement backprop through a sigmoid function."""
        # TODO: Update the line below.
        result = ADiffFloat(0.0)
        # Remember the definition of the sigmoidal function:
        # sig(x) = 1.0 / (1 + exp(-x))

        # TODO: Set the backprop function correctly.
        # HINT: sig'(x) = sig(x) * (1 - _sig(x))
        # Remember to apply the chain rule.

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
