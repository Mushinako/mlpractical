from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    _NDArrayShape = tuple[int, ...]
    _InputNDArray = np.ndarray[_NDArrayShape, np.dtype[np.float64]]
    _OutputNDArray = np.ndarray[_NDArrayShape, np.dtype[np.float64]]
    _MaskNDArray = np.ndarray[_NDArrayShape, np.dtype[np.bool_]]

seed = 22102017
rng = np.random.RandomState(seed)


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient: float):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0.0, "Penalty coefficient must be positive."
        self.coefficient = coefficient

    def __call__(self, parameter: _InputNDArray) -> float:
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * np.absolute(parameter).sum()

    def grad(self, parameter: _InputNDArray) -> _InputNDArray:
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * np.sign(parameter)

    def __repr__(self):
        return "L1Penalty({0})".format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient: float):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0.0, "Penalty coefficient must be positive."
        self.coefficient = coefficient

    def __call__(self, parameter: _InputNDArray) -> float:
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient / 2 * (parameter * parameter).sum()

    def grad(self, parameter: _InputNDArray) -> _InputNDArray:
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * parameter

    def __repr__(self):
        return "L2Penalty({0})".format(self.coefficient)
