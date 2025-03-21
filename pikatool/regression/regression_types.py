from typing import Protocol
import numpy as np


class Regression(Protocol):
    """
    A protocol that defines the interface for regression algorithms.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the regression model to the input data.

        Parameters:
        X : np.ndarray
            Feature matrix (independent variables).
        y : np.ndarray
            Target variable (dependent variable).

        Returns:
        None
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable for the given data.

        Parameters:
        X : np.ndarray
            Feature matrix (independent variables) for which predictions are made.

        Returns:
        np.ndarray
            Predicted values for the target variable.
        """
        ...
