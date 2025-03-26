import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
import statsmodels.api as sm
import pandas as pd
from .plot_utils import ConfidenceIntervalPlotter


class LogisticRegression:
    """
    A logistic regression model using statsmodels' Logit.

    This class allows fitting a logistic regression model, making predictions,
    retrieving marginal effects, confidence intervals for coefficients, and
    plotting confidence intervals.

    Attributes:
        model (sm.Logit): The logistic regression model instance.
        results (sm.LogitResults): The fitted model results.
        coefficients (pd.Series): The estimated coefficients.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from logistic_regression import LogisticRegression
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> logit_model = LogisticRegression()
        >>> logit_model.fit(y, X)
        >>> logit_model.get_marginal_effects()
        >>> logit_model.plot_confidence_interval()
    """

    def __init__(self):
        """Initializes the LogisticRegression class with placeholders for model attributes."""
        self.model = None
        self.results = None
        self.coefficients = None

    def _validate_fitted(self):
        """
        Validates if the model has been fitted.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.results is None:
            raise ValueError(
                "Model is not fitted. Please call 'fit' before predicting."
            )

    def _get_var_names(self) -> np.array:
        """
        Retrieves variable names from the fitted model, excluding the constant.

        Returns:
            np.array: Array of variable names.
        """
        self._validate_fitted()
        var_names = self.results.model.exog_names
        return np.array(var_names[1:])  # Exclude constant

    def fit(self, y: ArrayLike, X: ArrayLike) -> None:
        """
        Fits the logistic regression model using statsmodels.

        Args:
            y (ArrayLike): The target variable (binary).
            X (ArrayLike): The predictor variables.

        Returns:
            None
        """
        X = sm.add_constant(X)
        self.model = sm.Logit(y, X)
        self.results = self.model.fit()

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predicts probabilities using the fitted model.

        Args:
            X (ArrayLike): The predictor variables.

        Returns:
            ArrayLike: Predicted probabilities.
        """
        self._validate_fitted()
        X = sm.add_constant(X)
        return self.results.predict(X)

    def get_marginal_effects(self) -> pd.Series:
        """
        Computes and returns the marginal effects of the predictor variables.

        Returns:
            pd.Series: Marginal effects with variable names as index.
        """
        self._validate_fitted()
        var_names = self._get_var_names()
        marginal_effects = self.results.get_margeff().margeff
        return pd.Series(marginal_effects, index=var_names)

    def get_coefficient_interval(self) -> pd.DataFrame:
        """
        Retrieves the confidence intervals for model coefficients.

        Returns:
            pd.DataFrame: Confidence intervals for each coefficient.
        """
        self._validate_fitted()
        return self.results.conf_int().iloc[1:, :]  # Exclude constant

    def plot_confidence_interval(self, theme: Optional[dict] = None) -> None:
        """
        Plots the confidence intervals for the model coefficients.

        Returns:
            None
        """
        self._validate_fitted()
        ci_plotter = ConfidenceIntervalPlotter(
            self.get_coefficient_interval(), theme=theme
        )
        ci_plotter.plot()
