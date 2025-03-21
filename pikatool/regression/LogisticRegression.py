import numpy as np
from numpy.typing import ArrayLike
import statsmodels as sm
import pandas as pd


class LogisticRegression:
    def __init__(self):
        self.model = None
        self.results = None
        self.coefficients = None

    def _validate_fitted(self):
        """"""
        if not self.model:
            ValueError(
                "Model is not fitted. Please call 'fit' before predicting."
            )

    def _get_var_names(self) -> np.array:
        """"""
        var_names = self.results.model.exog_names
        var_names = np.array(var_names[1 : len(var_names)])  # REMOVE CONSTANT

    def fit(self, y: ArrayLike, X: ArrayLike) -> None:
        """"""
        X = sm.add_constant(X)
        self.model = sm.Logit(y, X)
        self.results = self.model.fit()

    def predict(self) -> pd.Series:
        """"""
        self._validate_fitted()
        var_names = self._get_var_names()
        marginal_effects = self.results.get_margeff().margeff
        return pd.Series(marginal_effects, index=var_names)

    def get_coefficient_interval(self) -> pd.Series:
        """"""
        self._validate_fitted()
        return self.results.conf_int().iloc[1:, :]  # REMOVE CONST
