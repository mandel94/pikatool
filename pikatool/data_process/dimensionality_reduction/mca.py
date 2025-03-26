import prince


class MCA:
    def __init__(self, mca_config: dict):
        # TODO Provide reasonable default configs
        default_mca_config = {  # Defaults
            "n_components": 3,
            "n_iter": 3,
            "copy": True,
            "check_input": True,
            "engine": "sklearn",
            "random_state": 42,
            "one_hot": True,
            # The way MCA works is that it one-hot encodes the dataset, and then fits a correspondence analysis.
            # In case your dataset is already one-hot encoded, you can specify one_hot=False to skip this step.
        }
        mca_config = {**default_mca_config, **mca_config}
        self.mca = prince.MCA(**mca_config)

    def _validate_is_fitted(self):
        if not self.mca:
            raise ValueError("MCA is not fit yet")

    def fit(self, data):
        self.fit_data = data
        self.mca.fit(data)

    def get_eigenvalues(self):
        self._validate_is_fitted()
        return self.mca.eigenvalues_summary

    def get_row_coordinates(self):
        self._validate_is_fitted()
        return self.mca.row_coordinates(self.fit_data)
