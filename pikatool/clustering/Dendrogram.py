import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import DistanceMetric
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from typing import Dict, Any, Optional
from pandas.api.types import is_string_dtype


class Dendrogram:
    def __init__(
        self,
        data: pd.DataFrame,
        distance_metric: str = "euclidean",
        linkage: str = "complete",
    ):
        """
        Initializes the dendrogram with the given data and parameters.

        :param data: DataFrame containing the data. Each row represents an observation, and columns represent features.
        :param distance_metric: Distance metric for calculating distances (default: "euclidean"). Options: 'euclidean', 'cosine', 'jaccard', etc.
        :param linkage: Linkage method for hierarchical clustering (default: "complete"). Options: 'single', 'complete', 'average', etc.

        Example:
        >>> df = pd.DataFrame([[0, 1], [1, 0], [0, 0], [1, 1]], columns=['Feature1', 'Feature2'])
        >>> dendrogram = Dendrogram(df)
        """
        self.data = data
        self.distance_metric = distance_metric
        self.linkage = linkage
        self.dist_matrix = None
        self.linkage_matrix = None
        if is_string_dtype(data.index):
            self.labels = data.index.str.replace(" ", "")
        else:
            self.labels = data.index
        self._compute_distance_matrix()
        self._compute_linkage_matrix()

    def _compute_distance_matrix(self):
        """
        Computes the distance matrix using the specified distance metric.

        This method calculates pairwise distances between observations in the data
        using the distance metric specified during the initialization.
        """
        dist = DistanceMetric.get_metric(self.distance_metric)
        self.dist_matrix = dist.pairwise(self.data)
        self.dist_matrix = pd.DataFrame(
            self.dist_matrix, index=self.data.index, columns=self.data.index
        )

    def _compute_linkage_matrix(self):
        """
        Computes the linkage matrix for hierarchical clustering.

        This method applies the hierarchical clustering using the linkage method
        (e.g., 'complete', 'average', etc.) on the computed distance matrix.
        """
        self.linkage_matrix = sch.linkage(
            squareform(self.dist_matrix),
            method=self.linkage,  # squareform ensures condensed distance matrix (statsmodels likes it)
        )

    def plot(self, config: Optional[Dict[str, Any]] = None):
        """
        Generates and displays the dendrogram using the provided configuration.

        :param config: The configuration settings for the dendrogram plot. It is a dictionary.

        Example:
        >>> config = {'title': 'Cluster Dendrogram', 'orientation': 'top'}
        >>> dendrogram.plot(config)
        """
        # Default configuration settings with reasonable values
        default_config = {
            "title": "Dendrogram",  # Default title
            "figsize": (
                18,
                12,
            ),  # Default size of the plot (wide aspect ratio for clarity)
            "font_size": 10,  # Reasonable font size for readability
            "orientation": "right",  # Default orientation for a clear left-to-right structure
            "color_threshold": -5,  # Reasonable threshold for color splitting
            "above_threshold_color": "black",  # Default color for branches above the threshold
            "leaf_font_size": 10,  # Font size for leaf labels
            "xlim": (None, None),  # Reasonable x-axis limit range
            "show_spines": False,  # By default, hide plot spines (borders)
        }

        # Merge provided config with defaults, overwriting default values with user input
        config = {} if not config else config
        config = {**default_config, **config}

        fig, ax = plt.subplots(figsize=config["figsize"])

        # Call dendrogram function with parameters from config
        sch.dendrogram(
            self.linkage_matrix,
            labels=self.labels,
            orientation=config["orientation"],
            leaf_font_size=config["leaf_font_size"],
            color_threshold=config["color_threshold"],
            above_threshold_color=config["above_threshold_color"],
        )

        # Hide spines if show_spines is False
        if not config["show_spines"]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # Set the title and xlim from config
        ax.set_title(config["title"], fontsize=15)
        plt.xlim(
            config["xlim"][0],
            config["xlim"][1] if config["xlim"][1] else plt.xlim()[1],
        )

        plt.show()
