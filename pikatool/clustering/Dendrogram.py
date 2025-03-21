from sklearn.metrics import DistanceMetric
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


class DendrogramConfig:
    def __init__(
        self,
        title: str = "Dendrogram",
        figsize: tuple = (18, 12),
        font_size: int = 10,
        orientation: str = "right",
        color_threshold: float = -5,
        above_threshold_color: str = "black",
        leaf_font_size: Optional[
            int
        ] = None,  # Optional, uses font_size if not provided
        xlim: tuple = (0.7, None),
        show_spines: bool = False,
    ):
        """
        Configuration class to hold dendrogram plot settings.

        :param title: Title of the dendrogram. Default is "Dendrogram".
        :param figsize: Tuple for the figure size (width, height). Default is (18, 12).
        :param font_size: Font size for the leaf labels. Default is 10.
        :param orientation: Orientation of the dendrogram ('left', 'right', 'top', 'bottom'). Default is "right".
        :param color_threshold: Threshold to color the branches above this threshold. Default is -5.
        :param above_threshold_color: Color for the branches above the color_threshold. Default is "black".
        :param leaf_font_size: Font size for leaf node labels. If None, will use font_size. Default is None.
        :param xlim: Limits for the x-axis. Default is (0.7, None).
        :param show_spines: Whether to show spines (borders) around the plot. Default is False.

        Example:
        >>> config = DendrogramConfig(title="Custom Dendrogram", figsize=(10, 8), font_size=12, show_spines=True)
        """
        self.title = title
        self.figsize = figsize
        self.font_size = font_size
        self.orientation = orientation
        self.color_threshold = color_threshold
        self.above_threshold_color = above_threshold_color
        self.leaf_font_size = (
            leaf_font_size or font_size
        )  # Use font_size if leaf_font_size is not provided
        self.xlim = xlim
        self.show_spines = show_spines

    def to_dict(self):
        """
        Converts the configuration to a dictionary.

        :return: A dictionary representation of the configuration.

        Example:
        >>> config = DendrogramConfig()
        >>> config.to_dict()
        {'title': 'Dendrogram', 'figsize': (18, 12), 'font_size': 10, 'orientation': 'right', 'color_threshold': -5,
           'above_threshold_color': 'black', 'leaf_font_size': 10, 'xlim': (0.7, None), 'show_spines': False}
        """
        return {
            "title": self.title,
            "figsize": self.figsize,
            "font_size": self.font_size,
            "orientation": self.orientation,
            "color_threshold": self.color_threshold,
            "above_threshold_color": self.above_threshold_color,
            "leaf_font_size": self.leaf_font_size,
            "xlim": self.xlim,
            "show_spines": self.show_spines,
        }


class Dendrogram:
    def __init__(
        self,
        data: pd.DataFrame,
        distance_metric: str = "jaccard",
        linkage: str = "complete",
    ):
        """
        Initializes the dendrogram with the given data and parameters.

        :param data: DataFrame containing the data. Each row represents an observation, and columns represent features.
        :param distance_metric: Distance metric for calculating distances (default: "jaccard"). Options: 'euclidean', 'cosine', 'jaccard', etc.
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
        self.labels = None

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
            self.dist_matrix, method=self.linkage
        )

    def plot(self, config: DendrogramConfig):
        """
        Generates and displays the dendrogram using the provided configuration.

        :param config: The configuration settings for the dendrogram plot.

        Example:
        >>> config = DendrogramConfig(title="Cluster Dendrogram", orientation="top")
        >>> dendrogram.plot(config)
        """
        fig, ax = plt.subplots(figsize=config.figsize)

        # Call dendrogram function with parameters from config
        sch.dendrogram(
            self.linkage_matrix,
            labels=self.labels,
            orientation=config.orientation,
            leaf_font_size=config.leaf_font_size,
            color_threshold=config.color_threshold,
            above_threshold_color=config.above_threshold_color,
        )

        # Hide spines if show_spines is False
        if not config.show_spines:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # Set the title and xlim from config
        ax.set_title(config.title, fontsize=15)
        plt.xlim(
            config.xlim[0], config.xlim[1] if config.xlim[1] else plt.xlim()[1]
        )

        plt.show()
