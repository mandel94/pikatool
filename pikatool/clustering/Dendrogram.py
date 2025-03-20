from sklearn.metrics import DistanceMetric
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd


class Dendrogram:
    def __init__(
        self,
        data: pd.DataFrame,
        distance_metric: str = "jaccard",
        linkage: str = "complete",
    ):
        """
        Initializes the dendrogram with the given data and parameters.

        :param data: DataFrame containing the data
        :param distance_metric: Distance metric for calculating distances (default: "jaccard")
        :param linkage: Linkage method (default: "complete")
        """
        self.data = data
        self.distance_metric = distance_metric
        self.linkage = linkage
        self.dist_matrix = None
        self.linkage_matrix = None
        self.labels = None

        self._compute_distance_matrix()
        self._compute_linkage_matrix()
        self._prepare_labels()

    def _compute_distance_matrix(self):
        """Computes the distance matrix."""
        dist = DistanceMetric.get_metric(self.distance_metric)
        self.dist_matrix = dist.pairwise(self.data)
        self.dist_matrix = pd.DataFrame(
            self.dist_matrix, index=self.data.index, columns=self.data.index
        )

    def _compute_linkage_matrix(self):
        """Computes the linkage matrix for hierarchical clustering."""
        self.linkage_matrix = sch.linkage(
            self.dist_matrix, method=self.linkage
        )

    def _prepare_labels(self):
        """Prepares the labels by removing spaces."""
        self.labels = self.data.index.str.replace(" ", "")

    def plot(
        self, title: str = "Dendrogram", figsize=(18, 12), font_size: int = 10
    ):
        """Generates and displays the dendrogram."""
        fig, ax = plt.subplots(figsize=figsize)
        sch.dendrogram(
            self.linkage_matrix,
            labels=self.labels,
            orientation="right",
            leaf_font_size=font_size,
            color_threshold=-5,
            above_threshold_color="black",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontsize=15)
        plt.xlim(0.7, plt.xlim()[1])
        plt.show()
