from typing import Protocol
import numpy as np
from typing import List
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


class ClustAlgo(Protocol):
    """
    A protocol that defines the interface for clustering algorithms.

    This protocol outlines the methods `fit` and `predict`, which all clustering algorithms
    should implement. The `fit` method computes the clustering model, and the `predict` method
    assigns cluster labels to the data.
    """

    def fit(self, data: np.array) -> None:
        """
        Fits the model to the data.

        :param data: The dataset to fit the model on.
        :type data: Dataset
        """
        ...

    def predict(self, data: np.array) -> List[int]:
        """
        Predicts the cluster labels for the data.

        :param data: The dataset to predict the cluster labels for.
        :type data: Dataset
        :return: A list of predicted cluster labels.
        :rtype: List[int]
        """
        ...


class Hclust:
    """
    A class that implements hierarchical clustering using scipy's linkage function.

    This class provides functionality to perform hierarchical clustering and assign cluster labels
    using a variety of criteria such as `maxclust`, `inconsistent`, etc. The `fit` method computes the
    hierarchical clustering and stores the linkage matrix. The `predict` method uses the linkage matrix
    to form flat clusters.

    Attributes:
    -----------
    n_clusters : int
        The number of clusters to form. This is used in the `predict` method to determine how many clusters
        should be formed from the hierarchical tree.
    method : str
        The linkage method used for clustering (e.g., 'ward', 'single', 'complete').
    linkage_matrix : np.ndarray or None
        A linkage matrix that encodes the hierarchical clustering. It is computed by the `fit` method.
    """

    def __init__(self, n_clusters: int, method: str):
        """
        Initializes the hierarchical clustering model.

        :param n_clusters: The number of clusters to form.
        :type n_clusters: int
        :param method: The linkage method to use for clustering. Valid options include:
            - "ward"
            - "single"
            - "complete"
            - "average"
            - "centroid"
            - "median"
            - "weighted"
        :type method: str
        """
        self.n_clusters = n_clusters
        self.method = (
            method  # Linkage method (e.g., 'ward', 'single', 'complete')
        )
        self.linkage_matrix = None  # To store the hierarchical structure

    def fit(self, data: np.array, dist_metric: str = "euclidean") -> None:
        """
        Computes the hierarchical clustering and stores the linkage matrix.

        This method computes the pairwise distances between data points using
        `scipy.spatial.distance.pdist` and then applies `scipy.cluster.hierarchy.linkage`
        to perform hierarchical clustering.

        :param data: The input data for hierarchical clustering. It should be an array-like
            structure, such as a 2D NumPy array or a Pandas DataFrame with numeric features.
            The shape should be `(n_samples, n_features)`.
        :type data: Dataset
        :param dist_metric: The distance metric used to compute pairwise distances between the data points.
            The available options are those supported by `scipy.spatial.distance.pdist`.
            Some common ones include:
            - **"euclidean"**: Euclidean distance (L2 norm).
            - **"cityblock"**: Manhattan distance (L1 norm).
            - **"cosine"**: Cosine distance.
            - **"hamming"**: Hamming distance, etc.
            (default is "euclidean")
        :type dist_metric: str, optional
        :raises ValueError: If the input data is not a valid array-like structure or has incompatible dimensions.
        :return: None
        """
        distance_matrix = pdist(
            data, metric=dist_metric
        )  # Compute pairwise distances
        self.linkage_matrix = linkage(
            distance_matrix, method=self.method
        )  # Compute the linkage matrix

    def predict(self, criterion: str = "maxclust") -> List[int]:
        """
        Assigns cluster labels based on the computed hierarchical structure.

        This method applies the `scipy.cluster.hierarchy.fcluster` function to the linkage
        matrix to determine flat clusters based on the selected `criterion`.

        :param criterion: The method used to form flat clusters from the hierarchical tree.
        Based on `scipy.cluster.hierarchy.fcluster`, the available options are:
            - **"inconsistent"**: Forms clusters based on an inconsistency threshold.
            - **"distance"**: Ensures that clusters are formed where all points have a
              cophenetic distance below a threshold.
            - **"maxclust"**: Forms a specific number (`self.n_clusters`) of clusters.
            - **"monocrit"**: Uses a custom monotonic criterion for cluster formation.
            - **"maxclust_monocrit"**: Ensures no more than `self.n_clusters` clusters are
              formed while applying a monotonic criterion.
        :type criterion: str, optional (default is "maxclust")
        :return: A list of cluster labels, where each entry corresponds to a data point in the
            original dataset.
        :rtype: List[int]
        :raises ValueError: If the model has not been fitted before calling `predict()`.
        """
        if self.linkage_matrix is None:
            raise ValueError("Model must be fitted before predicting.")

        cluster_labels = fcluster(
            self.linkage_matrix, self.n_clusters, criterion=criterion
        )
        return (
            cluster_labels.tolist()
        )  # Convert NumPy array to a list of integers
