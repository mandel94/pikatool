from typing import Protocol
import numpy as np
from typing import List
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import pandas as pd


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

    def predict(self) -> List[int]:
        """
        Predicts the cluster labels for the data.

        :return: A list of predicted cluster labels.
        :rtype: List[int]
        """
        ...


class KMeansClust:
    """
    A class that implements KMeans clustering using scikit-learn's KMeans algorithm.

    This class provides functionality to perform KMeans clustering and assign cluster labels.
    The `fit` method computes the clustering model, and the `predict` method assigns cluster labels
    to the data points.

    Attributes:
    -----------
    n_clusters : int
        The number of clusters to form.
    kmeans_model : KMeans
        The KMeans model fitted on the data. It stores the cluster centers and the labels for each data point.
    distance_metric : str
        The distance metric to use for clustering. By default, it is "euclidean".
    """

    def __init__(self, n_clusters: int, distance_metric: str = "euclidean"):
        """
        Initializes the KMeans clustering model.

        :param n_clusters: The number of clusters to form.
        :type n_clusters: int
        :param distance_metric: The distance metric to use for clustering. Default is "euclidean".
        :type distance_metric: str
        """
        self.n_clusters = n_clusters
        self.distance_metric = (
            distance_metric  # Distance metric (e.g., 'euclidean', 'jaccard')
        )

    def fit(self, data: np.array) -> None:
        """
        Computes the KMeans clustering model.

        This method uses `KMeans` from scikit-learn to fit the clustering model on the data.

        :param data: The input data for clustering. It should be an array-like structure,
            such as a 2D NumPy array or a Pandas DataFrame with numeric features.
            The shape should be `(n_samples, n_features)`.
        :type data: np.ndarray
        :raises ValueError: If the input data is not a valid array-like structure or has incompatible dimensions.
        :return: None
        """
        if self.distance_metric == "jaccard":
            # For KMeans, Jaccard distance is not supported directly, so we need to compute pairwise distances.
            distance_matrix = pairwise_distances(data, metric="jaccard")
            # Apply clustering on the distance matrix (this approach may not be optimal with KMeans)
            # Convert the distance matrix into a form KMeans can work with
            # (this is a basic workaround and may not provide meaningful results with KMeans)
            data = (
                1 - distance_matrix
            )  # Convert Jaccard distance to similarity
            self.data = data
        self.kmeans_model = KMeans(n_clusters=self.n_clusters)
        self.kmeans_model.fit(data)

    def predict(self) -> List[int]:
        """
        Predicts the cluster labels for the data.

        This method assigns cluster labels based on the fitted KMeans model.

        :return: A list of predicted cluster labels, where each entry corresponds to a data point.
        :rtype: List[int]
        :raises ValueError: If the model has not been fitted before calling `predict()`.
        """
        if self.kmeans_model is None:
            raise ValueError("Model must be fitted before predicting.")
        cluster_labels = self.kmeans_model.predict(self.data)
        return (
            cluster_labels.tolist()
        )  # Convert NumPy array to a list of integers


class AggloClust:
    """
    A class that implements agglomerative (hierarchical) clustering using scikit-learn's AgglomerativeClustering algorithm.

    This class provides functionality to perform agglomerative clustering and assign cluster labels.
    The `fit` method computes the clustering model, and the `predict` method assigns cluster labels
    to the data points.

    Attributes:
    -----------
    n_clusters : int
        The number of clusters to form.
    linkage : str
        The linkage criterion to use for clustering (options: 'ward', 'complete', 'average', 'single').
    agglomerative_model : AgglomerativeClustering
        The AgglomerativeClustering model fitted on the data. It stores the cluster labels for each data point.
    distance_metric : str
        The distance metric to use for clustering. Default is 'euclidean', but can be changed to 'jaccard'.
    """

    def __init__(
        self,
        n_clusters: int,
        linkage: str = "ward",
        distance_metric: str = "euclidean",
    ):
        """
        Initializes the agglomerative clustering model.

        :param n_clusters: The number of clusters to form.
        :type n_clusters: int
        :param linkage: The linkage criterion to use. Valid options are:
            - "ward"
            - "complete"
            - "average"
            - "single"
        :type linkage: str
        :param distance_metric: The distance metric to use for clustering (default is "euclidean").
        :type distance_metric: str
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = (
            distance_metric  # Distance metric (e.g., 'euclidean', 'jaccard')
        )

    def fit(self, data: np.array) -> None:
        """
        Computes the agglomerative clustering model.

        This method uses `AgglomerativeClustering` from scikit-learn to fit the clustering model on the data.

        :param data: The input data for clustering. It should be an array-like structure,
            such as a 2D NumPy array or a Pandas DataFrame with numeric features.
            The shape should be `(n_samples, n_features)`.
        :type data: np.ndarray
        :raises ValueError: If the input data is not a valid array-like structure or has incompatible dimensions.
        :return: None
        """

        if isinstance(data, pd.DataFrame):
            data = data.values  # Convert DataFrame to NumPy array

        if self.distance_metric == "jaccard":
            # Compute pairwise Jaccard distances and pass it to AgglomerativeClustering
            distance_matrix = pairwise_distances(data, metric="jaccard")
            self.agglomerative_model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric="precomputed",
                linkage=self.linkage,
            )
            self.agglomerative_model.fit(distance_matrix)
        else:
            self.agglomerative_model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric=self.distance_metric,
                linkage=self.linkage,
            )
            self.agglomerative_model.fit(data)

    def predict(self) -> List[int]:
        """
        Predicts the cluster labels for the data.

        This method assigns cluster labels based on the fitted AgglomerativeClustering model.

        :return: A list of predicted cluster labels, where each entry corresponds to a data point.
        :rtype: List[int]
        :raises ValueError: If the model has not been fitted before calling `predict()`.
        """
        if self.agglomerative_model is None:
            raise ValueError("Model must be fitted before predicting.")
        cluster_labels = self.agglomerative_model.labels_
        return (
            cluster_labels.tolist()
        )  # Convert NumPy array to a list of integers
