import numpy as np

from numpy.typing import NDArray, ArrayLike
from typing import Optional, Self


class KMeans_NP:
    def __init__(self, n_clusters=3, max_iters=300, random_state=None):
        """Initialize K-means clustering estimator.

        Parameters
        ----------
        n_clusters : int, default=3
            The number of clusters to form.
        max_iters : int, default=300
            Maximum number of iterations of the k-means algorithm for a single run.
        random_state : int, optional
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids: Optional[NDArray] = None

    @property
    def cluster_centers_(self) -> NDArray:
        """The coordinates of the cluster centers, shape (n_clusters, n_features)."""
        if self.centroids is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'cluster_centers_' until fit is called."
            )
        return self.centroids

    def fit(self, X: ArrayLike) -> Self:
        """Fit the K-means clustering model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : KMeans_NP
            Fitted estimator.
        """
        # Convert to numpy array to ensure consistent type
        X = np.asarray(X)

        if self.random_state:
            np.random.seed(self.random_state)

        # 1. Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx].astype(X.dtype)  # Ensure centroids match data type

        for _ in range(self.max_iters):
            # 2. Assignment Step
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            # 3. Update Step
            new_centroids = np.array(
                [X[labels == k].mean(axis=0) for k in range(self.n_clusters)]
            )

            # 4. Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # The final self.centroids array has the shape (K, C)
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the closest cluster for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.centroids is None:
            raise AttributeError("Model has not been fitted yet. Call .fit() first.")

        X = np.asarray(X)
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: ArrayLike) -> NDArray:
        """Fit the model and predict cluster labels for the input data.

        This is a convenience method that combines fit() and predict() in one call.

        Parameters
        ----------
        X : array-like
            Training data to fit the model and predict labels for.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).predict(X)

    def _compute_distances(self, X) -> NDArray:
        """Compute pairwise Euclidean distances between samples and centroids.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        distances : ndarray of shape (n_samples, n_clusters)
            Euclidean distance from each sample to each centroid.
        """
        X = np.asarray(X)
        if self.centroids is None:
            raise AttributeError("Centroids are not initialized.")
        centroids = np.asarray(self.centroids)
        # compute pairwise euclidean distances between each sample and each centroid
        # result shape: (n_samples, n_clusters)
        return np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
