import numpy as np
import mlx.core as mx

from numpy.typing import NDArray
from typing import Optional, Self, Union


class KMeans_MLX:
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
        self.centroids: Optional[mx.array] = None

    @property
    def cluster_centers_(self):
        """The coordinates of the cluster centers, shape (n_clusters, n_features)."""
        if self.centroids is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'cluster_centers_' until fit is called."
            )
        return self.centroids

    def fit(self, X: Union[mx.array, NDArray]) -> Self:
        """Fit the K-means clustering model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : KMeans_MLX
            Fitted estimator.
        """
        # Convert to mlx array to ensure consistent type
        if isinstance(X, np.ndarray):
            X = mx.array(X)
        elif not isinstance(X, mx.array):
            if hasattr(X, "to_numpy"):
                X = mx.array(X.to_numpy())

        if self.random_state:
            mx.random.seed(self.random_state)

        # 1. Initialize centroids randomly
        idx = mx.random.permutation(len(X))[: self.n_clusters]
        self.centroids = X[idx].astype(X.dtype)  # Ensure centroids match data type

        _i = 0
        while _i < self.max_iters:
            _i += 1

            # 2. Assignment Step
            distances = self._compute_distances(X)
            labels = mx.argmin(distances, axis=1)

            # 3. Update Step
            new_centroids = []
            for k in range(self.n_clusters):
                # Create mask for cluster membership
                mask = mx.array(labels == k, dtype=mx.float32)
                cluster_size = mx.sum(mask)

                if cluster_size > 0:
                    # Weighted mean: multiply each point by mask and divide by cluster size
                    # mask is shape (n_samples,), need to reshape for broadcasting
                    weighted_sum = mx.sum(X * mask[:, mx.newaxis], axis=0)
                    new_centroids.append(weighted_sum / cluster_size)
                else:
                    # Keep the old centroid if cluster is empty
                    new_centroids.append(self.centroids[k])
            new_centroids = mx.stack(new_centroids)

            # 4. Check for convergence
            if mx.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X) -> mx.array:
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

        if isinstance(X, np.ndarray):
            X = mx.array(X)
        elif not isinstance(X, mx.array):
            if hasattr(X, "to_numpy"):
                X = mx.array(X.to_numpy())

        distances = self._compute_distances(X)
        return mx.argmin(distances, axis=1)

    def fit_predict(self, X) -> mx.array:
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
        if isinstance(X, np.ndarray):
            X = mx.array(X)
        elif not isinstance(X, mx.array):
            if hasattr(X, "to_numpy"):
                X = mx.array(X.to_numpy())

        return self.fit(X).predict(X)

    def _compute_distances(self, X) -> mx.array:
        """Compute pairwise Euclidean distances between samples and centroids.

        Parameters
        ----------
        X : mx.array of shape (n_samples, n_features)
            Input data (already converted to mx.array by caller).

        Returns
        -------
        distances : mx.array of shape (n_samples, n_clusters)
            Euclidean distance from each sample to each centroid.
        """
        if self.centroids is None:
            raise AttributeError("Centroids are not initialized.")
        # compute pairwise euclidean distances between each sample and each centroid
        # result shape: (n_samples, n_clusters)
        return mx.linalg.norm(
            X[:, mx.newaxis, :] - self.centroids[mx.newaxis, :, :], axis=2
        )
