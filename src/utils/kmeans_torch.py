import torch
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Optional, Self


class KMeans_Torch:
    def __init__(self, n_clusters=3, max_iters=300, random_state=None):
        """Initialize K-means clustering estimator (PyTorch/MPS version).

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
        self.centroids: Optional[torch.Tensor] = None

        # Set the device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():  # For future-proofing
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @property
    def cluster_centers_(self) -> torch.Tensor:
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
            Training data to cluster. Assumed to be a NumPy array.

        Returns
        -------
        self : KMeans_Torch
            Fitted estimator.
        """
        # Convert to float tensor and move to device
        X_tensor = torch.from_numpy(np.asarray(X)).float().to(self.device)

        if self.random_state:
            torch.manual_seed(self.random_state)

        # 1. Initialize centroids randomly
        # Use torch.randperm for efficient random selection on device
        idx = torch.randperm(len(X_tensor), device=self.device)[: self.n_clusters]
        self.centroids = X_tensor[idx]

        for _ in range(self.max_iters):
            # 2. Assignment Step
            distances = self._compute_distances(X_tensor)
            labels = torch.argmin(distances, dim=1)

            # 3. Update Step (GPU-optimized)
            # Use one-hot encoding and matrix multiplication for a parallel update
            labels_one_hot = torch.nn.functional.one_hot(
                labels, self.n_clusters
            ).float()

            # Sum of points in each cluster
            # (n_clusters, n_samples) @ (n_samples, n_features) -> (n_clusters, n_features)
            new_centroids = labels_one_hot.T @ X_tensor

            # Count of points in each cluster
            counts = labels_one_hot.sum(dim=0).unsqueeze(1)

            # Handle empty clusters to avoid division by zero (results in NaN, which is fine)
            new_centroids = new_centroids / (counts + 1e-8)  # Add epsilon for stability

            # 4. Check for convergence
            if torch.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X: ArrayLike) -> torch.Tensor:
        """Predict the closest cluster for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict. Assumed to be a NumPy array.

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
            Tensor of cluster indices.
        """
        if self.centroids is None:
            raise AttributeError("Model has not been fitted yet. Call .fit() first.")

        # Convert to float tensor and move to device
        X_tensor = torch.from_numpy(np.asarray(X)).float().to(self.device)

        distances = self._compute_distances(X_tensor)
        return torch.argmin(distances, dim=1)

    def fit_predict(self, X: ArrayLike) -> torch.Tensor:
        """Fit the model and predict cluster labels for the input data.

        Parameters
        ----------
        X : array-like
            Training data to fit the model and predict labels for.

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
            Tensor of cluster indices.
        """
        return self.fit(X).predict(X)

    def _compute_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between samples and centroids.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data (must be on the correct device).

        Returns
        -------
        distances : torch.Tensor of shape (n_samples, n_clusters)
            Euclidean distance from each sample to each centroid.
        """
        if self.centroids is None:
            raise AttributeError("Centroids are not initialized.")

        # Use broadcasting to compute pairwise distances
        # X shape: (n_samples, n_features) -> (n_samples, 1, n_features)
        # centroids shape: (n_clusters, n_features) -> (1, n_clusters, n_features)
        X_expanded = X.unsqueeze(1)
        centroids_expanded = self.centroids.unsqueeze(0)

        # Resulting shape: (n_samples, n_clusters)
        return torch.linalg.norm(X_expanded - centroids_expanded, dim=2)
