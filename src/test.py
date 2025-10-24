import numpy as np
import pandas as pd
import mlx.core as mx

import time
from tqdm.auto import tqdm

from sklearn.cluster import KMeans
from utils.kmeans_numpy import KMeans_NP
from utils.kmeans_mlx import KMeans_MLX
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.datasets import fetch_openml

# Configuration
NUM_ROUNDS = 15  # Number of rounds to run
N_CLUSTERS = 10
SAMPLE_SIZE = 5_000

# Load and prepare data
data = fetch_openml("mnist_784", as_frame=True)
data = data.frame.sample(SAMPLE_SIZE, random_state=42)

X = data.drop(columns="class").to_numpy()
y = data["class"].to_numpy()

y_int = list(map(lambda x: int(x), y))

mlx_X = mx.array(X)
mlx_y = mx.array(y_int)

print(f"Dataset Shape: X: {X.shape}, y: {y.shape}")
print(f"Running {NUM_ROUNDS} rounds of experiments...\n")


def run_sklearn_timed(X, n_clusters, random_state):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    elapsed = time.time() - start_time
    return labels, elapsed


def run_numpy_timed(X, n_clusters, random_state):
    start_time = time.time()
    kmeans = KMeans_NP(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    elapsed = time.time() - start_time
    return labels, elapsed


def run_mlx_timed(X, n_clusters, random_state):
    start_time = time.time()
    kmeans = KMeans_MLX(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    elapsed = time.time() - start_time
    return labels, elapsed


# Storage for results across rounds
all_results = {
    "sklearn": {"times": [], "ari": [], "nmi": [], "silhouette": []},
    "numpy": {"times": [], "ari": [], "nmi": [], "silhouette": []},
    "mlx": {"times": [], "ari": [], "nmi": [], "silhouette": []},
}

# Store labels from last round for visualization
last_labels = {}

# Run multiple rounds
for round_num in tqdm(range(NUM_ROUNDS), desc='Running tests...', unit='round'):
    # Run each implementation
    labels_sklearn, time_sklearn = run_sklearn_timed(
        X, N_CLUSTERS, random_state=42 + round_num
    )
    labels_numpy, time_numpy = run_numpy_timed(
        X, N_CLUSTERS, random_state=42 + round_num
    )
    labels_mlx, time_mlx = run_mlx_timed(mlx_X, N_CLUSTERS, random_state=42 + round_num)

    # Convert MLX array to numpy for metric calculations
    labels_mlx_np = np.array(labels_mlx)

    # Calculate metrics
    all_results["sklearn"]["times"].append(time_sklearn)
    all_results["sklearn"]["ari"].append(adjusted_rand_score(y, labels_sklearn))
    all_results["sklearn"]["nmi"].append(
        normalized_mutual_info_score(y, labels_sklearn)
    )
    all_results["sklearn"]["silhouette"].append(silhouette_score(X, labels_sklearn))

    all_results["numpy"]["times"].append(time_numpy)
    all_results["numpy"]["ari"].append(adjusted_rand_score(y, labels_numpy))
    all_results["numpy"]["nmi"].append(normalized_mutual_info_score(y, labels_numpy))
    all_results["numpy"]["silhouette"].append(silhouette_score(X, labels_numpy))

    all_results["mlx"]["times"].append(time_mlx)
    all_results["mlx"]["ari"].append(adjusted_rand_score(y, labels_mlx_np))
    all_results["mlx"]["nmi"].append(normalized_mutual_info_score(y, labels_mlx_np))
    all_results["mlx"]["silhouette"].append(silhouette_score(X, labels_mlx_np))

    # Store last round labels for visualization
    if round_num == NUM_ROUNDS - 1:
        last_labels = {
            "sklearn": labels_sklearn,
            "numpy": labels_numpy,
            "mlx": labels_mlx,
        }

# Calculate average metrics
print("\n" + "=" * 60)
print("AVERAGE RESULTS ACROSS ALL ROUNDS")
print("=" * 60)

avg_metrics = {
    "sklearn": [
        np.mean(all_results["sklearn"]["ari"]),
        np.mean(all_results["sklearn"]["nmi"]),
        np.mean(all_results["sklearn"]["silhouette"]),
        np.mean(all_results["sklearn"]["times"]),
    ],
    "numpy": [
        np.mean(all_results["numpy"]["ari"]),
        np.mean(all_results["numpy"]["nmi"]),
        np.mean(all_results["numpy"]["silhouette"]),
        np.mean(all_results["numpy"]["times"]),
    ],
    "mlx": [
        np.mean(all_results["mlx"]["ari"]),
        np.mean(all_results["mlx"]["nmi"]),
        np.mean(all_results["mlx"]["silhouette"]),
        np.mean(all_results["mlx"]["times"]),
    ],
}

std_metrics = {
    "sklearn": [
        np.std(all_results["sklearn"]["ari"]),
        np.std(all_results["sklearn"]["nmi"]),
        np.std(all_results["sklearn"]["silhouette"]),
        np.std(all_results["sklearn"]["times"]),
    ],
    "numpy": [
        np.std(all_results["numpy"]["ari"]),
        np.std(all_results["numpy"]["nmi"]),
        np.std(all_results["numpy"]["silhouette"]),
        np.std(all_results["numpy"]["times"]),
    ],
    "mlx": [
        np.std(all_results["mlx"]["ari"]),
        np.std(all_results["mlx"]["nmi"]),
        np.std(all_results["mlx"]["silhouette"]),
        np.std(all_results["mlx"]["times"]),
    ],
}

df_metrics = pd.DataFrame(avg_metrics, index=["ARI", "NMI", "Silhouette", "Time (s)"])
df_std = pd.DataFrame(std_metrics, index=["ARI", "NMI", "Silhouette", "Time (s)"])

print("\nMean Metrics:")
print(df_metrics)
print("\nStandard Deviation:")
print(df_std)

# Use labels from last round for visualization
labels_sklearn = last_labels["sklearn"]
labels_numpy = last_labels["numpy"]
labels_mlx = last_labels["mlx"]

# Apply PCA for visualization (reduce MNIST from 784D to 2D)
pca = PCA(n_components=2, random_state=44)
X_pca = pca.fit_transform(X)

# Create Plotly subplots layout
fig = make_subplots(
    rows=5,
    cols=1,
    subplot_titles=[
        f"Average Metrics Comparison (over {NUM_ROUNDS} rounds)",
        "Standard Deviation",
        "KMeans (Scikit-Learn) - Last Round",
        "KMeans (NumPy) - Last Round",
        "KMeans (MLX) - Last Round",
    ],
    specs=[
        [{"type": "table"}],
        [{"type": "table"}],
        [{"type": "scatter"}],
        [{"type": "scatter"}],
        [{"type": "scatter"}],
    ],
    vertical_spacing=0.06,
)

# Add average metrics comparison table
fig.add_trace(
    go.Table(
        header=dict(
            values=["Metric", "Scikit-Learn", "NumPy", "MLX"],
            fill_color="lightblue",
            align="center",
            font=dict(size=12, color="black"),
        ),
        cells=dict(
            values=[
                df_metrics.index,
                df_metrics["sklearn"].round(4),
                df_metrics["numpy"].round(4),
                df_metrics["mlx"].round(4),
            ],
            align="center",
            font=dict(size=11),
        ),
    ),
    row=1,
    col=1,
)

# Add standard deviation table
fig.add_trace(
    go.Table(
        header=dict(
            values=["Metric (Std)", "Scikit-Learn", "NumPy", "MLX"],
            fill_color="lightyellow",
            align="center",
            font=dict(size=12, color="black"),
        ),
        cells=dict(
            values=[
                df_std.index,
                df_std["sklearn"].round(4),
                df_std["numpy"].round(4),
                df_std["mlx"].round(4),
            ],
            align="center",
            font=dict(size=11),
        ),
    ),
    row=2,
    col=1,
)


# Utility function for consistent scatter traces
def add_cluster_scatter(fig, X_2d, labels, title, row):
    labels_numeric = np.array(labels)
    labels_text = [str(label) for label in labels_numeric]

    fig.add_trace(
        go.Scattergl(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode="markers",
            marker=dict(
                color=labels_numeric,
                size=4,
                showscale=True,
                colorscale="Viridis",
                colorbar=dict(tickmode="linear", tick0=0, dtick=1),
            ),
            text=labels_text,
            hovertemplate="Cluster: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text=title, row=row, col=1)


# Add clustering result scatter plots
add_cluster_scatter(
    fig,
    X_pca,
    labels_sklearn,
    "Scikit-Learn Clusters",
    3,
)
add_cluster_scatter(
    fig,
    X_pca,
    labels_numpy,
    "NumPy Clusters",
    4,
)
add_cluster_scatter(
    fig,
    X_pca,
    labels_mlx,
    "MLX Clusters",
    5,
)

fig.update_layout(
    height=1600,
    showlegend=False,
    title_text=f"KMeans Comparison: Scikit-Learn vs NumPy vs MLX (Averaged over {NUM_ROUNDS} rounds)",
)

# Show interactive figure
fig.show()
