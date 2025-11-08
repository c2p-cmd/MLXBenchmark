import time
import tracemalloc
from ..utils.kmeans_mlx import KMeans_MLX
from ucimlrepo import fetch_ucirepo
import numpy as np
import mlx.core as mx

# fetch dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# data (as pandas dataframes)
X = default_of_credit_card_clients.data.features
X.columns = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
y = default_of_credit_card_clients.data.targets

bill_cols = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]
pay_cols = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
payment_cols = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

X["avg_bill"] = X[bill_cols].mean(axis=1)
X["avg_payment"] = X[pay_cols].mean(axis=1)

X["credit_utilization"] = X["BILL_AMT1"] / (X["LIMIT_BAL"] + 1)
X["payment_ratio"] = X["PAY_AMT1"] / (X["LIMIT_BAL"] + 1)
X["num_delays"] = (X[payment_cols] > 0).sum(axis=1)
X = X[
    [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "LIMIT_BAL",
        "PAY_AMT1",
        "BILL_AMT1",
    ]
].to_numpy()

y = y.to_numpy()

X = mx.array(X)
y = mx.array(y)

print("KMeans (MLX) benchmarking...")
print("Dataset shape:", X.shape)
print()

start_time = time.time()
tracemalloc.start()

kmeans_mlx = KMeans_MLX(n_clusters=2, max_iters=500, random_state=42)
current_pre_fit, peak_pre_fit = tracemalloc.get_traced_memory()
labels = kmeans_mlx.fit_predict(X)
# Convert mlx array to numpy array
labels = np.array(labels)
fit_time = time.time() - start_time
current_post_fit, peak_post_fit = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"KMeans (MLX) fit time: {fit_time:.4f} seconds")
print(
    f"KMeans (MLX) pre-fit memory usage: {current_pre_fit / 10**6:.4f} MB; peak: {peak_pre_fit / 10**6:.4f} MB"
)
print(
    f"KMeans (MLX) current memory usage: {current_post_fit / 10**6:.4f} MB; peak: {peak_post_fit / 10**6:.4f} MB"
)

# Evaluate clustering performance
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    normalized_mutual_info_score,
)

ari_score = adjusted_rand_score(y.squeeze(), labels)
sil_score = silhouette_score(X, labels)
nmi_score = normalized_mutual_info_score(y.squeeze(), labels)

print(f"KMeans (MLX) Adjusted Rand Index: {ari_score:.4f}")
print(f"KMeans (MLX) Silhouette Score: {sil_score:.4f}")
print(f"KMeans (MLX) Normalized Mutual Information: {nmi_score:.4f}")

scores = {
    "name": "KMeans_MLX",
    "fit_time_seconds": fit_time,
    "pre_fit_memory_current_MB": current_pre_fit / 10**6,
    "pre_fit_memory_peak_MB": peak_pre_fit / 10**6,
    "post_fit_memory_current_MB": current_post_fit / 10**6,
    "post_fit_memory_peak_MB": peak_post_fit / 10**6,
    "adjusted_rand_index": ari_score,
    "silhouette_score": sil_score,
    "normalized_mutual_info": nmi_score,
}

np.savez("src/results/kmeans_mlx_benchmark_scores.npz", **scores)
