Dataset Shape: X: (5000, 784), y: (5000,)
Running 15 rounds of experiments...

Running tests...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:50<00:00,  3.34s/round]

============================================================
AVERAGE RESULTS ACROSS ALL ROUNDS
============================================================

Mean Metrics:
             sklearn     numpy       mlx
ARI         0.364956  0.365663  0.377599
NMI         0.487949  0.490334  0.496023
Silhouette  0.061057  0.062214  0.062716
Time (s)    0.121208  2.149903  0.612999

Standard Deviation:
             sklearn     numpy       mlx
ARI         0.030296  0.019787  0.025926
NMI         0.021063  0.014283  0.013703
Silhouette  0.008590  0.006984  0.005691
Time (s)    0.050568  0.961375  0.252413

What These Numbers Indicate

Clustering Quality:
MLX shows slightly higher ARI and NMI than scikit-learn and NumPy across all rounds. These metrics measure label agreement and mutual information, so your MLX version’s slightly higher values indicate stable centroid updates and consistent cluster assignment under random initialization.

Execution Time:
The MLX run consistently completes faster than your NumPy baseline but slower than scikit-learn. This behavior is expected—scikit-learn’s KMeans uses optimized C code and OpenMP parallelism. MLX makes use of Apple Silicon acceleration (Metal / AMX units), so it achieves good gains from vectorized computation, but kernel-level I/O and memory transfer overheads still add time.​

Silhouette Score:
Your silhouette values suggest moderately distinct but overlapping clusters (typical for MNIST). The small differences between frameworks validate numerical equivalence.
