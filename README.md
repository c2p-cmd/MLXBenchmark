# MLX Framework Benchmark for K-Means Clustering

A performance comparison of Apple's MLX framework against PyTorch, NumPy, and scikit-learn for K-Means clustering on credit card customer data.

## Overview

This project evaluates MLX's computational efficiency and clustering quality for unsupervised learning tasks, benchmarking it against established ML frameworks.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all benchmarks
bash run_experiments.sh
```

## Project Structure

- `src/scripts/` - Implementation for each framework (MLX, PyTorch, NumPy, sklearn)
- `src/utils/` - K-Means implementations and timing utilities
- `src/notebooks/` - EDA and results comparison
- `src/results/` - Benchmark scores and metrics

## Frameworks Tested

- **Apple MLX** - GPU-accelerated ML for Apple Silicon
- **PyTorch** - Industry-standard deep learning
- **NumPy** - Pure Python implementation
- **scikit-learn** - Classical ML baseline

## Metrics

- Runtime performance
- Memory efficiency
- Clustering quality (ARI, NMI, Silhouette Score)

## Conceptual Framework

[File](./conceptual_framework.mermaid)

![conceptual_framework](./Conceptual%20Framework-2025-11-08-185317.png)

## Results

![results](./src/results/evaluation_metrics.png)
