# Evaluating Apple's MLX Framework for Machine Learning: A Comparative Study Using Unsupervised Clustering Algorithms

---

## Abstract

This research investigates the computational performance and methodological efficacy of Apple's MLX framework for machine learning applications on Apple Silicon hardware. The study implements KMeans clustering algorithms across four frameworks—MLX, scikit-learn, NumPy, and PyTorch—and evaluates their performance on the MNIST dataset. Initial empirical results from 20 experimental rounds with 7,500 samples demonstrate that while MLX achieves competitive clustering quality metrics (ARI: 0.378, NMI: 0.494, Silhouette: 0.064), it exhibits intermediate computational performance between scikit-learn's optimized implementation (0.163s) and pure NumPy (2.778s). The research addresses a gap in the literature, as existing benchmarks predominantly focus on deep learning and large language models, leaving traditional machine learning algorithms on Apple Silicon largely unexplored. This study provides empirical evidence for MLX's viability in unsupervised learning tasks and offers practical insights for researchers and practitioners considering Apple Silicon for machine learning workflows.

**Keywords:** Machine Learning Frameworks, Apple MLX, Unsupervised Learning, KMeans Clustering, Performance Benchmarking, Apple Silicon

---

## Chapter 1. Introduction

### 1.1 Background

The landscape of machine learning computation has undergone significant transformation with the emergence of specialized hardware architectures and optimized software frameworks. Apple's introduction of Apple Silicon processors (M-series chips) represents a paradigm shift in consumer-grade computing, featuring a unified memory architecture that integrates CPU and GPU memory into a shared pool of up to 192GB. This architectural innovation eliminates the traditional data transfer overhead between CPU and GPU memory spaces, theoretically offering substantial advantages for machine learning workloads.

In response to this hardware evolution, Apple developed MLX, an array framework specifically optimized for machine learning on Apple Silicon. Unlike established frameworks such as PyTorch and TensorFlow, which were initially designed for NVIDIA CUDA architectures and later adapted for Apple hardware through the Metal Performance Shaders (MPS) backend, MLX was purpose-built to leverage Apple Silicon's unique unified memory architecture from the ground up.

The machine learning community has extensively explored supervised learning techniques for various applications, particularly in customer analytics and predictive modeling. However, unsupervised learning methods, such as clustering algorithms, remain fundamental for exploratory data analysis, pattern discovery, and customer segmentation—especially in scenarios where labeled data is scarce, expensive, or unavailable. KMeans clustering, despite its simplicity, continues to be one of the most widely deployed unsupervised algorithms in production environments due to its interpretability, computational efficiency, and effectiveness across diverse domains.

### 1.2 Research Problem

Current literature on Apple Silicon and MLX framework performance predominantly focuses on deep learning applications, particularly large language models (LLMs) and transformer architectures. Studies such as those by Feng et al. on profiling Apple Silicon for LLM training and Gu et al.'s CONSUMERBENCH for generative AI applications provide valuable insights into MLX's performance for neural network training. However, a critical gap exists in understanding how MLX performs with traditional machine learning algorithms—specifically clustering techniques that form the foundation of many real-world analytical workflows.

This gap is particularly significant because:

1. **Algorithmic Differences:** Traditional clustering algorithms have fundamentally different computational patterns compared to deep learning models, relying heavily on iterative distance calculations, array operations, and memory access patterns rather than backpropagation and gradient computations.

2. **Production Relevance:** Many organizations continue to rely on classical machine learning algorithms for production systems due to their interpretability, lower computational requirements, and proven effectiveness for structured data analysis.

3. **Framework Maturity:** MLX's API differs substantially from NumPy and PyTorch, lacking certain operations (e.g., boolean indexing, direct array filtering) that are fundamental to implementing classical algorithms. The practical implications of these limitations remain underexplored.

4. **Hardware Utilization:** It remains unclear whether Apple Silicon's unified memory architecture provides tangible benefits for the memory access patterns typical of clustering algorithms, or whether the lack of specialized acceleration (e.g., tensor cores) represents a fundamental limitation.

### 1.3 Research Rationale

This research is important for several compelling reasons:

**Academic Contribution:** The study addresses a documented gap in the machine learning systems literature by providing empirical evidence on framework performance for classical algorithms on emerging hardware architectures. While extensive research exists on GPU acceleration for deep learning, systematic evaluation of Apple Silicon for traditional machine learning remains limited.

**Practical Implications:** With the proliferation of Apple Silicon devices in research and industry settings (over 100 million M-series Macs deployed as of 2024), understanding the performance characteristics and implementation challenges of MLX has immediate practical relevance for data scientists and machine learning engineers making infrastructure decisions.

**Methodological Innovation:** The research employs a rigorous multi-framework comparative approach, implementing identical KMeans algorithms across four distinct frameworks to isolate framework-specific performance characteristics while controlling for algorithmic variations.

**Reproducibility and Transparency:** By documenting specific implementation challenges (e.g., MLX's lack of boolean indexing) and providing open-source implementations, this research contributes to the broader goal of reproducible computational research and facilitates knowledge transfer within the machine learning community.

### 1.4 Research Objectives

The primary objectives of this research are:

**Objective 1:** Implement KMeans clustering algorithms using Apple's MLX framework, adapting to its unique API constraints and design philosophy.

**Objective 2:** Develop equivalent implementations across three baseline frameworks (scikit-learn, NumPy, PyTorch) to enable systematic performance comparison.

**Objective 3:** Evaluate clustering quality using standard metrics (Adjusted Rand Index, Normalized Mutual Information, Silhouette Score) to verify algorithmic correctness and consistency across implementations.

**Objective 4:** Benchmark computational performance (runtime, memory efficiency) across all frameworks using statistically robust experimental protocols with multiple rounds of execution.

**Objective 5:** Document implementation challenges, API differences, and practical considerations for developers considering MLX for machine learning workflows.

**Objective 6:** Provide empirical evidence and actionable recommendations for practitioners regarding the suitability of Apple Silicon and MLX for unsupervised learning tasks.

### 1.5 Research Questions

This research addresses the following specific questions:

**RQ1:** How does MLX's computational performance for KMeans clustering compare to established frameworks (scikit-learn, NumPy, PyTorch) on Apple Silicon hardware?

**RQ2:** Does MLX's implementation produce clustering results of comparable quality (measured by ARI, NMI, and Silhouette Score) to baseline implementations?

**RQ3:** What are the primary API-level challenges and constraints when implementing traditional machine learning algorithms in MLX, and how do these constraints impact development efficiency?

**RQ4:** What are the practical implications and recommendations for researchers and practitioners considering Apple Silicon and MLX for machine learning workflows involving unsupervised learning algorithms?

---

## Chapter 2. Literature Review

### 2.1 Introduction

This literature review synthesizes relevant research across three interconnected domains: (1) unsupervised learning and clustering methodologies, (2) machine learning framework performance and benchmarking, and (3) Apple Silicon architecture and the MLX framework. The review identifies the theoretical foundations, methodological precedents, and existing knowledge gaps that motivate the current research.

### 2.2 Unsupervised Learning and Clustering Algorithms

#### 2.2.1 KMeans Clustering: Foundations and Applications

KMeans clustering remains one of the most extensively deployed unsupervised learning algorithms in both academic research and industrial applications. Chen's comprehensive investigation of machine learning approaches for customer segmentation establishes KMeans as the most frequently utilized technique across telecommunications and e-commerce sectors, attributed to its computational efficiency and interpretable results.

The algorithm's prevalence in financial analytics is particularly noteworthy. Tran et al. explicitly employ KMeans clustering to segment customers in the banking sector, demonstrating that cluster-based segmentation can enhance downstream churn prediction when combined with supervised models. Their research reveals that for less flexible models such as Logistic Regression, implementing separate models per cluster yields higher AUC values compared to benchmark approaches without segmentation, suggesting that clustering captures meaningful heterogeneity in customer behavior.

#### 2.2.2 Clustering Validation Metrics

The evaluation of clustering quality presents unique challenges due to the absence of ground truth labels in typical unsupervised scenarios. However, when labels exist for validation purposes, external metrics provide rigorous assessment frameworks.

**Adjusted Rand Index (ARI):** This metric measures the similarity between two clusterings while correcting for chance agreement. Rijnen's research on hybrid supervised-unsupervised models utilizes ARI to assess the consistency between clustering results and known churn labels, establishing it as a standard metric for evaluating cluster-label correspondence.

**Normalized Mutual Information (NMI):** NMI quantifies the mutual dependence between cluster assignments and true labels, normalized to account for varying cluster numbers. Bose and Chen employ NMI in their investigation of hybrid clustering approaches for churn prediction, demonstrating its utility in comparing clustering solutions with known categorical outcomes.

**Silhouette Score:** As an internal validation metric, the Silhouette Coefficient measures how similar an object is to its own cluster versus other clusters. Chen's segmentation study identifies this as the most frequently employed internal metric, noting its ability to assess cluster cohesion and separation without requiring external labels.

### 2.3 Machine Learning Framework Performance and Benchmarking

#### 2.3.1 Framework Comparison Methodologies

Bahrampour et al. establish methodological precedents for comparative deep learning framework studies, evaluating Caffe, Neon, TensorFlow, Theano, and Torch across metrics including speed, hardware utilization, and ease of implementation. Their protocol—averaging results over 20-1000 iterations with initial warm-up runs—provides a template for rigorous performance benchmarking.

For traditional machine learning, Pedregosa et al.'s foundational work on scikit-learn establishes this library as the de facto baseline for algorithm implementations in Python, emphasizing consistency, documentation, and computational efficiency. Scikit-learn's extensive optimization and mature codebase make it an essential benchmark for any new framework claiming to offer competitive performance.

#### 2.3.2 Statistical Robustness in Benchmarking

The reviewed literature emphasizes the importance of statistical rigor in performance evaluation. Studies of AutoML frameworks generate sets of 10 random seeds to ensure that performance metrics represent averages across multiple samples, accounting for initialization variability. This approach directly informs the present study's decision to conduct 20 rounds of experimentation with varying random seeds.

### 2.4 Apple Silicon Architecture and MLX Framework

#### 2.4.1 Unified Memory Architecture

Feng et al. provide the most comprehensive analysis of Apple Silicon's performance characteristics for machine learning workloads. Their profiling of M1/M2 processors reveals that the unified memory architecture—integrating CPU and GPU memory into a shared pool of up to 192GB—theoretically eliminates data transfer overhead between devices. This architectural advantage proves particularly beneficial for large-scale language model training where memory requirements exceed typical GPU VRAM limits.

However, Feng et al. also document significant challenges:

**Memory Management Issues:** Apple Silicon exhibits gradual memory consumption increases during training, with continuously increasing page faults indicating suboptimal memory management. When training near memory capacity, page fault frequency increases substantially, causing performance instability.

**BLAS Performance Gap:** Basic Linear Algebra Subprograms (BLAS) operations on Apple Silicon demonstrate inferior performance compared to optimized NVIDIA CUDA kernels, particularly for FP16 operations. While NVIDIA achieves 5x-6x acceleration with FP16 via Tensor Cores, MLX realizes only 20-30% improvement for certain operations.

#### 2.4.2 MLX Framework Characteristics

Hannun et al.'s foundational work introduces MLX as an array framework designed specifically for Apple Silicon, emphasizing its lazy evaluation model and unified memory utilization. The framework's design philosophy prioritizes eliminating data transfer overhead by allowing arrays to be operated across devices without explicit copying.

#### 2.4.3 Existing MLX Performance Studies

Current MLX benchmarking focuses predominantly on generative AI applications:

**LLM Performance:** Chaplia and Klym evaluate small quantized language models (1B-1.5B parameters) on Apple Silicon using MLX-LM, concluding that 4-bit quantized models strike a favorable balance of responsiveness and memory efficiency for local deployment. Their findings establish MLX's viability for inference workloads but do not address training performance or classical algorithms.

**Concurrent Workloads:** Gu et al.'s CONSUMERBENCH evaluates generative AI applications on consumer devices, including MacBook M1 Pro, under concurrent execution scenarios. Their study confirms that MLX accelerates specific workloads (e.g., Whisper-Large-v3-turbo) but focuses exclusively on deep learning architectures.

**Kernel-Level Analysis:** Feng et al. conduct detailed BLAS kernel benchmarking, revealing that MLX kernels achieve approximately 30-40% speed advantages over PyTorch MPS for certain operations when using FP32, but the advantage diminishes or reverses for FP16 operations due to limited hardware optimization.

### 2.5 Critical Gap Analysis

The comprehensive literature review reveals a substantial methodological gap: **existing benchmarks of Apple Silicon and MLX overwhelmingly focus on deep learning architectures (LLMs, CNNs, transformers) while traditional machine learning algorithms—particularly clustering methods—remain unexplored in this context.**

Specific gaps include:

1. **Algorithmic Scope:** No reviewed studies benchmark KMeans algorithm on MLX or comprehensively evaluate Apple Silicon for non-neural network workloads.

2. **API Usability:** While Feng et al. analyze low-level kernel performance, no studies document the practical challenges of implementing traditional algorithms given MLX's API constraints (e.g., lack of boolean indexing).

3. **Structured Data Performance:** Existing benchmarks focus on unstructured data (text, images) processed by neural networks, leaving performance characteristics for structured tabular data—common in business analytics—unexamined.

4. **Statistical Validation:** LLM benchmarks typically report single-run metrics or focus on throughput/latency, lacking the multi-round statistical validation protocols necessary for assessing algorithmic consistency and stability.

### 2.6 Theoretical Framework

This research adopts a **comparative empirical evaluation framework** grounded in experimental computer science methodology. The theoretical approach integrates:

**Computational Complexity Theory:** KMeans clustering exhibits O(n·k·i·d) time complexity where n = samples, k = clusters, i = iterations, d = dimensions. This theoretical foundation enables predictions about relative performance across frameworks and dataset sizes.

**Statistical Learning Theory:** The use of external validation metrics (ARI, NMI) against known labels and internal metrics (Silhouette Score) draws from cluster validation theory, providing rigorous assessment of clustering quality independent of computational performance.

**Systems Performance Evaluation:** Following Bahrampour et al.'s framework benchmarking methodology, the research employs controlled experimental protocols with warm-up iterations, multiple repetitions, and statistical aggregation to ensure reliable performance measurements.

### 2.7 Summary

The literature establishes KMeans clustering as a fundamental unsupervised learning algorithm with proven efficacy across financial, telecommunications, and e-commerce domains. While comprehensive benchmarking protocols exist for deep learning frameworks, and substantial research characterizes Apple Silicon's performance for neural network training, a critical gap exists regarding traditional machine learning algorithms on MLX. This research addresses this gap through systematic implementation, rigorous benchmarking, and comprehensive evaluation of KMeans clustering across multiple frameworks on Apple Silicon hardware.

---

## Chapter 3. Methodology and Method

### 3.1 Research Approach

This research employs a **quantitative experimental approach** utilizing comparative performance evaluation methodology. The study is fundamentally empirical, measuring and comparing computational performance and algorithmic quality metrics across multiple machine learning framework implementations. This approach is appropriate for addressing the research questions because it enables:

1. **Objective Measurement:** Quantitative metrics (runtime, ARI, NMI, Silhouette Score) provide objective, reproducible assessments of framework performance and clustering quality.

2. **Controlled Comparison:** By implementing identical algorithms across frameworks and executing them on the same hardware with identical datasets, the study isolates framework-specific performance characteristics.

3. **Statistical Rigor:** Multiple experimental rounds with varying random seeds enable statistical analysis of performance variability and ensure findings are not artifacts of specific initializations.

4. **Reproducibility:** Quantitative approaches with well-documented protocols facilitate independent verification and extension of findings by other researchers.

The research does not employ qualitative or mixed methods, as the primary objectives concern measurable computational performance and algorithmic correctness rather than subjective user experiences or contextual interpretations.

### 3.2 Conceptual Framework

The research is guided by a four-component conceptual framework:

#### 3.2.1 Project Inputs

- **Data Source:** Credit Card Dataset from UCI ML repo comprising of 30k data points with 23 attributes about the individuals.
- **Ground Truth:** Yes/No whether the individual defaulted on their next month's payment or not.

#### 3.2.2 Experimental Processes

Four parallel implementation tracks:

1. **MLX Framework:** Custom KMeans implementation leveraging MLX's array operations and Apple Silicon optimization.
2. **scikit-learn:** Baseline using scikit-learn's highly optimized `KMeans` class, representing production-grade implementation quality.
3. **NumPy:** Pure NumPy implementation without high-level optimizations, establishing a baseline for unoptimized Python-based computation.
4. **PyTorch:** Implementation using PyTorch tensors with MPS (Metal Performance Shaders) backend for GPU acceleration on Apple Silicon.

#### 3.2.3 Outputs and Evaluation

- **Cluster Assignments:** Label vectors indicating cluster membership for each sample.
- **Clustering Quality Metrics:** ARI, NMI (external metrics comparing clusters to true labels), Silhouette Score (internal metric assessing cluster cohesion).
- **Computational Performance Metrics:** Execution time (seconds), measured from algorithm initialization through convergence.
- **Visualization:** PCA-reduced 2D projections of clustering results for qualitative assessment.

#### 3.2.4 Hypotheses and Expected Outcomes

- **H1 (Methodological Efficacy):** All implementations produce statistically equivalent clustering quality metrics, validating algorithmic correctness across frameworks.
- **H2 (Computational Performance):** MLX demonstrates competitive runtime performance relative to PyTorch and scikit-learn, potentially offering advantages for specific dataset sizes or configurations.

### 3.3 Research Design and Method

#### 3.3.1 Algorithm Implementation

**KMeans Clustering Algorithm:**
The study implements the standard KMeans algorithm following Lloyd's algorithm:

```bash
1. Initialization: Randomly select k samples as initial centroids
2. Assignment Step: Assign each sample to nearest centroid (Euclidean distance)
3. Update Step: Recalculate centroids as mean of assigned samples
4. Convergence: Repeat steps 2-3 until centroids stabilize or max_iterations reached
```

**Implementation Parameters:**

- Number of clusters (k): 10 (matching MNIST's 10 digit classes)
- Maximum iterations: 300 (default scikit-learn value)
- Convergence criterion: Centroid positions change < tolerance threshold
- Random state: 44 + round_number (44-63 across 20 rounds)

#### 3.3.2 MLX-Specific Implementation Challenges

The MLX implementation required substantial adaptations due to API differences:

- **Challenge 1: Absence of Boolean Indexing**

- **Problem:** MLX does not support direct boolean array indexing (e.g., `X[labels == k]`)
- **Solution:** Implemented mask-based weighted operations:

- **Challenge 2: Limited Random Number Generation API**

- **Problem:** No direct equivalent to NumPy's `np.random.choice()`
- **Solution:** Used `mx.random.permutation()` followed by slicing for centroid initialization

- **Challenge 3: Type System Differences**

- **Problem:** MLX arrays require careful type checking and conversion
- **Solution:** Implemented explicit input validation and type conversion logic

#### 3.3.3 Data Collection Protocol

**Experimental Protocol:**

1. **Environment Setup:**
   - Hardware: Apple Silicon Mac (M-series processor)
   - Operating System: macOS
   - Python: 3.11 with frameworks installed per requirements.txt
   - Dataset: From UCI ML Repo

2. **Preprocessing:**
   - Feature scaling & engineering: Engineering custom features and scaling them between 0 and 1 which is better for ML.

3. **Execution Protocol:**
   - **Rounds:** 20 independent executions
   - **Random Seeds:** Sequential seeds ensuring different initializations per round
   - **Timing Method:** Python's `time.time()` measuring wall-clock time from fit initialization to convergence

4. **Metric Calculation:**
   - **Per Round:** Calculate ARI, NMI, Silhouette Score for each framework's clustering
   - **Aggregation:** Compute mean and standard deviation across 20 rounds
   - **Statistical Validity:** Multiple rounds enable assessment of initialization sensitivity and performance stability

#### 3.3.4 Evaluation Metrics

**Clustering Quality Metrics:**

1. **Adjusted Rand Index (ARI):**
   - Range: [-1, 1], where 1 indicates perfect agreement
   - Interpretation: Measures similarity between clustering and true labels, corrected for chance
   - Formula: $\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}{\frac{1}{2}[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}] - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}$

2. **Normalized Mutual Information (NMI):**
   - Range: [0, 1], where 1 indicates perfect correspondence
   - Interpretation: Measures mutual information between cluster assignments and true labels, normalized by entropy
   - Formula: $\text{NMI}(U,V) = \frac{2 \cdot I(U;V)}{H(U) + H(V)}$

3. **Silhouette Score:**
   - Range: [-1, 1], where 1 indicates optimal cluster separation
   - Interpretation: Measures cluster cohesion (intra-cluster distance) vs. separation (inter-cluster distance)
   - Formula: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean nearest-cluster distance

**Performance Metrics:**

1. **Execution Time:**
   - Measurement: Wall-clock seconds from algorithm start to convergence
   - Precision: Millisecond resolution via `time.time()`
   - Aggregation: Mean ± standard deviation across 20 rounds

2. **Memory Efficiency:**
   - Qualitative assessment based on framework characteristics
   - MLX unified memory vs. discrete CPU/GPU memory models

### 3.4 Validity, Reliability, and Feasibility

#### 3.4.1 Internal Validity

**Control of Confounding Variables:**

- **Hardware Consistency:** All experiments executed on identical hardware to eliminate platform variability
- **Algorithmic Equivalence:** Implementations follow identical logical steps despite API differences
- **Data Consistency:** Same samples used across all frameworks within each round
- **Parameter Consistency:** Identical k, max_iterations, and convergence criteria across implementations

**Threats Addressed:**

- **Initialization Bias:** Mitigated through 20 rounds with varying random seeds
- **Order Effects:** Consistent execution order per round controls for thermal or caching effects
- **Implementation Errors:** Cross-validation through comparison with established scikit-learn baseline

#### 3.4.2 Reliability

**Reproducibility Measures:**

- **Fixed Random Seeds:** Documented seeds enable exact replication of experimental conditions
- **Open Source Implementation:** All code available for independent verification
- **Detailed Documentation:** Implementation challenges and solutions documented in supplementary materials
- **Statistical Aggregation:** Mean and standard deviation reporting quantifies measurement stability

**Consistency Assessment:**

- Multiple rounds reveal performance variability
- Standard deviation metrics assess initialization sensitivity
- Framework-to-framework comparison validates measurement procedures

#### 3.4.3 Feasibility

**Practical Considerations:**

- **Computational Resources:** Experiments completable on consumer-grade Apple Silicon Mac within reasonable timeframe (minutes per round)
- **Software Availability:** All frameworks available through standard package managers (uv)
- **Data Accessibility:** Data publicly available
- **Technical Expertise:** Implementation feasible with intermediate Python and machine learning knowledge

### 3.5 Data Analysis Plan

#### 3.5.1 Quantitative Analysis

**Descriptive Statistics:**

- Mean and standard deviation for each metric (ARI, NMI, Silhouette, Time) per framework
- Tabular summary presentation for direct comparison

**Comparative Analysis:**

- Framework ranking by performance metrics
- Percentage differences relative to baseline (scikit-learn)
- Coefficient of variation to assess stability

**Visualization:**

- Performance comparison tables
- Scatter plots of PCA-reduced cluster assignments
- Time-series or distribution plots if substantial variability observed

#### 3.5.2 Interpretation Framework

**Performance Assessment Criteria:**

- **Clustering Quality:** Implementations considered equivalent if ARI/NMI/Silhouette differences < 5% (accounting for initialization variability)
- **Computational Efficiency:** Frameworks ranked by mean execution time; practical significance threshold of >20% difference
- **Stability:** Lower standard deviation indicates more consistent performance across initializations

#### 3.5.3 Limitations and Mitigation Strategies

**Acknowledged Limitations:**

1. **Single Algorithm:** Findings specific to KMeans; may not generalize to all clustering methods
   - *Mitigation:* Future work to extend to more methods

2. **Hardware Specificity:** Results tied to specific Apple Silicon generation
   - *Mitigation:* Documentation of exact hardware specifications for context

3. **Thermal Considerations:** Prolonged execution may induce thermal throttling
   - *Mitigation:* Consistent execution order and rest periods between rounds

### 3.6 Summary

This methodology chapter establishes a rigorous quantitative experimental framework for evaluating MLX's performance for KMeans clustering. The approach ensures internal validity through careful control of confounding variables, external validity through use of standard benchmarks, and reliability through statistical aggregation across multiple experimental rounds. The implementation of equivalent algorithms across four frameworks, measured using established clustering quality metrics and computational performance indicators, provides a robust foundation for addressing the research questions and contributing empirical evidence to the identified literature gap.

---

## Preliminary Results (Initial Testing Phase)

### Experimental Configuration

- **Dataset:** MNIST (7,500 samples, 784 features)
- **Clusters (k):** 10
- **Experimental Rounds:** 20
- **Random Seeds:** 44-63

### Mean Performance Metrics Across 20 Rounds

| Metric | scikit-learn | NumPy | MLX | PyTorch |
|--------|--------------|-------|-----|---------|
| **ARI** | 0.3708 | 0.3686 | **0.3777** | 0.3762 |
| **NMI** | 0.4878 | 0.4894 | 0.4937 | **0.4954** |
| **Silhouette** | 0.0591 | 0.0609 | **0.0636** | 0.0634 |
| **Time (s)** | **0.1633** | 2.7779 | 1.1029 | 0.6054 |

### Standard Deviation

| Metric | scikit-learn | NumPy | MLX | PyTorch |
|--------|--------------|-------|-----|---------|
| **ARI** | 0.0290 | 0.0239 | **0.0215** | 0.0234 |
| **NMI** | 0.0178 | 0.0145 | **0.0112** | 0.0131 |
| **Silhouette** | 0.0094 | 0.0087 | **0.0079** | 0.0086 |
| **Time (s)** | 0.0868 | 1.3259 | 0.8260 | 0.3053 |

### Key Observations

1. **Clustering Quality:** MLX achieves competitive or superior clustering quality metrics, with the highest ARI (0.3777) and Silhouette (0.0636) scores among all implementations. The differences in quality metrics across frameworks remain within 5%, suggesting algorithmic equivalence.

2. **Computational Performance:** scikit-learn demonstrates superior runtime performance (0.163s), benefiting from highly optimized C implementations. MLX achieves intermediate performance (1.103s), outperforming pure NumPy (2.778s) by approximately 60% but trailing PyTorch (0.605s) and scikit-learn.

3. **Stability:** MLX exhibits the lowest standard deviations for clustering quality metrics (ARI std: 0.0215, NMI std: 0.0112), indicating more consistent performance across different initializations.

4. **Framework Ranking by Speed:** scikit-learn (fastest) > PyTorch > MLX > NumPy (slowest)

5. **Framework Ranking by Quality:** MLX ≈ PyTorch ≈ scikit-learn > NumPy (minimal practical differences)

### Implications

These preliminary results suggest that MLX successfully implements KMeans clustering with quality metrics comparable to established frameworks while achieving moderate computational efficiency. The performance gap between MLX and scikit-learn likely reflects differences in low-level optimization maturity rather than fundamental architectural limitations.

This is reproducible and can be used for the full Credit Card Dataset.

---

## References

Bahrampour, S., Ramakrishnan, N., Schott, L., & Shah, M. (2016). Comparative study of deep learning software frameworks. *International Conference on Learning Representations (ICLR)*.

Bose, I., & Chen, X. (2009). Hybrid models using unsupervised clustering for prediction of customer churn. *Journal of Organizational Computing and Electronic Commerce, 19*(2), 133-151.

Chaplia, O., & Klym, H. (2024). Evaluating small quantized language models on Apple Silicon. *arXiv preprint*.

Chen, Z. (2023). The investigation of machine learning approaches for customer segmentation. *IEEE International Conference on Machine Learning and Applications*.

Feng, D., Xu, Z., Wang, R., & Lin, F. X. (2024). Profiling Apple Silicon performance for machine learning training. *Proceedings of Machine Learning and Systems (MLSys)*.

Gu, Y., Kadekodi, R., Nguyen, H., et al. (2024). CONSUMERBENCH: Benchmarking generative AI applications on end-user devices. *arXiv preprint*.

Hannun, A., Digani, J., Katharopoulos, A., & Collobert, R. (2023). MLX: Efficient and flexible machine learning on Apple Silicon. *Apple Machine Learning Research*.

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

Rijnen, M. (2020). Predicting churn using hybrid supervised-unsupervised models. *Master's Thesis, Eindhoven University of Technology*.

Snehalatha, N., et al. (2021). Customer segmentation and profiling for e-commerce using DBSCAN and Fuzzy C-Means. *International Journal of Engineering Trends and Technology*.

Spanoudes, P., & Nguyen, T. (2017). Deep learning in customer churn prediction: Unsupervised feature learning on abstract company independent feature vectors. *arXiv preprint*.

Tran, H., Le, N., & Nguyen, V. H. (2021). Customer churn prediction in the banking sector using machine learning-based classification models. *International Journal of Advanced Computer Science and Applications, 12*(6).

Varmedja, D., Karanovic, M., Sladojevic, S., et al. (2019). Credit card fraud detection - machine learning methods. *International Symposium on Intelligent Systems and Informatics (SISY)*.

---

## Appendices

### Appendix A: Technical Implementation Details

**Hardware Specifications:**

- Processor: Apple Silicon (M-series)
- Unified Memory: Available pool size
- Operating System: macOS (version)

**Software Environment:**

- Python: 3.11
- MLX: (version from requirements.txt)
- PyTorch: (version from requirements.txt)
- scikit-learn: (version from requirements.txt)
- NumPy: (version from requirements.txt)

**Complete package dependencies available in:** `requirements.txt`

### Appendix B: Implementation Challenges

Detailed documentation of MLX API differences and implementation adaptations available in: `MLX_API_Challenges.md`

**Key Implementation Files:**

- `src/utils/kmeans_mlx.py` - MLX implementation
- `src/utils/kmeans_numpy.py` - NumPy baseline
- `src/utils/kmeans_torch.py` - PyTorch implementation
- `src/test.py` - Main experimental script

### Appendix C: Conceptual Framework Diagram

Visual representation of the research framework available in: `conceptual_framework.mermaid`

### Appendix D: Data Sources

**Primary Dataset:** MNIST via scikit-learn `fetch_openml("mnist_784")`

**Future Datasets (Proposed):**

- Credit Card Customer Data (Kaggle)
- Credit Card Customers Churn Dataset (Kaggle)
- Default of Credit Card Clients (UC Irvine ML Repository)

Complete dataset documentation available in: `Markdown/Datasets.md`

---
