# **Project Proposal**

## **Title**

Evaluating Apple’s MLX Framework for Unsupervised Structure Discovery in Credit Card Customer Data

---

## **1. Research Problem**

Most customer churn studies rely on **supervised learning**, which depends on labeled data. In real-world business contexts, such labels are often **incomplete, noisy, or unavailable**.
This project explores whether **unsupervised learning algorithms**, implemented using Apple’s new **MLX framework**, can uncover meaningful customer segments that align with churn behavior. It also benchmarks MLX’s **computational performance and usability** against established frameworks such as **PyTorch** and **scikit-learn**.

---

## **2. Objectives**

1. Implement clustering algorithms (e.g., **K-Means**, **DBSCAN**, or **PCA + K-Means**) using **MLX**.  
2. Train models on the **Credit Card Customers dataset** without using churn labels.  
3. Evaluate clustering quality against known labels using metrics such as **Adjusted Rand Index (ARI)**, **Normalized Mutual Information (NMI)**, and **Silhouette Score**.  
4. Benchmark **runtime performance and memory efficiency** against **PyTorch** and **scikit-learn** implementations.  
5. Interpret discovered clusters to assess their **business relevance** and potential for **customer retention strategies**.

---

## **3. Methodology Overview**

1. **Data Preparation:**  
   Load and preprocess the *Credit Card Customers* dataset (handling missing values, feature scaling, and dimensionality reduction).  

2. **Model Implementation:**  
   - Develop unsupervised algorithms using **Apple’s MLX framework**.  
   - Create equivalent implementations in **PyTorch** or **scikit-learn** for comparison.  

3. **Evaluation:**  
   - Use internal (Silhouette Score) and external (ARI, NMI) metrics to assess cluster quality.  
   - Compare runtime and memory efficiency across frameworks.  

4. **Analysis:**  
   - Map cluster characteristics to churn-related business insights.  
   - Summarize MLX’s performance and usability findings.

---

## **4. Tools & Technologies**

- **Frameworks:** Apple MLX, PyTorch, scikit-learn  
- **Programming Languages:** Python (for MLX and baselines)  
- **Hardware:** Apple Silicon Mac (M1/M2/M3)  
- **Visualization:** Matplotlib, Seaborn, Plotly (3D & Interactive)
- **Dataset:** *Credit Card Customers* dataset

---

## **5. Expected Outcomes**

- Insight into whether **unsupervised learning** can effectively capture churn-related structure.  
- A **performance and usability benchmark** of MLX versus mainstream ML tools.  
- Practical recommendations for **applying MLX in business data analytics**.  
- A reproducible workflow demonstrating **MLX’s capabilities on Apple Silicon**.
