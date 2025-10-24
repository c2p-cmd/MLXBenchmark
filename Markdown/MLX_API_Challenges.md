# MLX API Challenges and Differences

This document outlines the key challenges and API differences encountered while implementing K-means clustering in MLX compared to NumPy.

## Major Challenges

### 1. Boolean Indexing Not Supported

**Challenge:** MLX does not support boolean indexing like NumPy.

**NumPy approach:**

```python
# This works in NumPy
cluster_points = X[labels == k]
cluster_mean = cluster_points.mean(axis=0)
```

**MLX approach:**

```python
# MLX requires mask-based weighted operations
mask = mx.array(labels == k, dtype=mx.float32)
cluster_size = mx.sum(mask)
weighted_sum = mx.sum(X * mask[:, mx.newaxis], axis=0)
cluster_mean = weighted_sum / cluster_size
```

**Impact:** This required a complete refactor of the centroid update step, using mask-based weighted sums instead of direct filtering.

---

### 2. Type Checking Differences

**Challenge:** MLX arrays cannot be type-checked using standard Python type checking methods.

**Incorrect approach:**

```python
# These don't work with MLX
if type(X) is mx.array:  # mx.array is not a class type
    pass

if type(X) is NDArray:  # NDArray is a type alias, not a class
    pass
```

**Correct approach:**

```python
# Use isinstance with proper types
if isinstance(X, np.ndarray):
    X = mx.array(X)
elif not isinstance(X, mx.array):
    X = mx.array(X)
```

**Impact:** Required careful validation of input types and proper use of `isinstance()` for type checking.

---

### 3. Random Number Generation API

**Challenge:** MLX uses a different API for random operations compared to NumPy.

**NumPy approach:**

```python
np.random.seed(random_state)
idx = np.random.choice(len(X), n_clusters, replace=False)
```

**MLX approach:**

```python
mx.random.seed(random_state)
idx = mx.random.permutation(len(X))[:n_clusters]
```

**Impact:** No direct equivalent to `np.random.choice()` with `replace=False`. Used `mx.random.permutation()` followed by slicing instead.

---

### 4. Array Indexing Functions

**Challenge:** MLX lacks functions like `argwhere()` or NumPy-style `where()`.

**NumPy approach:**

```python
# NumPy has argwhere for finding indices
indices = np.argwhere(mask).flatten()
points = X[indices]
```

**MLX limitation:**

```python
# mx.argwhere() does not exist
# mx.where() has different signature (condition, true_value, false_value)
```

**Workaround:** Use mask-based operations instead of index-based filtering.

**Impact:** Forced to adopt a more functional programming style with mask broadcasting rather than imperative filtering.

---

### 5. Type Conversion Methods

**Challenge:** Type conversion syntax differs between NumPy and MLX.

**NumPy approach:**

```python
mask = (labels == k).astype(np.float32)
```

**MLX approach:**

```python
# astype() may not work on comparison results
mask = mx.array(labels == k, dtype=mx.float32)
```

**Impact:** Need to be explicit about dtype during array creation rather than converting afterward.

---

## API Similarities (Positive Points)

### 1. `mx.newaxis` Works Like NumPy

```python
# Both work identically
X[:, mx.newaxis, :]  # MLX
X[:, np.newaxis, :]  # NumPy
```

### 2. Most Mathematical Operations Are Identical

```python
# These work the same in both frameworks
mx.sum(), mx.mean(), mx.argmin(), mx.allclose()
np.sum(), np.mean(), np.argmin(), np.allclose()
```

### 3. Array Broadcasting

MLX supports NumPy-style broadcasting:

```python
X * mask[:, mx.newaxis]  # Broadcasting works as expected
```

### 4. Linear Algebra Operations

```python
mx.linalg.norm()  # Same as np.linalg.norm()
```

---

## Summary

The main challenge in porting NumPy code to MLX is the **lack of boolean indexing** and some helper functions like `argwhere()`. This requires rethinking algorithms to use mask-based operations and broadcasting instead of direct filtering. However, MLX maintains good API compatibility for most mathematical operations, making the transition manageable once these key differences are understood.

### Key Takeaways

- ✅ Use mask-based operations instead of boolean indexing
- ✅ Use `isinstance()` for type checking, not `type() is`
- ✅ Use `mx.random.permutation()` instead of `np.random.choice()`
- ✅ Leverage broadcasting and vectorized operations
- ✅ Most mathematical functions have direct equivalents
