# Sparse Oblique Decision Trees via L1-Regularized Proximal Gradient Descent

A from-scratch implementation of **Oblique Decision Trees (ODTs)** that replace axis-aligned splits with L1-regularized linear hyperplanes, learned via Proximal Gradient Descent (ISTA). Benchmarked against standard CART across 23 datasets from the OpenML CC18 suite.

> Academic project for *Introduction to Machine Learning and Data Mining* — FEUP, University of Porto (Dec 2025)


---

## The Problem

Standard decision trees (CART) split along single feature axes (`x₁ < 5`). When the true decision boundary is diagonal or involves feature correlations, CART is forced to approximate it with a **staircase** of many orthogonal cuts — leading to unnecessarily deep, high-variance trees.

## Our Approach

We replace axis-aligned splits with **multivariate hyperplanes** (`w₁x₁ + w₂x₂ + ... + b < 0`) at each node. The key components are:

- **Splitting via Logistic Regression** — At each node, we frame the split as a binary classification problem and learn an optimal separating hyperplane.
- **L1 Regularization (Lasso)** — Enforces sparsity in the weight vectors, performing embedded feature selection and preserving interpretability.
- **Proximal Gradient Descent (ISTA)** — Solves the non-smooth L1-penalized objective using the Soft-Thresholding operator.
- **Ideal Outcomes Heuristic** — For multiclass problems, we enumerate class bipartitions to find the best binary grouping before optimizing the hyperplane.

## Key Results

Benchmarked on 23 datasets from the [OpenML CC18](https://www.openml.org/s/99) suite with `max_depth=3` for both methods:

| Metric | Value |
|--------|-------|
| **Datasets where ODT > CART** | 16 / 20 |
| **Avg. tree size (nodes)** | 9.0 (ODT) vs 14.7 (CART) |
| **Best accuracy gain** | +17.3% on `balance-scale` |
| **Avg. depth reduction** | ~35% fewer nodes |

The oblique approach excels on datasets with **linear or rotational feature dependencies** (e.g., `balance-scale`, `spambase`, `analcatdata_authorship`) while CART remains competitive on problems with inherently axis-aligned boundaries (e.g., `wall-robot-navigation`).

## Repository Structure

```
├── tree.ipynb              # Full implementation + benchmark + visualizations
├── docs/
│   └── report.pdf          # Theoretical report (proofs, derivations, complexity analysis)
├── results/
│   ├── benchmark_results.csv
│   └── figures/            # Pre-rendered plots
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn openml
```

### Run

Open `tree.ipynb` and run all cells. The notebook is self-contained and organized in 5 parts:

1. **Implementation** — ISTA optimizer, hybrid splitting strategy, tree classes
2. **Benchmark Suite** — Automated evaluation on OpenML CC18 datasets
3. **Performance Visualization** — Accuracy comparison, precision/recall diagnostics, training cost
4. **Deep Dive** — Synthetic data analysis, decision boundary visualization, tree topology comparison
5. **Advanced Diagnostics** — Feature importance (informative vs noise), sparsity analysis, ROC curves

## Theoretical Highlights

The accompanying [report](docs/report.pdf) covers:

- **NP-completeness** of optimal tree induction and the necessity of greedy heuristics
- **Analytical link** between Gini impurity and Shannon entropy via Taylor expansion
- **Convex optimization framework** for oblique splits using logistic surrogates
- **Derivation of the Proximal Operator** and the Soft-Thresholding closed-form solution
- **Complexity analysis**: CART's `O(dN log²N)` vs ODT's `O(2^R · k · d · N · D)`

## Technologies

`Python` · `NumPy` · `SciPy` · `scikit-learn` · `OpenML` · `Matplotlib` · `Seaborn`

## Authors

- **Paolo Pascarelli** — [GitHub](https://github.com/paolopasca)
- **Pedro Martins**

## References

1. Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning* (2009)
2. Breiman et al. — *Classification and Regression Trees* (1984)
3. Murthy, Kasif & Salzberg — *A System for Induction of Oblique Decision Trees* (1994)
4. Hyafil & Rivest — *Constructing Optimal Binary Decision Trees is NP-Complete* (1976)

## License

This project was developed for academic purposes at FEUP, University of Porto.
