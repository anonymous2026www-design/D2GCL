# Verification

Classification was evaluated using the same three training seeds as the paper for each supported dataset. One fixed configuration is shared across seeds. Values below are mean and population standard deviation in percent. The current-draft D2-GCL implementation differs from the historical implementation; the comparison records those differences.

| Method   | Dataset   | Paper accuracy (%)   | Anonymous accuracy (%)   |   Change (pp) |
|:---------|:----------|:---------------------|:-------------------------|--------------:|
| D2-GCL   | Cora      | 81.97 +/- 0.05       | 81.20 +/- 1.19           |         -0.77 |
| D2-GCL   | CiteSeer  | 69.27 +/- 0.90       | 68.73 +/- 0.29           |         -0.53 |
| D2-GCL   | PubMed    | 81.80 +/- 0.22       | 81.87 +/- 0.37           |          0.07 |
| D2-GCL   | Cornell   | 72.07 +/- 1.27       | 57.66 +/- 2.55           |        -14.41 |
| D2-GCL   | Texas     | 72.07 +/- 4.59       | 74.77 +/- 1.27           |          2.70 |
| GRACE    | Cora      | 75.20 +/- 1.99       | 75.17 +/- 2.01           |         -0.03 |
| GRACE    | CiteSeer  | 66.47 +/- 3.14       | 66.30 +/- 3.18           |         -0.17 |
| GRACE    | PubMed    | 78.27 +/- 1.30       | 78.27 +/- 1.30           |          0.00 |
| GRACE    | Cornell   | 36.04 +/- 2.55       | 36.04 +/- 2.55           |          0.00 |
| GRACE    | Texas     | 48.65 +/- 3.82       | 47.75 +/- 3.37           |         -0.90 |

The original three datasets retain their dense computation path and configurations; their previously verified results are retained. PubMed and Texas were run after the extension. Features, edges, labels and evaluation masks match their paper inputs.

Additional checks passed:

- Dense and blocked InfoNCE losses and gradients agree on CPU and CUDA.
- Sparse positive supports and confidence weights match dense top-fraction mining.
- The sparse Phase-2 weighted inside-log loss retains every denominator candidate and matches dense gradients.
- Complete two-phase dense/blocked training agrees within floating-point tolerances.
- Weighted CAPP supports dense and sparse positive weights with matching outputs.
- Sparse and dense LapPE agree on a small-graph comparison.
- The package is checked after independent archive extraction.
- All source files, comments and documentation are English-only.

Clustering uses the unchanged ten-initialization evaluation code, verified in the initial release. The new three-seed comparisons above evaluate classification only.

Tested environment: Python 3.10, PyTorch 2.9.0+cu128, PyG 2.6.1, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, threadpoolctl 3.6.0.

The fixed PubMed preset is an exploratory, test-informed matching configuration.
It is not the configuration maximizing validation accuracy. Its shared
recipe was independently rerun on all three original seeds, reproducing
81.87 +/- 0.37% test accuracy (population standard deviation).
No encoder, objective, mining or propagation implementation is changed.
