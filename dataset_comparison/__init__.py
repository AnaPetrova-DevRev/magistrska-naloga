"""
Dataset Comparison Module

Provides functionality for cross-dataset comparison and clustering:
- Dataset fingerprinting: compute summary feature vectors per sync unit
- Pairwise comparison: distance metrics, statistical tests, divergence measures
- Clustering: dataset-level and record-level clustering with interpretation
- Visualization: charts for fingerprints, comparisons, and clustering results
"""
from .dataset_comparison_app import DatasetComparisonApp
from .fingerprint import DatasetFingerprint, compute_fingerprint, compute_all_fingerprints
from .comparison import PairwiseComparison, compute_all_pairwise, compute_distance_matrix
from .clustering import (
    DatasetClusterResult,
    RecordClusterResult,
    cluster_datasets,
    cluster_records,
)

__all__ = [
    'DatasetComparisonApp',
    'DatasetFingerprint',
    'compute_fingerprint',
    'compute_all_fingerprints',
    'PairwiseComparison',
    'compute_all_pairwise',
    'compute_distance_matrix',
    'DatasetClusterResult',
    'RecordClusterResult',
    'cluster_datasets',
    'cluster_records',
]
