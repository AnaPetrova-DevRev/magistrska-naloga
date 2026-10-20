"""
Clustering Module

Provides dataset-level and record-level clustering:
- Dataset-level: hierarchical (dendrogram) and K-means on fingerprint vectors
- Record-level: MiniBatchKMeans / DBSCAN on all records across sync units
- Feature importance via DecisionTreeClassifier
- Dimensionality reduction (PCA, t-SNE) for visualization
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from storage_access import StorageAccess
from sync_unit import SyncUnit
from global_vars import (
    PIPELINE_EXPLORER_APP_FOLDER_NAME,
    ANALYSIS_PER_RECORD_FILE_NAME_FORMAT,
)
from .fingerprint import DatasetFingerprint

logger = logging.getLogger(__name__)

# Maximum records for full clustering (subsample above this)
MAX_RECORDS_FULL = 50_000

# Columns to exclude from record clustering features
RECORD_EXCLUDE_COLS = {"id", "first_created_date", "first_modified_date", "last_modified_date"}


@dataclass
class DatasetClusterResult:
    """Result of dataset-level clustering."""

    method: str  # "hierarchical" or "kmeans"
    n_clusters: int
    labels: dict[str, int]  # sync_unit_name -> cluster_id
    silhouette: float
    feature_importances: dict[str, float] = field(default_factory=dict)
    linkage_matrix: Optional[list] = None  # For dendrogram rendering
    pca_2d: Optional[dict[str, list[float]]] = None  # sync_unit_name -> [x, y]
    tsne_2d: Optional[dict[str, list[float]]] = None

    def to_dict(self) -> dict:
        result = {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "labels": self.labels,
            "silhouette": self.silhouette,
            "feature_importances": self.feature_importances,
        }
        if self.linkage_matrix is not None:
            result["linkage_matrix"] = self.linkage_matrix
        if self.pca_2d is not None:
            result["pca_2d"] = self.pca_2d
        if self.tsne_2d is not None:
            result["tsne_2d"] = self.tsne_2d
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetClusterResult":
        return cls(
            method=data["method"],
            n_clusters=data["n_clusters"],
            labels=data["labels"],
            silhouette=data["silhouette"],
            feature_importances=data.get("feature_importances", {}),
            linkage_matrix=data.get("linkage_matrix"),
            pca_2d=data.get("pca_2d"),
            tsne_2d=data.get("tsne_2d"),
        )


@dataclass
class RecordClusterResult:
    """Result of record-level clustering across multiple sync units."""

    n_clusters: int
    method: str  # "minibatch_kmeans" or "dbscan"
    # DataFrame with columns: id, org, system, project, cluster_id
    record_labels: Optional[pd.DataFrame] = None
    cluster_centers: Optional[np.ndarray] = None  # (n_clusters, n_features)
    silhouette: float = 0.0
    pca_2d: Optional[np.ndarray] = None  # (n_records, 2)
    tsne_2d: Optional[np.ndarray] = None  # (n_records, 2)
    feature_names: list[str] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)
    was_subsampled: bool = False
    total_records: int = 0
    sample_size: int = 0


def _compute_feature_importance(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute feature importance using a decision tree on cluster labels."""
    if len(np.unique(labels)) < 2:
        return {}

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(features, labels)

    importances = tree.feature_importances_
    result = {
        name: float(imp)
        for name, imp in zip(feature_names, importances)
        if imp > 0.01
    }
    # Sort by importance descending
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def cluster_datasets(
    fingerprints: list[DatasetFingerprint],
    method: str = "hierarchical",
    n_clusters: int = 3,
) -> DatasetClusterResult:
    """Cluster sync units by their fingerprint vectors.

    Args:
        fingerprints: List of dataset fingerprints.
        method: "hierarchical" or "kmeans".
        n_clusters: Number of clusters to create.

    Returns:
        DatasetClusterResult with labels, silhouette score, and projections.
    """
    if len(fingerprints) < 2:
        logger.warning("Need at least 2 fingerprints to cluster")
        return DatasetClusterResult(
            method=method,
            n_clusters=0,
            labels={fp.sync_unit_name: 0 for fp in fingerprints},
            silhouette=0.0,
        )

    feature_names = fingerprints[0].get_feature_names()
    vectors = np.array([fp.to_feature_vector(feature_names) for fp in fingerprints])
    names = [fp.sync_unit_name for fp in fingerprints]

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    # Clamp n_clusters
    n_clusters = min(n_clusters, len(fingerprints) - 1)
    n_clusters = max(n_clusters, 2)

    linkage_mat = None

    if method == "hierarchical":
        linkage_mat = linkage(scaled, method="ward")
        cluster_labels = fcluster(linkage_mat, t=n_clusters, criterion="maxclust")
        cluster_labels = cluster_labels - 1  # 0-indexed
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(scaled)

    # Silhouette (only if n_clusters >= 2 and < n_samples)
    sil = 0.0
    if 2 <= n_clusters < len(fingerprints):
        sil = float(silhouette_score(scaled, cluster_labels))

    # PCA 2D projection
    n_components = min(2, scaled.shape[1], scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(scaled)
    pca_2d = {name: pca_coords[i].tolist() for i, name in enumerate(names)}

    # t-SNE 2D (only if enough data points)
    tsne_2d = None
    if len(fingerprints) >= 4:
        perplexity = min(3, len(fingerprints) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_coords = tsne.fit_transform(scaled)
        tsne_2d = {name: tsne_coords[i].tolist() for i, name in enumerate(names)}

    # Feature importance
    importances = _compute_feature_importance(scaled, cluster_labels, feature_names)

    labels_dict = {name: int(cluster_labels[i]) for i, name in enumerate(names)}

    return DatasetClusterResult(
        method=method,
        n_clusters=n_clusters,
        labels=labels_dict,
        silhouette=sil,
        feature_importances=importances,
        linkage_matrix=linkage_mat.tolist() if linkage_mat is not None else None,
        pca_2d=pca_2d,
        tsne_2d=tsne_2d,
    )


def _load_all_records(
    sync_units: list[SyncUnit],
    max_total: int = MAX_RECORDS_FULL,
) -> tuple[pd.DataFrame, list[str], bool]:
    """Load analysis_per_record from all sync units, adding metadata columns.

    Returns:
        (combined_df, feature_columns, was_subsampled)
    """
    sa = StorageAccess()
    all_dfs = []
    counts = {}

    for su in sync_units:
        path = (
            f"{su.get_path_for_sync_unit()}"
            f"/{PIPELINE_EXPLORER_APP_FOLDER_NAME}"
            f"/{ANALYSIS_PER_RECORD_FILE_NAME_FORMAT}"
        )
        if not sa.check_if_file_exists(path):
            logger.warning("analysis_per_record not found for %s", su.get_name())
            continue

        df = sa.read_picklefile_as_dataframe(path)
        df["_org"] = su.organization_id
        df["_system"] = su.external_system_id
        df["_project"] = su.project_id
        df["_sync_unit"] = su.get_name()
        all_dfs.append(df)
        counts[su.get_name()] = len(df)

    if not all_dfs:
        return pd.DataFrame(), [], False

    combined = pd.concat(all_dfs, ignore_index=True)
    total = len(combined)
    logger.info("Loaded %d total records from %d sync units", total, len(all_dfs))

    # Identify numeric feature columns (exclude metadata and dates)
    feature_cols = [
        c for c in combined.columns
        if c not in RECORD_EXCLUDE_COLS
        and not c.startswith("_")
        and combined[c].dtype in [np.float64, np.int64, np.float32, np.int32, "bool"]
    ]

    # Subsample if needed (stratified by sync unit)
    was_subsampled = False
    if total > max_total:
        logger.info("Subsampling from %d to %d records (stratified)", total, max_total)
        fraction = max_total / total
        sampled_dfs = []
        for su_name, group in combined.groupby("_sync_unit"):
            n_sample = max(1, int(len(group) * fraction))
            sampled_dfs.append(group.sample(n=n_sample, random_state=42))
        combined = pd.concat(sampled_dfs, ignore_index=True)
        was_subsampled = True
        logger.info("Subsampled to %d records", len(combined))

    return combined, feature_cols, was_subsampled


def cluster_records(
    sync_units: list[SyncUnit],
    n_clusters: int = 8,
    method: str = "minibatch_kmeans",
    max_total: int = MAX_RECORDS_FULL,
) -> RecordClusterResult:
    """Cluster individual records across all sync units.

    Args:
        sync_units: List of sync units to include.
        n_clusters: Number of clusters for K-means.
        method: "minibatch_kmeans" or "dbscan".
        max_total: Maximum records before subsampling.

    Returns:
        RecordClusterResult with labels, projections, and feature importances.
    """
    combined, feature_cols, was_subsampled = _load_all_records(sync_units, max_total)

    if len(combined) == 0 or len(feature_cols) == 0:
        logger.warning("No records or features available for clustering")
        return RecordClusterResult(n_clusters=0, method=method)

    logger.info(
        "Clustering %d records with %d features using %s",
        len(combined), len(feature_cols), method,
    )

    # Prepare feature matrix
    X = combined[feature_cols].copy()
    # Impute NaN with column median
    for col in feature_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    # Handle any remaining NaN (e.g., all-NaN columns)
    X = X.fillna(0)

    feature_matrix = X.values.astype(float)

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    # PCA for dimensionality reduction before clustering
    n_pca_components = min(50, scaled.shape[1], scaled.shape[0] - 1)
    pca_reducer = PCA(n_components=n_pca_components, random_state=42)
    pca_features = pca_reducer.fit_transform(scaled)

    # Cluster
    if method == "dbscan":
        clusterer = DBSCAN(eps=3.0, min_samples=5)
        cluster_labels = clusterer.fit_predict(pca_features)
        actual_n_clusters = len(set(cluster_labels) - {-1})
        centers = None
    else:
        n_clusters = min(n_clusters, len(combined) - 1)
        n_clusters = max(n_clusters, 2)
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(1024, len(combined)),
        )
        cluster_labels = clusterer.fit_predict(pca_features)
        actual_n_clusters = n_clusters
        centers = clusterer.cluster_centers_

    # Silhouette (sample if large)
    sil = 0.0
    n_unique = len(set(cluster_labels) - {-1})
    if n_unique >= 2:
        if len(combined) > 10_000:
            # Subsample for silhouette computation
            rng = np.random.default_rng(42)
            idx = rng.choice(len(combined), size=10_000, replace=False)
            sil = float(silhouette_score(pca_features[idx], cluster_labels[idx]))
        else:
            sil = float(silhouette_score(pca_features, cluster_labels))

    # 2D projections for visualization
    pca_2d_reducer = PCA(n_components=2, random_state=42)
    pca_2d = pca_2d_reducer.fit_transform(pca_features)

    # t-SNE (on PCA-reduced features for speed)
    tsne_2d = None
    if len(combined) <= 50_000:
        perplexity = min(30, len(combined) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_2d = tsne.fit_transform(pca_features)

    # Feature importance
    importances = _compute_feature_importance(scaled, cluster_labels, feature_cols)

    # Build result labels DataFrame
    labels_df = combined[["_org", "_system", "_project", "_sync_unit"]].copy()
    if "id" in combined.columns:
        labels_df["id"] = combined["id"]
    labels_df["cluster_id"] = cluster_labels
    labels_df = labels_df.rename(columns={
        "_org": "org",
        "_system": "system",
        "_project": "project",
        "_sync_unit": "sync_unit",
    })

    return RecordClusterResult(
        n_clusters=actual_n_clusters,
        method=method,
        record_labels=labels_df,
        cluster_centers=centers,
        silhouette=sil,
        pca_2d=pca_2d,
        tsne_2d=tsne_2d,
        feature_names=feature_cols,
        feature_importances=importances,
        was_subsampled=was_subsampled,
        total_records=len(combined),
        sample_size=len(combined),
    )
