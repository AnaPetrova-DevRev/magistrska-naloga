"""
Dataset Comparison App - implements the App interface for cross-dataset analysis.

This app provides actions to compute fingerprints, compare datasets pairwise,
and cluster both datasets and records across sync units. Unlike other apps that
operate on a single sync unit, this app is inherently cross-dataset: it reads
artifacts from multiple sync units and stores results in a shared location.
"""
import json
import logging

from app_base import App
from orchestrator_message import OrchestratorResponse
from sync_unit import SyncUnit, get_sync_units
from storage_access import StorageAccess
from app_actions_constants import (
    COMPUTE_FINGERPRINTS,
    COMPUTE_PAIRWISE_COMPARISON,
    CLUSTER_DATASETS,
    CLUSTER_RECORDS,
)
from global_vars import (
    DATASET_COMPARISON_FOLDER_NAME,
    FINGERPRINTS_FILE_NAME,
    PAIRWISE_DISTANCES_FILE_NAME,
    DATASET_CLUSTERS_FILE_NAME,
    RECORD_CLUSTERS_FILE_NAME,
)

from .fingerprint import compute_all_fingerprints, DatasetFingerprint
from .comparison import compute_all_pairwise, load_delta_samples
from .clustering import cluster_datasets, cluster_records

logger = logging.getLogger(__name__)


def _ensure_folder(sa: StorageAccess) -> str:
    """Ensure the dataset_comparison folder exists and return its path."""
    folder = DATASET_COMPARISON_FOLDER_NAME
    if not sa.check_if_file_exists(folder):
        sa.create_folder(folder)
    return folder


def _resolve_sync_units(parameters: dict) -> list[SyncUnit]:
    """Resolve which sync units to operate on from parameters."""
    su_param = parameters.get("sync_units", "all")
    all_units = get_sync_units()

    if su_param == "all" or su_param is None:
        return all_units

    if isinstance(su_param, list):
        names = set(su_param)
        return [su for su in all_units if su.get_name() in names]

    return all_units


class DatasetComparisonApp(App):
    """
    Cross-dataset comparison and clustering app.

    Available actions:
    - compute_fingerprints: Compute summary feature vectors for all sync units
    - compute_pairwise_comparison: Compute distance/similarity between all pairs
    - cluster_datasets: Cluster sync units by fingerprint similarity
    - cluster_records: Cluster individual records across sync units
    """

    def get_name(self) -> str:
        return "dataset_comparison"

    def get_available_actions(self) -> list[str]:
        return [
            COMPUTE_FINGERPRINTS,
            COMPUTE_PAIRWISE_COMPARISON,
            CLUSTER_DATASETS,
            CLUSTER_RECORDS,
        ]

    def execute_action(
        self,
        sync_unit_obj: SyncUnit,
        action: str,
        parameters: dict = None,
    ) -> OrchestratorResponse:
        if not self.supports_action(action):
            raise ValueError(
                f"Action '{action}' not supported by {self.get_name()}"
            )

        if parameters is None:
            parameters = {}

        logger.info("Executing dataset_comparison action '%s'", action)

        if action == COMPUTE_FINGERPRINTS:
            return self._compute_fingerprints(parameters)
        if action == COMPUTE_PAIRWISE_COMPARISON:
            return self._compute_pairwise(parameters)
        if action == CLUSTER_DATASETS:
            return self._cluster_datasets(parameters)
        if action == CLUSTER_RECORDS:
            return self._cluster_records(parameters)

        return OrchestratorResponse(
            success=False,
            message=f"Unknown action: {action}",
        )

    def _compute_fingerprints(self, parameters: dict) -> OrchestratorResponse:
        logs = []
        try:
            sync_units = _resolve_sync_units(parameters)
            logs.append(f"Computing fingerprints for {len(sync_units)} sync units")

            fingerprints = compute_all_fingerprints(sync_units)
            logs.append(f"Successfully computed {len(fingerprints)} fingerprints")

            # Save to storage
            sa = StorageAccess()
            folder = _ensure_folder(sa)
            data = [fp.to_dict() for fp in fingerprints]
            sa.write_json_to_storage(
                f"{folder}/{FINGERPRINTS_FILE_NAME}", data
            )
            logs.append(f"Saved fingerprints to {folder}/{FINGERPRINTS_FILE_NAME}")

            skipped = len(sync_units) - len(fingerprints)
            if skipped > 0:
                logs.append(f"Skipped {skipped} sync units (missing artifacts)")

            return OrchestratorResponse(
                success=True,
                message=f"Computed {len(fingerprints)} fingerprints",
                logs=logs,
                is_done=True,
            )
        except Exception as e:
            error_msg = f"Error computing fingerprints: {e}"
            logger.exception(error_msg)
            logs.append(error_msg)
            return OrchestratorResponse(
                success=False, message=error_msg, logs=logs
            )

    def _compute_pairwise(self, parameters: dict) -> OrchestratorResponse:
        logs = []
        try:
            # Load fingerprints
            sa = StorageAccess()
            folder = _ensure_folder(sa)
            fp_path = f"{folder}/{FINGERPRINTS_FILE_NAME}"

            fp_data = sa.read_json_from_storage(fp_path)
            if not fp_data:
                return OrchestratorResponse(
                    success=False,
                    message="No fingerprints found. Run compute_fingerprints first.",
                    logs=["Fingerprints file not found or empty"],
                )

            fingerprints = [DatasetFingerprint.from_dict(d) for d in fp_data]
            logs.append(f"Loaded {len(fingerprints)} fingerprints")

            # Optionally load delta samples for KS-test
            delta_samples = None
            if parameters.get("include_ks_test", True):
                sync_units = _resolve_sync_units(parameters)
                delta_samples = load_delta_samples(sync_units)
                logs.append(f"Loaded delta samples for {len(delta_samples)} sync units")

            comparisons = compute_all_pairwise(fingerprints, delta_samples)
            logs.append(f"Computed {len(comparisons)} pairwise comparisons")

            # Save
            data = [c.to_dict() for c in comparisons]
            sa.write_json_to_storage(
                f"{folder}/{PAIRWISE_DISTANCES_FILE_NAME}", data
            )
            logs.append(f"Saved to {folder}/{PAIRWISE_DISTANCES_FILE_NAME}")

            return OrchestratorResponse(
                success=True,
                message=f"Computed {len(comparisons)} comparisons",
                logs=logs,
                is_done=True,
            )
        except Exception as e:
            error_msg = f"Error computing pairwise comparisons: {e}"
            logger.exception(error_msg)
            logs.append(error_msg)
            return OrchestratorResponse(
                success=False, message=error_msg, logs=logs
            )

    def _cluster_datasets(self, parameters: dict) -> OrchestratorResponse:
        logs = []
        try:
            sa = StorageAccess()
            folder = _ensure_folder(sa)
            fp_path = f"{folder}/{FINGERPRINTS_FILE_NAME}"

            fp_data = sa.read_json_from_storage(fp_path)
            if not fp_data:
                return OrchestratorResponse(
                    success=False,
                    message="No fingerprints found. Run compute_fingerprints first.",
                    logs=["Fingerprints file not found or empty"],
                )

            fingerprints = [DatasetFingerprint.from_dict(d) for d in fp_data]
            method = parameters.get("method", "hierarchical")
            n_clusters = parameters.get("n_clusters", 3)
            logs.append(
                f"Clustering {len(fingerprints)} datasets "
                f"(method={method}, k={n_clusters})"
            )

            result = cluster_datasets(fingerprints, method=method, n_clusters=n_clusters)
            logs.append(f"Silhouette score: {result.silhouette:.3f}")
            logs.append(f"Cluster assignments: {result.labels}")

            # Save
            sa.write_json_to_storage(
                f"{folder}/{DATASET_CLUSTERS_FILE_NAME}",
                result.to_dict(),
            )
            logs.append(f"Saved to {folder}/{DATASET_CLUSTERS_FILE_NAME}")

            return OrchestratorResponse(
                success=True,
                message=f"Clustered into {result.n_clusters} groups (silhouette={result.silhouette:.3f})",
                logs=logs,
                is_done=True,
            )
        except Exception as e:
            error_msg = f"Error clustering datasets: {e}"
            logger.exception(error_msg)
            logs.append(error_msg)
            return OrchestratorResponse(
                success=False, message=error_msg, logs=logs
            )

    def _cluster_records(self, parameters: dict) -> OrchestratorResponse:
        logs = []
        try:
            sync_units = _resolve_sync_units(parameters)
            n_clusters = parameters.get("n_clusters", 8)
            method = parameters.get("method", "minibatch_kmeans")
            max_total = parameters.get("max_records", 50_000)

            logs.append(
                f"Clustering records from {len(sync_units)} sync units "
                f"(method={method}, k={n_clusters}, max={max_total})"
            )

            result = cluster_records(
                sync_units,
                n_clusters=n_clusters,
                method=method,
                max_total=max_total,
            )

            if result.record_labels is None or len(result.record_labels) == 0:
                return OrchestratorResponse(
                    success=False,
                    message="No records available for clustering",
                    logs=logs,
                )

            logs.append(f"Clustered {result.sample_size} records into {result.n_clusters} groups")
            logs.append(f"Silhouette score: {result.silhouette:.3f}")
            if result.was_subsampled:
                logs.append(f"Subsampled from {result.total_records} total records")

            # Save labels and projections as pickle (large data)
            sa = StorageAccess()
            folder = _ensure_folder(sa)
            sa.save_dataframe_as_picklefile(
                result.record_labels,
                f"{folder}/{RECORD_CLUSTERS_FILE_NAME}",
            )
            logs.append(f"Saved record clusters to {folder}/{RECORD_CLUSTERS_FILE_NAME}")

            return OrchestratorResponse(
                success=True,
                message=(
                    f"Clustered {result.sample_size} records into "
                    f"{result.n_clusters} groups (silhouette={result.silhouette:.3f})"
                ),
                logs=logs,
                is_done=True,
                data={
                    "n_clusters": result.n_clusters,
                    "silhouette": result.silhouette,
                    "total_records": result.total_records,
                    "sample_size": result.sample_size,
                    "was_subsampled": result.was_subsampled,
                    "feature_importances": result.feature_importances,
                },
            )
        except Exception as e:
            error_msg = f"Error clustering records: {e}"
            logger.exception(error_msg)
            logs.append(error_msg)
            return OrchestratorResponse(
                success=False, message=error_msg, logs=logs
            )
