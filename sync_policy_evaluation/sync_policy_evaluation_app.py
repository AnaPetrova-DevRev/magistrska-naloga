"""Sync policy evaluation app for the orchestrator system.

Evaluates trained RL policies and heuristic baselines on sync unit data,
producing artifacts for the sync policy dashboard.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app_base import App
from orchestrator_message import OrchestratorResponse
import storage_access
import sync_unit as sync_unit_module
from app_actions_constants import EVALUATE_SYNC_POLICY, COMPARE_POLICIES
from modeling.core_types import DatasetConfig, TaskType
from modeling.dataset_builder import (
    build_event_rate_dataset,
    build_time_to_event_dataset,
)
from rl_training.sync_env import SyncSchedulingEnv
from rl_training.prediction_provider import (
    ActionAwarePredictionProvider,
    MaskedPredictionProvider,
    NoPredictionProvider,
    PrecomputedPredictionProvider,
)
from rl_training.reward import RewardConfig
from sync_policy_evaluation.policy_loader import (
    ModelPredictionPolicy,
    load_policy,
    resolve_trained_policy_file,
)
from sync_policy_evaluation.evaluation_engine import (
    run_evaluation,
    run_evaluation_summary_only,
    rescore_decisions,
    compute_summary_metrics,
)
from sync_policy_evaluation.artifact_writer import write_evaluation_artifacts

logger = logging.getLogger(__name__)

RL_POLICIES_FOLDER = "rl_policies"


class SyncPolicyEvaluationApp(App):
    """App for evaluating sync scheduling policies.

    Actions:
        evaluate_sync_policy: Evaluate a single policy (trained or heuristic).
        compare_policies: Compare multiple evaluation artifacts.
    """

    def __init__(self, sa: Optional[storage_access.StorageAccess] = None):
        self._sa = sa or storage_access.StorageAccess()

    def get_name(self) -> str:
        return "sync_policy_evaluation"

    def get_available_actions(self) -> list[str]:
        return [EVALUATE_SYNC_POLICY, COMPARE_POLICIES]

    def execute_action(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        if not self.supports_action(action):
            raise ValueError(
                f"Action '{action}' not supported by {self.get_name()}"
            )

        parameters = parameters or {}

        if action == EVALUATE_SYNC_POLICY:
            return self._handle_evaluate(sync_unit_obj, parameters)
        elif action == COMPARE_POLICIES:
            return self._handle_compare(sync_unit_obj, parameters)

        raise ValueError(f"Unknown action: {action}")

    # evaluate_sync_policy

    def _handle_evaluate(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Evaluate a single policy on sync unit data."""
        task_type_str = parameters.get("task_type", "event_rate")
        dataset_params = parameters.get("dataset", {})
        env_params = parameters.get("env", {})
        caller_reward_params = parameters.get("reward", {})

        # Always evaluate with the standardized reward so that
        # total_reward is comparable across policies.
        standard_rc = RewardConfig.standard_eval(mode=task_type_str)
        reward_params = standard_rc.to_dict()
        prediction_params = parameters.get("predictions", {})
        prediction_task_type_str = prediction_params.get(
            "prediction_task_type", task_type_str
        )
        policy_name = parameters.get("policy_name", "eval")

        # Policy loading
        policy_type = parameters.get("policy_type", "trained")
        policy_path = parameters.get("policy_path")
        policy_checkpoint = parameters.get("policy_checkpoint", "final")
        heuristic_config = parameters.get("heuristic")

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        try:
            policy = load_policy(
                policy_type=policy_type,
                policy_path=policy_path,
                heuristic_config=heuristic_config,
                name=policy_name,
                checkpoint=policy_checkpoint,
            )
        except Exception as e:
            return OrchestratorResponse(
                success=False,
                message=f"Failed to load policy: {e}",
                is_done=True,
            )

        # Load training metadata for trained policies (cron patterns + split info)
        cron_patterns = None
        hourly_cron_patterns = None
        training_split_meta = None
        if policy_type == "trained" and policy_path:
            meta_path = os.path.join(policy_path, "training_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        training_meta = json.load(f)
                    training_split_meta = training_meta.get("data_split")
                except Exception as e:
                    logger.warning("Failed to load training metadata: %s", e)

            cron_file = os.path.join(policy_path, "cron_patterns.json")
            if os.path.exists(cron_file):
                try:
                    with open(cron_file) as f:
                        data = json.load(f)
                    cron_patterns = data.get("patterns", [])
                    hourly_cron_patterns = data.get("hourly_patterns")
                    logger.info(
                        "Loaded %d daily + %d hourly cron patterns from %s",
                        len(cron_patterns),
                        len(hourly_cron_patterns or []),
                        cron_file,
                    )
                except Exception as e:
                    logger.warning("Failed to load cron patterns: %s", e)

        # Allow explicit split timestamps in config (for heuristics or overrides)
        config_split_ts = parameters.get("split_timestamps")
        if config_split_ts and not training_split_meta:
            training_split_meta = config_split_ts

        # Build dataset with no split ratios — we slice by timestamp below
        eval_split = parameters.get("eval_split", "all")
        effective_dataset_params = dict(dataset_params)
        effective_dataset_params["train_ratio"] = 1.0
        effective_dataset_params["val_ratio"] = 0.0

        try:
            dataset, _ = self._build_dataset(
                sync_unit_obj, task_type, effective_dataset_params,
                cron_patterns=cron_patterns,
                hourly_cron_patterns=hourly_cron_patterns,
            )
        except Exception as e:
            return OrchestratorResponse(
                success=False,
                message=f"Failed to build dataset: {e}",
                is_done=True,
            )

        full_timestamps, full_actual_values, full_covariates = (
            self._extract_data_arrays(dataset, task_type)
        )

        # Slice to requested eval split using training metadata timestamps
        timestamps, actual_values, covariates = self._apply_eval_split_by_timestamp(
            eval_split, training_split_meta, dataset_params,
            full_timestamps, full_actual_values, full_covariates,
        )
        bootstrap_timestamps, bootstrap_actual_values = self._history_before_split(
            timestamps[0], full_timestamps, full_actual_values,
        )

        prediction_provider = self._build_prediction_provider(
            sync_unit_obj, prediction_task_type_str, prediction_params,
            timestamps, actual_values,
        )

        # Inject raw predictions into ModelPredictionPolicy
        if isinstance(policy, ModelPredictionPolicy):
            if isinstance(prediction_provider, PrecomputedPredictionProvider):
                policy.set_predictions(prediction_provider.raw_values)
            else:
                return OrchestratorResponse(
                    success=False,
                    message=(
                        "ModelPredictionPolicy requires precomputed predictions. "
                        "Set predictions.mode='precomputed' in the config."
                    ),
                    is_done=True,
                )

        # Create evaluation environment (full split, no random start)
        env_config = {
            "timestamps": timestamps,
            "actual_values": actual_values,
            "covariates": covariates,
            "prediction_provider": prediction_provider,
            "task_type": task_type_str,
            "prediction_task_type": prediction_task_type_str,
            "episode_length": None,
            "random_start": False,
            "prediction_horizon_steps": env_params.get(
                "prediction_horizon_steps", 60
            ),
            "observation_features": env_params.get("observation_features", [
                "time_since_last_sync",
                "current_aoi",
                "predictions",
                "covariates",
                "items_synced_last",
                "cumulative_empty_syncs",
            ]),
            "empty_sync_window": env_params.get("empty_sync_window", 60),
            "reward_config": reward_params,
            "bootstrap_timestamps": bootstrap_timestamps,
            "bootstrap_actual_values": bootstrap_actual_values,
        }

        try:
            env = SyncSchedulingEnv(env_config)
            decisions_df, summary = run_evaluation(policy, env)

            if not summary:
                return OrchestratorResponse(
                    success=False,
                    message=f"Policy '{policy_name}' produced no evaluation data",
                    is_done=True,
                )

            # Rescore with the standard reward using true_aoi so that
            # total_reward is comparable across all policies.
            std_rewards = rescore_decisions(decisions_df, standard_rc)
            summary["total_reward"] = float(std_rewards.sum())
            summary["mean_reward_per_step"] = float(
                std_rewards.sum() / len(std_rewards),
            )
            summary["reward_config_used"] = reward_params
            summary["is_standard_reward"] = True

            # If the caller provided a non-standard reward (e.g. the
            # agent's training reward), compute secondary metrics for
            # diagnostics.
            if caller_reward_params:
                try:
                    caller_rc = RewardConfig.from_dict(
                        {**caller_reward_params, "mode": task_type_str},
                    )
                    if not caller_rc.is_standard_eval():
                        tr_rewards = rescore_decisions(
                            decisions_df, caller_rc,
                        )
                        summary["training_reward_total"] = float(
                            tr_rewards.sum(),
                        )
                        summary["training_reward_mean"] = float(
                            tr_rewards.sum() / len(tr_rewards),
                        )
                        summary["training_reward_config"] = (
                            caller_reward_params
                        )
                except (ValueError, TypeError):
                    pass

            eval_dir = self._get_eval_dir(sync_unit_obj, policy_name)
            policy_metadata = {
                "policy_type": policy_type,
                "policy_path": policy_path,
                "policy_checkpoint": policy_checkpoint,
                "heuristic_config": heuristic_config,
                "policy_name": policy_name,
                "task_type": task_type_str,
                "reward_config": reward_params,
                "caller_reward_config": caller_reward_params or None,
                "env_config": env_params,
                "prediction_config": prediction_params,
                "prediction_cache_metadata": (
                    prediction_provider.metadata
                    if hasattr(prediction_provider, "metadata")
                    else None
                ),
                "dataset_config": dataset_params,
                "eval_split": eval_split,
                "total_steps": len(decisions_df),
                "sync_unit": sync_unit_obj.get_name(),
                "is_standard_reward": True,
                "evaluation_bootstrap_cutoff": (
                    str(bootstrap_timestamps[-1]) if bootstrap_timestamps else None
                ),
                "evaluation_bootstrap_steps": len(bootstrap_timestamps),
            }
            if policy_type == "trained" and policy_path:
                artifact_path = self._resolve_policy_artifact(
                    policy_path, policy_checkpoint
                )
                policy_metadata["evaluated_policy_artifact"] = artifact_path
                policy_metadata["evaluated_policy_sha256"] = self._file_sha256(
                    artifact_path
                )
            if training_split_meta:
                policy_metadata["data_split"] = dict(training_split_meta)
            elif eval_split != "all":
                n = len(dataset.timestamps)
                train_ratio = dataset_params.get("train_ratio", 0.7)
                val_ratio = dataset_params.get("val_ratio", 0.15)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                policy_metadata["data_split"] = {
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "train_end_timestamp": str(dataset.timestamps[n_train - 1]),
                    "val_end_timestamp": str(dataset.timestamps[n_train + n_val - 1]),
                    "data_start_timestamp": str(dataset.timestamps[0]),
                    "data_end_timestamp": str(dataset.timestamps[-1]),
                    "train_size": n_train,
                    "val_size": n_val,
                    "test_size": n - n_train - n_val,
                }
            write_evaluation_artifacts(
                eval_dir, decisions_df, summary, policy_metadata
            )

            # Compute generalization metrics (train/val/test comparison)
            compute_gen = parameters.get("compute_generalization", True)
            if compute_gen and eval_split != "all" and training_split_meta:
                self._compute_split_metrics_inline(
                    eval_dir=eval_dir,
                    policy=policy,
                    task_type_str=task_type_str,
                    prediction_task_type_str=prediction_task_type_str,
                    env_params=env_params,
                    reward_params=reward_params,
                    prediction_params=prediction_params,
                    training_split_meta=training_split_meta,
                    dataset_params=dataset_params,
                    full_timestamps=full_timestamps,
                    full_actual_values=full_actual_values,
                    full_covariates=full_covariates,
                    primary_split=eval_split,
                    primary_summary=summary,
                    sync_unit_obj=sync_unit_obj,
                )

            response = OrchestratorResponse(
                success=True,
                message=(
                    f"Policy '{policy_name}' evaluated: "
                    f"{summary['total_steps']} steps, "
                    f"reward={summary['total_reward']:.2f}, "
                    f"sync_rate={summary['sync_frequency']:.2%}"
                ),
                is_done=True,
            )
            response.data = {
                "evaluation_path": eval_dir,
                "summary": summary,
            }
            return response

        except Exception as e:
            logger.error("Evaluation failed: %s", e, exc_info=True)
            return OrchestratorResponse(
                success=False,
                message=f"Evaluation failed: {e}",
                is_done=True,
            )

    # compare_policies

    def _handle_compare(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Compare multiple evaluated policies side-by-side."""
        evaluation_paths = parameters.get("evaluation_paths", [])
        output_name = parameters.get("output_name", "comparison")

        if len(evaluation_paths) < 2:
            return OrchestratorResponse(
                success=False,
                message="Need at least 2 evaluation_paths to compare",
                is_done=True,
            )

        comparison = {}
        for path in evaluation_paths:
            summary_file = os.path.join(path, "summary_metrics.json")
            meta_file = os.path.join(path, "policy_metadata.json")

            if not os.path.exists(summary_file):
                logger.warning("Missing summary_metrics.json in %s", path)
                continue

            with open(summary_file) as f:
                summary = json.load(f)

            meta = {}
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    meta = json.load(f)

            name = meta.get("policy_name", os.path.basename(path.rstrip("/")))
            comparison[name] = {
                "summary": summary,
                "metadata": meta,
                "path": path,
            }

        # Validate compatibility across compared evaluations
        comparison_warnings = self._validate_comparison_compatibility(
            comparison
        )
        for w in comparison_warnings:
            logger.warning("Comparison warning: %s", w)

        base_path = os.path.join("data", sync_unit_obj.get_path_for_sync_unit())
        comparison_dir = os.path.join(
            base_path, RL_POLICIES_FOLDER,
            f"{output_name}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        )
        os.makedirs(comparison_dir, exist_ok=True)

        comparison_output = {
            "policies": comparison,
            "warnings": comparison_warnings,
        }
        with open(os.path.join(comparison_dir, "comparison.json"), "w") as f:
            json.dump(comparison_output, f, indent=2)

        msg = f"Compared {len(comparison)} policies"
        if comparison_warnings:
            msg += f" ({len(comparison_warnings)} compatibility warnings)"

        response = OrchestratorResponse(
            success=True,
            message=msg,
            is_done=True,
        )
        response.data = {
            "comparison_path": comparison_dir,
            "policies": list(comparison.keys()),
            "comparison": comparison,
            "warnings": comparison_warnings,
        }
        return response

    # Generalization metrics

    def _compute_split_metrics_inline(
        self,
        eval_dir: str,
        policy,
        task_type_str: str,
        prediction_task_type_str: str,
        env_params: dict,
        reward_params: dict,
        prediction_params: dict,
        training_split_meta: dict,
        dataset_params: dict,
        full_timestamps: list,
        full_actual_values: list,
        full_covariates: dict,
        primary_split: str,
        primary_summary: dict,
        sync_unit_obj,
    ) -> None:
        """Run lightweight eval on train/val splits and write split_metrics.json."""
        splits_to_compute = [s for s in ("train", "val", "test") if s != primary_split]

        split_metrics = {primary_split: {
            k: v for k, v in primary_summary.items()
            if isinstance(v, (int, float, str, bool)) and k != "problem_periods"
        }}

        standard_rc = RewardConfig.standard_eval(mode=task_type_str)

        for split_name in splits_to_compute:
            try:
                ts, vals, covs = self._apply_eval_split_by_timestamp(
                    split_name, training_split_meta, dataset_params,
                    full_timestamps, full_actual_values, full_covariates,
                )
            except ValueError:
                logger.info("No data for split '%s', skipping", split_name)
                continue

            pred_provider = self._build_prediction_provider(
                sync_unit_obj, prediction_task_type_str, prediction_params,
                ts, vals,
            )
            bootstrap_ts, bootstrap_vals = self._history_before_split(
                ts[0], full_timestamps, full_actual_values,
            )

            env_config = {
                "timestamps": ts,
                "actual_values": vals,
                "covariates": covs,
                "prediction_provider": pred_provider,
                "task_type": task_type_str,
                "prediction_task_type": prediction_task_type_str,
                "episode_length": None,
                "random_start": False,
                "prediction_horizon_steps": env_params.get(
                    "prediction_horizon_steps", 60
                ),
                "observation_features": env_params.get("observation_features", [
                    "time_since_last_sync", "current_aoi",
                    "predictions", "covariates", "cumulative_empty_syncs",
                ]),
                "empty_sync_window": env_params.get("empty_sync_window", 60),
                "reward_config": reward_params,
                "bootstrap_timestamps": bootstrap_ts,
                "bootstrap_actual_values": bootstrap_vals,
            }

            try:
                env = SyncSchedulingEnv(env_config)
                split_summary = run_evaluation_summary_only(policy, env)
                split_metrics[split_name] = split_summary
                logger.info(
                    "Split '%s': %d steps, reward/step=%.4f",
                    split_name, split_summary.get("total_steps", 0),
                    split_summary.get("mean_reward_per_step", 0),
                )
            except Exception as e:
                logger.warning("Failed to evaluate split '%s': %s", split_name, e)

        # Compute generalization percentages
        if "train" in split_metrics and "test" in split_metrics:
            train_r = split_metrics["train"].get("mean_reward_per_step", 0)
            test_r = split_metrics["test"].get("mean_reward_per_step", 0)
            if train_r != 0:
                primary_summary["generalization_train_to_test_pct"] = (
                    (test_r - train_r) / abs(train_r) * 100
                )

        if "train" in split_metrics and "val" in split_metrics:
            train_r = split_metrics["train"].get("mean_reward_per_step", 0)
            val_r = split_metrics["val"].get("mean_reward_per_step", 0)
            if train_r != 0:
                primary_summary["generalization_train_to_val_pct"] = (
                    (val_r - train_r) / abs(train_r) * 100
                )

        if "val" in split_metrics and "test" in split_metrics:
            val_r = split_metrics["val"].get("mean_reward_per_step", 0)
            test_r = split_metrics["test"].get("mean_reward_per_step", 0)
            if val_r != 0:
                primary_summary["generalization_val_to_test_pct"] = (
                    (test_r - val_r) / abs(val_r) * 100
                )

        train_end_ts = training_split_meta.get("train_end_timestamp")
        val_end_ts = training_split_meta.get("val_end_timestamp")
        to_save = dict(split_metrics)
        to_save["_train_end_ts"] = train_end_ts
        to_save["_val_end_ts"] = val_end_ts
        cache_path = os.path.join(eval_dir, "split_metrics.json")
        try:
            with open(cache_path, "w") as f:
                json.dump(to_save, f, default=str)
            logger.info("Wrote split_metrics.json to %s", eval_dir)
        except Exception as e:
            logger.warning("Failed to write split_metrics.json: %s", e)

        summary_path = os.path.join(eval_dir, "summary_metrics.json")
        try:
            with open(summary_path) as f:
                saved_summary = json.load(f)
            for key in ("generalization_train_to_test_pct",
                        "generalization_train_to_val_pct",
                        "generalization_val_to_test_pct"):
                if key in primary_summary:
                    saved_summary[key] = primary_summary[key]
            with open(summary_path, "w") as f:
                json.dump(saved_summary, f, default=str)
        except Exception as e:
            logger.warning("Failed to update summary_metrics.json: %s", e)

    # Helpers (reused from RLTrainingApp pattern)

    @staticmethod
    def _validate_comparison_compatibility(
        comparison: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Check that compared policies have compatible evaluation parameters."""
        warnings: List[str] = []
        if len(comparison) < 2:
            return warnings

        names = list(comparison.keys())
        ref_meta = comparison[names[0]].get("metadata", {})
        ref_name = names[0]

        ref_eval_split = ref_meta.get("eval_split", "all")
        ref_task_type = ref_meta.get("task_type", "event_rate")
        ref_total_steps = ref_meta.get("total_steps")
        ref_ds_cfg = ref_meta.get("dataset_config", {})

        for name in names[1:]:
            meta = comparison[name].get("metadata", {})

            eval_split = meta.get("eval_split", "all")
            if eval_split != ref_eval_split:
                warnings.append(
                    f"eval_split mismatch: '{ref_name}' uses '{ref_eval_split}' "
                    f"but '{name}' uses '{eval_split}'"
                )

            task_type = meta.get("task_type", "event_rate")
            if task_type != ref_task_type:
                warnings.append(
                    f"task_type mismatch: '{ref_name}' uses '{ref_task_type}' "
                    f"but '{name}' uses '{task_type}'"
                )

            total_steps = meta.get("total_steps")
            if (ref_total_steps and total_steps
                    and total_steps != ref_total_steps):
                warnings.append(
                    f"total_steps mismatch: '{ref_name}' has {ref_total_steps} "
                    f"but '{name}' has {total_steps}"
                )

            ds_cfg = meta.get("dataset_config", {})
            for key in ("start_time", "end_time"):
                ref_val = ref_ds_cfg.get(key)
                cmp_val = ds_cfg.get(key)
                if ref_val and cmp_val and ref_val != cmp_val:
                    warnings.append(
                        f"dataset_config.{key} mismatch: '{ref_name}' uses "
                        f"'{ref_val}' but '{name}' uses '{cmp_val}'"
                    )

        return warnings

    def _build_dataset(
        self, sync_unit_obj, task_type, dataset_params,
        cron_patterns=None, hourly_cron_patterns=None,
    ):
        start_time_str = dataset_params.get("start_time")
        end_time_str = dataset_params.get("end_time")

        if start_time_str:
            start_time = datetime.fromisoformat(
                start_time_str.replace("Z", "+00:00")
            )
        else:
            start_time = datetime.min.replace(tzinfo=UTC)

        if end_time_str:
            end_time = datetime.fromisoformat(
                end_time_str.replace("Z", "+00:00")
            )
        else:
            end_time = datetime.max.replace(tzinfo=UTC)

        resample_freq = dataset_params.get("resample_freq")
        if task_type == TaskType.EVENT_RATE:
            if resample_freq is None:
                resample_freq = "1min"
                logger.info(
                    "resample_freq not set — defaulting to '1min' for "
                    "event_rate policy evaluation"
                )
            elif resample_freq != "1min":
                logger.warning(
                    "resample_freq='%s' overridden to '1min' — policy "
                    "evaluation requires minute-level event_rate data",
                    resample_freq,
                )
                resample_freq = "1min"

        config = DatasetConfig(
            task_type=task_type,
            start_time=start_time,
            end_time=end_time,
            train_ratio=dataset_params.get("train_ratio", 1.0),
            val_ratio=dataset_params.get("val_ratio", 0.0),
            filters=dataset_params.get("filters", {}),
            resample_freq=resample_freq,
            covariates=dataset_params.get("covariates", []),
        )

        if task_type == TaskType.EVENT_RATE:
            return build_event_rate_dataset(
                sync_unit_obj, config, self._sa,
                cron_patterns=cron_patterns,
                hourly_cron_patterns=hourly_cron_patterns,
            )
        elif task_type == TaskType.TIME_TO_EVENT:
            return build_time_to_event_dataset(sync_unit_obj, config, self._sa)
        else:
            raise NotImplementedError(f"Unsupported task type: {task_type}")

    def _extract_data_arrays(self, dataset, task_type):
        timestamps = list(dataset.timestamps)
        if task_type == TaskType.EVENT_RATE:
            actual_values = list(dataset.events_per_minute)
        else:
            actual_values = list(dataset.minutes_to_next_event)
        covariates = {k: list(v) for k, v in dataset.covariates.items()}
        return timestamps, actual_values, covariates

    @staticmethod
    def _apply_eval_split_by_timestamp(
        eval_split, training_split_meta, dataset_params,
        timestamps, actual_values, covariates,
    ):
        """Slice data arrays using timestamp boundaries from training metadata.

        For trained policies, uses val_end_timestamp / train_end_timestamp from
        the agent's training metadata so each agent is evaluated on its own
        true out-of-sample segment. For heuristic policies (no training metadata),
        falls back to ratio-based slicing from the eval config.
        """
        if eval_split == "all":
            return timestamps, actual_values, covariates

        # Determine boundary timestamps
        if training_split_meta:
            train_end = training_split_meta.get("train_end_timestamp")
            val_end = training_split_meta.get("val_end_timestamp")
        else:
            train_end = None
            val_end = None

        # Resolve start/end for the requested split
        from pandas import Timestamp

        if eval_split == "test":
            if val_end:
                split_start = Timestamp(val_end)
                logger.info(
                    "Eval split 'test': using val_end_timestamp=%s from training metadata",
                    val_end,
                )
            elif train_end:
                split_start = Timestamp(train_end)
                logger.warning(
                    "No val_end_timestamp — using train_end_timestamp=%s as test start",
                    train_end,
                )
            else:
                split_start = SyncPolicyEvaluationApp._ratio_based_boundary(
                    timestamps, dataset_params, "test"
                )
            split_end = None
        elif eval_split == "val":
            if train_end and val_end:
                split_start = Timestamp(train_end)
                split_end = Timestamp(val_end)
                logger.info(
                    "Eval split 'val': %s to %s from training metadata",
                    train_end, val_end,
                )
            else:
                split_start = SyncPolicyEvaluationApp._ratio_based_boundary(
                    timestamps, dataset_params, "val_start"
                )
                split_end = SyncPolicyEvaluationApp._ratio_based_boundary(
                    timestamps, dataset_params, "val_end"
                )
        elif eval_split == "train":
            split_start = None
            if train_end:
                split_end = Timestamp(train_end)
            else:
                split_end = SyncPolicyEvaluationApp._ratio_based_boundary(
                    timestamps, dataset_params, "train_end"
                )
        else:
            raise ValueError(
                f"Invalid eval_split '{eval_split}'. "
                f"Must be 'all', 'train', 'val', or 'test'."
            )

        # Find slice indices by timestamp comparison
        start_idx = 0
        end_idx = len(timestamps)

        if split_start is not None:
            # No timestamp clears the start boundary => the split is empty. Seek
            # past the end so the guard below fires; leaving start_idx at 0 would
            # silently return the whole span as the requested split
            start_idx = end_idx
            for i, ts in enumerate(timestamps):
                if Timestamp(ts) > split_start:
                    start_idx = i
                    break

        if split_end is not None:
            # No timestamp exceeds the end boundary => the split legitimately runs
            # to the end of the data, so end_idx stays at len(timestamps)
            for i, ts in enumerate(timestamps):
                if Timestamp(ts) > split_end:
                    end_idx = i
                    break

        if start_idx >= end_idx:
            raise ValueError(
                f"No data for eval_split='{eval_split}' with boundaries "
                f"start={split_start}, end={split_end}."
            )

        sliced_ts = timestamps[start_idx:end_idx]
        sliced_vals = actual_values[start_idx:end_idx]
        sliced_covs = {k: v[start_idx:end_idx] for k, v in covariates.items()}

        logger.info(
            "Eval split '%s': %d steps (%s to %s)",
            eval_split, len(sliced_ts), sliced_ts[0], sliced_ts[-1],
        )
        return sliced_ts, sliced_vals, sliced_covs

    @staticmethod
    def _ratio_based_boundary(timestamps, dataset_params, boundary_type):
        """Fallback: compute a boundary from ratios when no training metadata."""
        from pandas import Timestamp
        n = len(timestamps)
        train_ratio = dataset_params.get("train_ratio", 0.7)
        val_ratio = dataset_params.get("val_ratio", 0.15)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if boundary_type == "train_end":
            return Timestamp(timestamps[n_train - 1])
        elif boundary_type == "val_start":
            return Timestamp(timestamps[n_train - 1])
        elif boundary_type == "val_end":
            return Timestamp(timestamps[n_train + n_val - 1])
        elif boundary_type == "test":
            return Timestamp(timestamps[n_train + n_val - 1])
        raise ValueError(f"Unknown boundary_type: {boundary_type}")

    def _build_prediction_provider(
        self, sync_unit_obj, prediction_task_type: str, prediction_params,
        timestamps, actual_values,
    ):
        mode = prediction_params.get("mode", "none")
        if mode == "masked":
            return MaskedPredictionProvider(prediction_task_type)
        if mode in ("precomputed", "causal_precomputed", "explicit"):
            cache_path = prediction_params.get("cache_path")
            if not cache_path:
                raise ValueError(
                    f"predictions.mode='{mode}' requires a causal cache_path"
                )
            provider = PrecomputedPredictionProvider.from_cache(cache_path)
            provider.validate_timestamps(timestamps)
            cache_task = provider.metadata.get("task_type")
            if cache_task and cache_task != prediction_task_type:
                raise ValueError(
                    f"Prediction cache task_type={cache_task!r} does not match "
                    f"requested {prediction_task_type!r}"
                )
            return provider
        if mode == "action_aware":
            from rl_training.action_aware_forecaster import ExplicitHistoryForecaster

            model_selection = prediction_params.get("model_selection", {})
            if model_selection.get("mode") != "explicit":
                raise ValueError(
                    "action_aware predictions require explicit model_selection"
                )
            model_path = model_selection.get("model_id")
            if not model_path:
                raise ValueError("action_aware predictions require a model_id")
            forecaster = ExplicitHistoryForecaster.from_model_path(
                model_path, prediction_task_type,
            )
            fitting_cutoff = prediction_params.get(
                "forecasting_model_fitting_cutoff"
            )
            if fitting_cutoff is None:
                recorded = forecaster.metadata.get("model_training_data_end")
                if not recorded or str(recorded).startswith("9999-"):
                    raise ValueError(
                        "action_aware historical evaluation requires "
                        "forecasting_model_fitting_cutoff provenance"
                    )
                fitting_cutoff = recorded
                cutoff_source = "model_training_metadata"
            else:
                cutoff_source = "prediction_config"
            forecaster.metadata["forecasting_model_fitting_cutoff"] = fitting_cutoff
            forecaster.metadata["fitting_cutoff_source"] = cutoff_source
            forecaster.metadata["fitting_cutoff_semantics"] = (
                prediction_params.get(
                    "fitting_cutoff_semantics",
                    "last_training_target_timestamp",
                )
            )
            return ActionAwarePredictionProvider(
                forecaster,
                trajectory_horizon_steps=prediction_params.get(
                    "trajectory_horizon_steps", 60
                ),
            )
        return NoPredictionProvider()

    @staticmethod
    def _history_before_split(
        split_start, full_timestamps, full_actual_values,
    ) -> Tuple[list, list]:
        """Return only the complete chronological prefix before a split."""
        try:
            start_index = full_timestamps.index(split_start)
        except ValueError as exc:
            raise ValueError("evaluation split start is absent from full dataset") from exc
        return (
            list(full_timestamps[:start_index]),
            list(full_actual_values[:start_index]),
        )

    @staticmethod
    def _resolve_policy_artifact(
        policy_path: str, checkpoint: str = "final",
    ) -> str:
        """Resolve the exact immutable policy file selected by the loader."""
        return resolve_trained_policy_file(policy_path, checkpoint)

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _get_eval_dir(self, sync_unit_obj, policy_name: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
        folder = f"{policy_name}_{timestamp}"
        base_path = os.path.join("data", sync_unit_obj.get_path_for_sync_unit())
        return os.path.join(base_path, RL_POLICIES_FOLDER, folder)
