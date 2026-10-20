"""CLI for generating timestamp-keyed causal RL forecast caches."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_serving.model_serving_app import ModelServingApp
from modeling.core_types import DatasetConfig, TaskType
from modeling.dataset_builder import build_event_rate_dataset, build_time_to_event_dataset
from rl_training.causal_prediction_cache import (
    CausalContextBuilder,
    CausalForecastGenerator,
    save_causal_cache,
)
from storage_access import StorageAccess
from sync_unit import SyncUnit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-unit", required=True, help="organization/system/project")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task-type", required=True, choices=["event_rate", "time_to_event"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="Benchmark only the first N origins")
    return parser.parse_args()


def _dataset_config(task_type: TaskType, resample_freq: str | None, filters=None):
    return DatasetConfig(
        task_type=task_type,
        start_time=datetime.min,
        end_time=datetime.max,
        train_ratio=1.0,
        val_ratio=0.0,
        filters=filters or {},
        resample_freq=resample_freq,
    )


def main() -> None:
    args = _parse_args()
    parts = args.sync_unit.split("/")
    if len(parts) != 3:
        raise ValueError("--sync-unit must be organization/system/project")
    sync_unit = SyncUnit(*parts)
    storage = StorageAccess()
    model, model_info = ModelServingApp._load_from_artifact_path(
        args.model_path, TaskType(args.task_type)
    )
    with open(os.path.join(args.model_path, "training_metadata.json"), encoding="utf-8") as f:
        training_metadata = json.load(f)

    decision_dataset, _ = build_event_rate_dataset(
        sync_unit, _dataset_config(TaskType.EVENT_RATE, "1min"), storage
    )
    origins = list(decision_dataset.timestamps)
    if args.limit:
        origins = origins[:args.limit]

    model_freq = getattr(model, "_freq", None) or training_metadata.get("resample_freq") or "1min"
    input_size = int(
        model.model_config.get("input_size", model.model_config.get("input_window", 30))
    )
    task_type = TaskType(args.task_type)
    if task_type == TaskType.EVENT_RATE:
        source_timestamps = decision_dataset.timestamps
        source_values = decision_dataset.events_per_minute
    else:
        snapshot_filters = {}
        snapshot_path = training_metadata.get("snapshot_metadata_path")
        if snapshot_path:
            full_snapshot_path = storage.create_full_path_string(snapshot_path)
            if os.path.isfile(full_snapshot_path):
                with open(full_snapshot_path, encoding="utf-8") as f:
                    snapshot_filters = json.load(f).get("filters", {})
        event_dataset, _ = build_time_to_event_dataset(
            sync_unit,
            _dataset_config(TaskType.TIME_TO_EVENT, None, snapshot_filters),
            storage,
        )
        source_timestamps = event_dataset.timestamps
        source_values = None

    context_builder = CausalContextBuilder(
        task_type, source_timestamps, source_values, model_freq, input_size
    )
    predictions = CausalForecastGenerator(model, context_builder, args.horizon).generate(
        origins, batch_size=args.batch_size
    )
    save_causal_cache(
        args.output,
        origins,
        predictions,
        {
            "sync_unit": args.sync_unit,
            "task_type": args.task_type,
            "model_id": model_info["model_id"],
            "model_path": args.model_path,
            "model_type": model_info["model_type"],
            "model_frequency": model_freq,
            "input_size": input_size,
            "cold_start_policy": "zero_er_or_censored_tte_left_padding",
            "future_elapsed_policy": "no_event_increment",
            "fitted_parameter_scope": "existing_chronological_training_artifact",
        },
    )
    print(f"Saved {len(origins)} x {args.horizon} predictions to {args.output}")


if __name__ == "__main__":
    main()