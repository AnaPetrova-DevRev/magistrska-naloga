#!/usr/bin/env python3
"""Create invented input chunks; optionally run the six synthetic pipeline configs."""

from __future__ import annotations

import argparse
from contextlib import chdir
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_records.json"
SYNTHETIC_UNIT = {
    "organization": "test_org",
    "external_system": "test_system",
    "project": "test_project",
}
CONFIG_NAMES = (
    "synthetic_parse.json",
    "synthetic_clean.json",
    "synthetic_original_deltas.json",
    "synthetic_merge.json",
    "synthetic_merged_deltas.json",
    "synthetic_event_counts.json",
)
ACTIONS = (
    "parse_json", "clean_data", "calculate_original_time_deltas",
    "create_merged_dataset", "calculate_merged_time_deltas",
    "calculate_merged_events_per_minute",
)


def load_fixture(path: Path) -> dict:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != 1 or fixture.get("sync_unit") != SYNTHETIC_UNIT:
        raise ValueError("Expected the version 1 test_org synthetic fixture")
    chunks = fixture.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 2:
        raise ValueError("Expected two non-empty synthetic input chunks")
    for chunk in chunks:
        if not isinstance(chunk, list) or not chunk:
            raise ValueError("Synthetic chunks must be non-empty lists")
        for record in chunk:
            if not isinstance(record, dict) or set(record) != {
                "id", "created_date", "modified_date",
            }:
                raise ValueError("Synthetic records need id, created_date and modified_date")
            if not isinstance(record["id"], str) or not re.fullmatch(
                r"synthetic-\d{3}", record["id"],
            ):
                raise ValueError("Fixture identifiers must use the synthetic-NNN format")
            try:
                created = datetime.fromisoformat(record["created_date"])
                modified = datetime.fromisoformat(record["modified_date"])
                if created.tzinfo != UTC or modified.tzinfo != UTC or modified < created:
                    raise ValueError("Expected UTC dates with modified_date >= created_date")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid synthetic timestamps for {record['id']}") from error
    return fixture


def create_fixture(root: Path, *, fixture_path: Path = FIXTURE_PATH) -> Path:
    """Create a fresh data directory, refusing any existing data or registry."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("The output root must be an existing directory")
    data = root / "data"
    if data.exists() or data.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing data: {data}")
    fixture = load_fixture(fixture_path)
    initial = Path("test_org/test_system/test_project/initial_data")
    files = {
        Path("sync_units_register.json"): json.dumps([SYNTHETIC_UNIT], indent=2) + "\n",
    }
    for number, records in enumerate(fixture["chunks"], start=1):
        files[initial / f"work_{number}.json"] = "\n".join(
            json.dumps(record, indent=2) for record in records
        ) + "\n"

    # mkdir is exclusive, so a concurrent creator cannot have its data overwritten.
    data.mkdir()
    try:
        # The existing cleaner expects its output directory to be provisioned.
        (data / initial.parent / "clean_data").mkdir(parents=True)
        for relative, content in files.items():
            target = data / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                handle.write(content)
    except Exception:
        shutil.rmtree(data)
        raise
    return data


def run_pipeline(root: Path) -> list:
    """Run only cleanup and preprocessing; no forecasting or RL imports are needed."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from app_registry import AppRegistry
    from data_cleanup.data_cleanup_app import DataCleanupApp
    from data_preprocessing.data_preprocessing_app import DataPreprocessingApp
    from orchestrator import Orchestrator
    from orchestrator_message import OrchestratorMessage
    from sync_unit import SyncUnit

    configs = [
        json.loads((PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8"))
        for name in CONFIG_NAMES
    ]
    unit = SyncUnit("test_org", "test_system", "test_project")
    for config, action in zip(configs, ACTIONS, strict=True):
        if (
            config.get("action") != action
            or config.get("apply_to_all") is not False
            or config.get("sync_units") != [unit.get_name()]
            or config.get("parameters") != {}
        ):
            raise ValueError("Synthetic configs must target only the fixed test sync unit")
    registry = AppRegistry()
    registry.register_app(DataCleanupApp())
    registry.register_app(DataPreprocessingApp())
    orchestrator = Orchestrator(registry, enable_sync_unit_logging=False)
    responses = []
    with chdir(root):
        register_path = Path("data/sync_units_register.json")
        if json.loads(register_path.read_text(encoding="utf-8")) != [SYNTHETIC_UNIT]:
            raise ValueError("Expected an isolated synthetic fixture registry")
        for config in configs:
            response = orchestrator.send_message(OrchestratorMessage(
                sync_unit_obj=unit, action=config["action"], parameters={},
            ))
            if not response.success:
                raise RuntimeError(f"{config['action']} failed: {response.message}")
            responses.append(response)
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-pipeline", action="store_true")
    arguments = parser.parse_args()
    data = create_fixture(arguments.root)
    print(f"Created two invented input chunks under {data}")
    if arguments.run_pipeline:
        responses = run_pipeline(arguments.root)
        print(f"Completed {len(responses)} cleanup and preprocessing actions")


if __name__ == "__main__":
    main()
