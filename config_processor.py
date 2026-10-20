"""
Config Processor - Execute orchestrator actions based on configuration files

This script reads a configuration file that specifies sync units and actions,
then executes them through the orchestrator.
"""
# Must be first: patch pandas nargsort bug on Python 3.14
from modeling.pandas_argsort_patch import apply as _apply_pandas_patch
_apply_pandas_patch()

import json
import logging
import sys
from typing import List, Dict, Any
from orchestrator import Orchestrator
from app_registry import AppRegistry
from data_cleanup.data_cleanup_app import DataCleanupApp
from data_preprocessing.data_preprocessing_app import DataPreprocessingApp
from model_training.model_training_app import ModelTrainingApp
from model_serving.model_serving_app import ModelServingApp
from model_lifecycle.model_lifecycle_app import ModelLifecycleApp
from rl_training.rl_training_app import RLTrainingApp
from sync_policy_evaluation.sync_policy_evaluation_app import SyncPolicyEvaluationApp
from dataset_comparison.dataset_comparison_app import DatasetComparisonApp
from orchestrator_message import OrchestratorMessage
import sync_unit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_registry() -> AppRegistry:
    """Initialize and return the app registry with all apps"""
    registry = AppRegistry()
    registry.register_app(DataCleanupApp())
    registry.register_app(DataPreprocessingApp())
    registry.register_app(ModelTrainingApp())
    registry.register_app(ModelServingApp())
    registry.register_app(ModelLifecycleApp())
    registry.register_app(RLTrainingApp())
    registry.register_app(SyncPolicyEvaluationApp())
    registry.register_app(DatasetComparisonApp())
    return registry


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)


def get_sync_units_from_config(config: Dict[str, Any]) -> List[sync_unit.SyncUnit]:
    """
    Get sync units based on configuration.

    If config has "apply_to_all": true, returns all sync units.
    Otherwise, returns only the specified sync units.
    """
    all_sync_units = sync_unit.get_sync_units()

    if config.get("apply_to_all", False):
        logger.info("Applying to all sync units")
        return all_sync_units

    sync_unit_names = config.get("sync_units", [])

    if not sync_unit_names:
        logger.warning("No sync units specified in configuration")
        return []

    selected_units = []
    all_unit_names = {unit.get_name(): unit for unit in all_sync_units}

    for name in sync_unit_names:
        if name in all_unit_names:
            selected_units.append(all_unit_names[name])
            logger.info(f"Selected sync unit: {name}")
        else:
            logger.warning(f"Sync unit not found: {name}")

    return selected_units


def process_config(config_file: str, dry_run: bool = False):
    """
    Process a configuration file and execute the specified actions.

    Args:
        config_file: Path to the JSON configuration file
        dry_run: If True, only show what would be executed without actually running
    """
    config = load_config(config_file)

    action = config.get("action")
    if not action:
        logger.error("No action specified in configuration")
        sys.exit(1)

    parameters = config.get("parameters", {})

    sync_units = get_sync_units_from_config(config)

    if not sync_units:
        logger.error("No valid sync units to process")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"Configuration Summary:")
    logger.info(f"  Action: {action}")
    logger.info(f"  Parameters: {parameters}")
    logger.info(f"  Sync Units: {len(sync_units)}")
    logger.info("=" * 80)

    if dry_run:
        logger.info("DRY RUN - No actions will be executed")
        for unit in sync_units:
            logger.info(f"Would execute '{action}' on {unit.get_name()}")
        return

    registry = initialize_registry()
    orchestrator = Orchestrator(app_registry=registry, enable_sync_unit_logging=True)

    results = []
    for i, unit in enumerate(sync_units, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing {i}/{len(sync_units)}: {unit.get_name()}")
        logger.info("=" * 80)

        try:
            message = OrchestratorMessage(
                sync_unit_obj=unit,
                action=action,
                parameters=parameters
            )

            response = orchestrator.send_message(message)

            results.append({
                "sync_unit": unit.get_name(),
                "success": response.success,
                "message": response.message
            })

            if response.success:
                logger.info(f"✅ Success: {response.message}")
            else:
                logger.error(f"❌ Failed: {response.message}")

        except Exception as e:
            logger.error(f"❌ Error processing {unit.get_name()}: {e}", exc_info=True)
            results.append({
                "sync_unit": unit.get_name(),
                "success": False,
                "message": str(e)
            })

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 80)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    logger.info(f"Total: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if failed > 0:
        logger.info("")
        logger.info("Failed sync units:")
        for result in results:
            if not result["success"]:
                logger.info(f"  - {result['sync_unit']}: {result['message']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process orchestrator configuration file")
    parser.add_argument("config", help="Path to configuration JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed without running")

    args = parser.parse_args()

    process_config(args.config, dry_run=args.dry_run)
