"""
Test script to verify the reset_state functionality for apps.

This script tests that the reset_state action correctly resets
the state for a specific action within an app.
"""
import unittest
from unittest.mock import Mock, patch
from orchestrator import Orchestrator
from app_registry import AppRegistry
from data_cleanup.data_cleanup_app import DataCleanupApp
from data_preprocessing.data_preprocessing_app import DataPreprocessingApp
from pipeline_explorer.pipeline_explorer_app import PipelineExplorerApp
import sync_unit
from orchestrator_message import OrchestratorMessage
import storage_access
import state
from data_cleanup.invoke_data_cleaner import DataCleanerState
from data_preprocessing.invoke_data_preprocesser import DataPreprocessingState
from pipeline_explorer.pipeline_explorer_app import PipelineExplorerState


class TestResetStateDataCleanup(unittest.TestCase):
    """Test reset_state functionality for data_cleanup app"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = AppRegistry()
        self.registry.register_app(DataCleanupApp())
        self.orchestrator = Orchestrator(
            app_registry=self.registry,
            enable_sync_unit_logging=False
        )
        self.test_unit = sync_unit.SyncUnit(
            organization_id="test_org",
            external_system_id="test_system",
            project_id="test_project"
        )

    def test_reset_state_action_available(self):
        """Test that reset_state is in available actions"""
        app = self.registry.get_app("data_cleanup")
        self.assertIn("reset_state", app.get_available_actions())

    def test_reset_state_missing_action_name(self):
        """Test reset_state fails when action_name parameter is missing"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={}
        )

        response = self.orchestrator.send_message(message)

        self.assertFalse(response.success)
        self.assertIn("action_name", response.message)

    def test_reset_state_invalid_action_name(self):
        """Test reset_state fails with invalid action name"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "invalid_action"}
        )

        response = self.orchestrator.send_message(message)

        self.assertFalse(response.success)
        self.assertIn("not a valid action", response.message)

    def test_reset_state_valid_parse_json(self):
        """Test reset_state succeeds for parse_json action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "parse_json"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("parse_json", response.message)

    def test_reset_state_valid_clean_data(self):
        """Test reset_state succeeds for clean_data action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "clean_data"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("clean_data", response.message)


class TestResetStateDataPreprocessing(unittest.TestCase):
    """Test reset_state functionality for data_preprocessing app"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = AppRegistry()
        self.registry.register_app(DataPreprocessingApp())
        self.orchestrator = Orchestrator(
            app_registry=self.registry,
            enable_sync_unit_logging=False
        )
        self.test_unit = sync_unit.SyncUnit(
            organization_id="test_org",
            external_system_id="test_system",
            project_id="test_project"
        )

    def test_reset_state_action_available(self):
        """Test that reset_state is in available actions"""
        app = self.registry.get_app("data_preprocessing")
        self.assertIn("reset_state", app.get_available_actions())

    def test_reset_state_valid_calculate_original_time_deltas(self):
        """Test reset_state succeeds for calculate_original_time_deltas action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "calculate_original_time_deltas"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("calculate_original_time_deltas", response.message)

    def test_reset_state_valid_create_merged_dataset(self):
        """Test reset_state succeeds for create_merged_dataset action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "create_merged_dataset"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("create_merged_dataset", response.message)

    def test_reset_state_valid_calculate_merged_time_deltas(self):
        """Test reset_state succeeds for calculate_merged_time_deltas action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "calculate_merged_time_deltas"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("calculate_merged_time_deltas", response.message)

    def test_reset_state_valid_calculate_merged_events_per_minute(self):
        """Test reset_state succeeds for calculate_merged_events_per_minute action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "calculate_merged_events_per_minute"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("calculate_merged_events_per_minute", response.message)


class TestCalculateMergedEventsPerMinuteAction(unittest.TestCase):
    """Test the calculate_merged_events_per_minute action execution"""

    def setUp(self):
        """Set up test fixtures"""
        import os
        import pandas as pd
        from datetime import datetime

        self.registry = AppRegistry()
        self.registry.register_app(DataPreprocessingApp())
        self.orchestrator = Orchestrator(
            app_registry=self.registry,
            enable_sync_unit_logging=False
        )
        self.test_unit = sync_unit.SyncUnit(
            organization_id="test_org",
            external_system_id="test_system",
            project_id="test_project"
        )

        # Create test merged dataset file
        self.test_data_dir = f"data/{self.test_unit.get_path_for_sync_unit()}/data_preprocessing"
        os.makedirs(self.test_data_dir, exist_ok=True)

        # Create a merged dataset DataFrame with 'events' column
        self.test_df = pd.DataFrame({
            'id': ['1', '2', '3', '4', '5', '6'],
            'events': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 30),
                datetime(2024, 1, 1, 10, 1, 0),
                datetime(2024, 1, 1, 10, 1, 15),
                datetime(2024, 1, 1, 10, 1, 45),
                datetime(2024, 1, 1, 10, 2, 0)
            ]
        })

        # Save as merged_dataset_1.pkl (chunk number 1)
        self.merged_dataset_path = f"{self.test_data_dir}/merged_dataset_1.pkl"
        self.test_df.to_pickle(self.merged_dataset_path)

        # Reset the state for calculate_merged_events_per_minute
        reset_message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "calculate_merged_events_per_minute"}
        )
        self.orchestrator.send_message(reset_message)

    def tearDown(self):
        """Clean up test files"""
        import os

        # Remove test files
        output_path = f"{self.test_data_dir}/merged_events_per_minute_1.pkl"
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(self.merged_dataset_path):
            os.remove(self.merged_dataset_path)

    def test_calculate_merged_events_per_minute_action(self):
        """Test that calculate_merged_events_per_minute action processes file correctly"""
        import os
        import pandas as pd

        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="calculate_merged_events_per_minute"
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertTrue(response.is_done)

        # Verify output file was created
        output_path = f"{self.test_data_dir}/merged_events_per_minute_1.pkl"
        self.assertTrue(os.path.exists(output_path))

        # Read and verify the output
        result_df = pd.read_pickle(output_path)

        # Should have 3 minutes: 10:00, 10:01, 10:02
        self.assertEqual(len(result_df), 3)
        self.assertIn('minute', result_df.columns)
        self.assertIn('events_per_minute', result_df.columns)

        # Check event counts per minute
        # 10:00 - 2 events, 10:01 - 3 events, 10:02 - 1 event
        events_counts = result_df['events_per_minute'].tolist()
        self.assertEqual(events_counts[0], 2)  # 10:00
        self.assertEqual(events_counts[1], 3)  # 10:01
        self.assertEqual(events_counts[2], 1)  # 10:02

    def test_calculate_merged_events_per_minute_no_input_file(self):
        """Test that action handles missing input file gracefully"""
        import os

        # Remove the merged dataset file
        if os.path.exists(self.merged_dataset_path):
            os.remove(self.merged_dataset_path)

        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="calculate_merged_events_per_minute"
        )

        response = self.orchestrator.send_message(message)

        # Should succeed with "all files processed" message (no files to process)
        self.assertTrue(response.success)
        self.assertTrue(response.is_done)
        self.assertIn("Processed all existing files", response.message)


class TestResetStatePipelineExplorer(unittest.TestCase):
    """Test reset_state functionality for pipeline_explorer app"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = AppRegistry()
        self.registry.register_app(PipelineExplorerApp())
        self.orchestrator = Orchestrator(
            app_registry=self.registry,
            enable_sync_unit_logging=False
        )
        self.test_unit = sync_unit.SyncUnit(
            organization_id="test_org",
            external_system_id="test_system",
            project_id="test_project"
        )

    def test_reset_state_action_available(self):
        """Test that reset_state is in available actions"""
        app = self.registry.get_app("pipeline_explorer")
        self.assertIn("reset_state", app.get_available_actions())

    def test_reset_state_missing_action_name(self):
        """Test reset_state fails when action_name parameter is missing"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={}
        )

        response = self.orchestrator.send_message(message)

        self.assertFalse(response.success)
        self.assertIn("action_name", response.message)

    def test_reset_state_invalid_action_name(self):
        """Test reset_state fails with invalid action_name"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "invalid_action"}
        )

        response = self.orchestrator.send_message(message)

        self.assertFalse(response.success)
        self.assertIn("not a valid action", response.message)

    def test_reset_state_calculate_all_pipeline_metadata(self):
        """Test reset_state for calculate_all_pipeline_metadata action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "calculate_all_pipeline_metadata"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("Successfully reset state", response.message)
        self.assertIn("calculate_all_pipeline_metadata", response.message)

    def test_reset_state_validate_pipeline(self):
        """Test reset_state for validate_pipeline action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "validate_pipeline"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("Successfully reset state", response.message)

    def test_reset_state_analysis_per_record(self):
        """Test reset_state for analysis_per_record action"""
        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "analysis_per_record"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)
        self.assertIn("Successfully reset state", response.message)

    @patch.object(storage_access.StorageAccess, 'read_json_as_object')
    @patch.object(storage_access.StorageAccess, 'save_state')
    @patch.object(storage_access.StorageAccess, 'save_file_stats')
    @patch.object(storage_access.StorageAccess, 'check_if_file_exists')
    @patch.object(storage_access.StorageAccess, 'create_folder')
    def test_reset_state_clears_state_values(
        self,
        mock_create_folder,
        mock_check_exists,
        mock_save_stats,
        mock_save_state,
        mock_read_json
    ):
        """Test that reset_state actually clears the state values"""
        # Create a state with non-default values
        existing_state = PipelineExplorerState()
        existing_state.calculate_all_pipeline_metadata_state.last_processed_chunk_number = 10
        existing_state.calculate_all_pipeline_metadata_state.is_done = True
        existing_state.calculate_all_pipeline_metadata_state.total_processed = 500

        # Mock the read to return existing state
        def read_side_effect(path, constructor):
            if "state" in path.lower():
                return existing_state
            return None

        mock_read_json.side_effect = read_side_effect
        mock_check_exists.return_value = True

        message = OrchestratorMessage(
            sync_unit_obj=self.test_unit,
            action="reset_state",
            parameters={"action_name": "calculate_all_pipeline_metadata"}
        )

        response = self.orchestrator.send_message(message)

        self.assertTrue(response.success)

        # Verify save_state was called
        self.assertTrue(mock_save_state.called)

        # Get the state dict that was saved
        saved_state_dict = mock_save_state.call_args[0][0]

        # Verify the reset action's state was reset to defaults
        reset_state = saved_state_dict["calculate_all_pipeline_metadata_state"]
        self.assertEqual(reset_state["last_processed_chunk_number"], 0)
        self.assertFalse(reset_state["is_done"])
        self.assertEqual(reset_state["total_processed"], 0)


if __name__ == '__main__':
    unittest.main()
