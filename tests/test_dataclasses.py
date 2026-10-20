"""
Tests for dataclass implementations and serialization/deserialization.
"""
import unittest
from datetime import datetime
from dataclasses import asdict

# Import the dataclasses we've refactored
from file_stats import FileStatistics, Stats
from state import State
from sync_unit import SyncUnit
from orchestrator_message import OrchestratorMessage, OrchestratorResponse
from data_cleanup.parse_json import ParsedItem
from data_cleanup.invoke_data_cleaner import DataCleanerState


class TestFileStatisticsDataclass(unittest.TestCase):
    """Test FileStatistics dataclass"""

    def test_create_file_statistics(self):
        """Test creating a FileStatistics object"""
        fs = FileStatistics(
            file_name="test.pkl",
            size=100,
            input_file="input.json",
            output_file="output.pkl",
            created_by_function="parse_json"
        )

        self.assertEqual(fs.file_name, "test.pkl")
        self.assertEqual(fs.size, 100)
        self.assertEqual(fs.input_file, "input.json")
        self.assertEqual(fs.output_file, "output.pkl")
        self.assertEqual(fs.created_by_function, "parse_json")

    def test_file_statistics_to_dict(self):
        """Test converting FileStatistics to dict"""
        fs = FileStatistics(
            file_name="test.pkl",
            size=100,
            created_date=datetime(2026, 2, 7, 12, 0, 0)
        )

        fs_dict = asdict(fs)
        self.assertEqual(fs_dict["file_name"], "test.pkl")
        self.assertEqual(fs_dict["size"], 100)
        self.assertIsInstance(fs_dict["created_date"], datetime)

    def test_file_statistics_from_dict(self):
        """Test creating FileStatistics from dict"""
        data = {
            "file_name": "test.pkl",
            "size": 100,
            "input_file": "input.json",
            "output_file": "output.pkl",
            "created_by_function": "parse_json"
        }

        fs = FileStatistics.from_dict(data)
        self.assertEqual(fs.file_name, "test.pkl")
        self.assertEqual(fs.size, 100)
        self.assertEqual(fs.input_file, "input.json")


class TestStatsDataclass(unittest.TestCase):
    """Test Stats dataclass"""

    def test_create_stats(self):
        """Test creating a Stats object"""
        stats = Stats()
        self.assertEqual(len(stats.files), 0)

    def test_add_file_statistics(self):
        """Test adding file statistics"""
        stats = Stats()
        fs = FileStatistics(file_name="test.pkl", size=100)

        stats.add_file_statistics(fs)
        self.assertEqual(len(stats.files), 1)
        self.assertEqual(stats.files[0].file_name, "test.pkl")

    def test_stats_from_dict_array(self):
        """Test creating Stats from array format (as saved)"""
        data = [
            {"file_name": "test1.pkl", "size": 100},
            {"file_name": "test2.pkl", "size": 200}
        ]

        stats = Stats.from_dict(data)
        self.assertEqual(len(stats.files), 2)
        self.assertEqual(stats.files[0].file_name, "test1.pkl")
        self.assertEqual(stats.files[1].file_name, "test2.pkl")


class TestStateDataclass(unittest.TestCase):
    """Test State dataclass"""

    def test_create_state(self):
        """Test creating a State object"""
        state = State(
            last_processed_chunk_number=5,
            is_done=False,
            total_processed=100,
            function_name="parse_json"
        )

        self.assertEqual(state.last_processed_chunk_number, 5)
        self.assertFalse(state.is_done)
        self.assertEqual(state.total_processed, 100)
        self.assertEqual(state.function_name, "parse_json")

    def test_state_to_dict(self):
        """Test converting State to dict using asdict"""
        state = State(
            last_processed_chunk_number=5,
            is_done=False,
            total_processed=100,
            function_name="parse_json"
        )

        state_dict = asdict(state)
        self.assertEqual(state_dict["last_processed_chunk_number"], 5)
        self.assertEqual(state_dict["is_done"], False)
        self.assertEqual(state_dict["total_processed"], 100)
        self.assertEqual(state_dict["function_name"], "parse_json")

    def test_state_from_dict(self):
        """Test creating State from dict"""
        data = {
            "last_processed_chunk_number": 5,
            "is_done": False,
            "total_processed": 100,
            "function_name": "parse_json"
        }

        state = State.from_dict(data)
        self.assertEqual(state.last_processed_chunk_number, 5)
        self.assertFalse(state.is_done)
        self.assertEqual(state.total_processed, 100)
        self.assertEqual(state.function_name, "parse_json")


class TestSyncUnitDataclass(unittest.TestCase):
    """Test SyncUnit dataclass"""

    def test_create_sync_unit(self):
        """Test creating a SyncUnit object"""
        su = SyncUnit(
            organization_id="testorg",
            external_system_id="testsystem",
            project_id="testproject"
        )

        self.assertEqual(su.organization_id, "testorg")
        self.assertEqual(su.external_system_id, "testsystem")
        self.assertEqual(su.project_id, "testproject")

    def test_sync_unit_get_path(self):
        """Test getting path for sync unit"""
        su = SyncUnit(
            organization_id="testorg",
            external_system_id="testsystem",
            project_id="testproject"
        )

        path = su.get_path_for_sync_unit()
        self.assertEqual(path, "testorg/testsystem/testproject")

    def test_sync_unit_equality(self):
        """Test that dataclass provides equality comparison"""
        su1 = SyncUnit("org1", "sys1", "proj1")
        su2 = SyncUnit("org1", "sys1", "proj1")
        su3 = SyncUnit("org2", "sys1", "proj1")

        self.assertEqual(su1, su2)
        self.assertNotEqual(su1, su3)


class TestOrchestratorMessageDataclass(unittest.TestCase):
    """Test OrchestratorMessage dataclass"""

    def test_create_orchestrator_message(self):
        """Test creating an OrchestratorMessage"""
        su = SyncUnit("org1", "sys1", "proj1")
        msg = OrchestratorMessage(su, "parse_json")

        self.assertEqual(msg.action, "parse_json")
        self.assertEqual(msg.get_sync_unit(), su)

    def test_orchestrator_message_with_parameters(self):
        """Test creating message with parameters"""
        su = SyncUnit("org1", "sys1", "proj1")
        params = {"chunk_size": 100, "validate": True}
        msg = OrchestratorMessage(su, "parse_json", parameters=params)

        self.assertEqual(msg.parameters["chunk_size"], 100)
        self.assertTrue(msg.parameters["validate"])


class TestOrchestratorResponseDataclass(unittest.TestCase):
    """Test OrchestratorResponse dataclass"""

    def test_create_response_success(self):
        """Test creating a successful response"""
        response = OrchestratorResponse(True, "Operation completed")

        self.assertTrue(response.success)
        self.assertEqual(response.message, "Operation completed")
        self.assertFalse(response.is_done)
        self.assertEqual(len(response.logs), 0)

    def test_create_response_with_logs(self):
        """Test creating response with logs"""
        logs = ["Log entry 1", "Log entry 2"]
        response = OrchestratorResponse(True, "Done", is_done=True, logs=logs)

        self.assertTrue(response.success)
        self.assertTrue(response.is_done)
        self.assertEqual(len(response.logs), 2)

    def test_mutable_default_bug_fixed(self):
        """Test that the mutable default argument bug is fixed"""
        response1 = OrchestratorResponse(True, "Message 1")
        response2 = OrchestratorResponse(True, "Message 2")

        # Add logs to response1
        response1.logs.append("Log for response1")

        # response2 should have empty logs (bug would cause shared list)
        self.assertEqual(len(response1.logs), 1)
        self.assertEqual(len(response2.logs), 0)


class TestDataCleanerStateDataclass(unittest.TestCase):
    """Test DataCleanerState dataclass"""

    def test_create_data_cleaner_state(self):
        """Test creating a DataCleanerState"""
        state = DataCleanerState()

        self.assertIsInstance(state.parse_json_state, State)
        self.assertIsInstance(state.clean_data_state, State)

    def test_get_state_for_function(self):
        """Test getting state for a specific function"""
        state = DataCleanerState()
        state.parse_json_state.last_processed_chunk_number = 5

        parse_state = state.get_state_for_function("parse_json")
        self.assertEqual(parse_state.last_processed_chunk_number, 5)

    def test_update_state_for_function(self):
        """Test updating state for a specific function"""
        state = DataCleanerState()
        new_state = State(last_processed_chunk_number=10)

        state.update_state_for_function("parse_json", new_state)
        self.assertEqual(state.parse_json_state.last_processed_chunk_number, 10)

    def test_data_cleaner_state_serialization(self):
        """Test to_dict and from_dict"""
        state = DataCleanerState()
        state.parse_json_state.last_processed_chunk_number = 5
        state.clean_data_state.last_processed_chunk_number = 3

        # Serialize
        state_dict = state.to_dict()

        # Deserialize
        restored_state = DataCleanerState.from_dict(state_dict)

        self.assertEqual(restored_state.parse_json_state.last_processed_chunk_number, 5)
        self.assertEqual(restored_state.clean_data_state.last_processed_chunk_number, 3)


class TestPipelineExplorerStateDataclass(unittest.TestCase):
    """Test PipelineExplorerState dataclass"""

    def test_create_pipeline_explorer_state(self):
        """Test creating a PipelineExplorerState"""
        from pipeline_explorer.pipeline_explorer_app import PipelineExplorerState

        pe_state = PipelineExplorerState()

        self.assertIsInstance(pe_state.calculate_all_pipeline_metadata_state, State)
        self.assertIsInstance(pe_state.validate_pipeline_state, State)
        self.assertIsInstance(pe_state.analysis_per_record_state, State)

    def test_get_state_for_function(self):
        """Test getting state for a specific function"""
        from pipeline_explorer.pipeline_explorer_app import PipelineExplorerState
        from app_actions_constants import (
            CALCULATE_ALL_PIPELINE_METADATA,
            VALIDATE_PIPELINE,
            ANALYSIS_PER_RECORD
        )

        pe_state = PipelineExplorerState()
        pe_state.calculate_all_pipeline_metadata_state.last_processed_chunk_number = 5
        pe_state.validate_pipeline_state.last_processed_chunk_number = 3
        pe_state.analysis_per_record_state.last_processed_chunk_number = 7

        metadata_state = pe_state.get_state_for_function(CALCULATE_ALL_PIPELINE_METADATA)
        self.assertEqual(metadata_state.last_processed_chunk_number, 5)

        validate_state = pe_state.get_state_for_function(VALIDATE_PIPELINE)
        self.assertEqual(validate_state.last_processed_chunk_number, 3)

        analysis_state = pe_state.get_state_for_function(ANALYSIS_PER_RECORD)
        self.assertEqual(analysis_state.last_processed_chunk_number, 7)

    def test_get_state_for_invalid_function(self):
        """Test getting state for an invalid function raises ValueError"""
        from pipeline_explorer.pipeline_explorer_app import PipelineExplorerState

        pe_state = PipelineExplorerState()

        with self.assertRaises(ValueError) as context:
            pe_state.get_state_for_function("invalid_function")

        self.assertIn("not supported", str(context.exception))

    def test_update_state_for_function(self):
        """Test updating state for a specific function"""
        from pipeline_explorer.pipeline_explorer_app import PipelineExplorerState
        from app_actions_constants import CALCULATE_ALL_PIPELINE_METADATA

        pe_state = PipelineExplorerState()
        new_state = State(last_processed_chunk_number=10, is_done=True)

        pe_state.update_state_for_function(CALCULATE_ALL_PIPELINE_METADATA, new_state)

        self.assertEqual(
            pe_state.calculate_all_pipeline_metadata_state.last_processed_chunk_number,
            10
        )
        self.assertTrue(pe_state.calculate_all_pipeline_metadata_state.is_done)

    def test_update_state_for_invalid_function(self):
        """Test updating state for an invalid function raises ValueError"""
        from pipeline_explorer.pipeline_explorer_app import PipelineExplorerState

        pe_state = PipelineExplorerState()
        new_state = State()

        with self.assertRaises(ValueError) as context:
            pe_state.update_state_for_function("invalid_function", new_state)

        self.assertIn("not supported", str(context.exception))

    def test_pipeline_explorer_state_serialization(self):
        """Test to_dict and from_dict"""
        from pipeline_explorer.pipeline_explorer_app import PipelineExplorerState

        pe_state = PipelineExplorerState()
        pe_state.calculate_all_pipeline_metadata_state.last_processed_chunk_number = 5
        pe_state.calculate_all_pipeline_metadata_state.is_done = True
        pe_state.validate_pipeline_state.last_processed_chunk_number = 3
        pe_state.analysis_per_record_state.total_processed = 100

        # Serialize
        state_dict = pe_state.to_dict()

        # Verify dict structure
        self.assertIn("calculate_all_pipeline_metadata_state", state_dict)
        self.assertIn("validate_pipeline_state", state_dict)
        self.assertIn("analysis_per_record_state", state_dict)

        # Deserialize
        restored_state = PipelineExplorerState.from_dict(state_dict)

        self.assertEqual(
            restored_state.calculate_all_pipeline_metadata_state.last_processed_chunk_number,
            5
        )
        self.assertTrue(restored_state.calculate_all_pipeline_metadata_state.is_done)
        self.assertEqual(
            restored_state.validate_pipeline_state.last_processed_chunk_number,
            3
        )
        self.assertEqual(restored_state.analysis_per_record_state.total_processed, 100)

    def test_from_dict_with_empty_dict(self):
        """Test from_dict with empty dictionary creates default state"""
        from pipeline_explorer.pipeline_explorer_app import PipelineExplorerState

        pe_state = PipelineExplorerState.from_dict({})

        self.assertEqual(
            pe_state.calculate_all_pipeline_metadata_state.last_processed_chunk_number,
            0
        )
        self.assertFalse(pe_state.validate_pipeline_state.is_done)
        self.assertEqual(pe_state.analysis_per_record_state.total_processed, 0)


if __name__ == '__main__':
    unittest.main()
