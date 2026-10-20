"""
Tests for app flow definitions
"""
import os
import shutil
import unittest
from datetime import datetime, timezone, timedelta

import networkx as nx
import pandas as pd
from app_flow_definitions import (
    define_input_output_files,
    define_paths_for_input_output_files,
    build_full_app_flow_graph,
    extract_file_name_format,
    get_file_pipeline,
    extract_chunk_number,
    get_pipeline_file_paths,
    validate_pipeline_file_stats,
    calculate_pipeline_metadata,
    _construct_file_path_for_format,
    InputOutputFiles,
    InputOutputPaths
)
from sync_unit import SyncUnit
from app_actions_constants import *
from global_vars import *


class TestDefineInputOutputFiles(unittest.TestCase):
    """Test the define_input_output_files function"""

    def test_parse_json_action(self):
        """Test that PARSE_JSON action returns correct InputOutputFiles"""
        result = define_input_output_files(PARSE_JSON)

        self.assertIsInstance(result, InputOutputFiles)
        self.assertEqual(result.input_file, INITIAL_DATA_FILE_NAME_FORMAT)
        self.assertEqual(result.input_file_folder, INITIAL_DATA_FOLDER_NAME)
        self.assertEqual(result.output_files, [PARSED_DATA_FILE_NAME_FORMAT])
        self.assertEqual(result.output_file_folder, DATA_CLEANUP_APP_FOLDER_NAME)

    def test_cleanup_data_action(self):
        """Test that CLEANUP_DATA action returns correct InputOutputFiles"""
        result = define_input_output_files(CLEANUP_DATA)

        self.assertIsInstance(result, InputOutputFiles)
        self.assertEqual(result.input_file, PARSED_DATA_FILE_NAME_FORMAT)
        self.assertEqual(result.input_file_folder, DATA_CLEANUP_APP_FOLDER_NAME)
        self.assertEqual(result.output_files, [CLEANED_ITEMS_FILE_NAME_FORMAT])
        self.assertEqual(result.output_file_folder, CLEAN_DATA_FOLDER_NAME)

    def test_calculate_original_time_deltas_action(self):
        """Test that CALCULATE_ORIGINAL_TIME_DELTAS action returns correct InputOutputFiles"""
        result = define_input_output_files(CALCULATE_ORIGINAL_TIME_DELTAS)

        self.assertIsInstance(result, InputOutputFiles)
        self.assertEqual(result.input_file, CLEANED_ITEMS_FILE_NAME_FORMAT)
        self.assertEqual(result.input_file_folder, CLEAN_DATA_FOLDER_NAME)
        self.assertEqual(result.output_files, [
            CREATED_TIME_DELTA_FILE_NAME_FORMAT,
            MODIFIED_TIME_DELTA_FILE_NAME_FORMAT
        ])
        self.assertEqual(result.output_file_folder, DATA_PREPROCESSING_APP_FOLDER_NAME)

    def test_create_merged_dataset_action(self):
        """Test that CREATE_MERGED_DATASET action returns correct InputOutputFiles"""
        result = define_input_output_files(CREATE_MERGED_DATASET)

        self.assertIsInstance(result, InputOutputFiles)
        self.assertEqual(result.input_file, CLEANED_ITEMS_FILE_NAME_FORMAT)
        self.assertEqual(result.input_file_folder, CLEAN_DATA_FOLDER_NAME)
        self.assertEqual(result.output_files, [MERGED_DATASET_FILE_NAME_FORMAT])
        self.assertEqual(result.output_file_folder, DATA_PREPROCESSING_APP_FOLDER_NAME)

    def test_calculate_merged_time_deltas_action(self):
        """Test that CALCULATE_MERGED_TIME_DELTAS action returns correct InputOutputFiles"""
        result = define_input_output_files(CALCULATE_MERGED_TIME_DELTAS)

        self.assertIsInstance(result, InputOutputFiles)
        self.assertEqual(result.input_file, MERGED_DATASET_FILE_NAME_FORMAT)
        self.assertEqual(result.input_file_folder, DATA_PREPROCESSING_APP_FOLDER_NAME)
        self.assertEqual(result.output_files, [MERGED_TIME_DELTA_FILE_NAME_FORMAT])
        self.assertEqual(result.output_file_folder, DATA_PREPROCESSING_APP_FOLDER_NAME)

    def test_invalid_action(self):
        """Test that an invalid action returns ValueError"""
        result = define_input_output_files("invalid_action")

        self.assertIsInstance(result, ValueError)
        self.assertIn("invalid_action", str(result))
        self.assertIn("not mapped", str(result))

    def test_empty_action(self):
        """Test that an empty action returns ValueError"""
        result = define_input_output_files("")

        self.assertIsInstance(result, ValueError)


class TestDefinePathsForInputOutputFiles(unittest.TestCase):
    """Test the define_paths_for_input_output_files function"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_sync_unit = SyncUnit(
            organization_id="test_org",
            external_system_id="test_system",
            project_id="test_project"
        )
        self.test_chunk_number = 5

    def test_parse_json_paths(self):
        """Test that PARSE_JSON action generates correct paths"""
        result = define_paths_for_input_output_files(
            PARSE_JSON,
            self.test_sync_unit,
            self.test_chunk_number
        )

        self.assertIsInstance(result, InputOutputPaths)

        # Check input path
        expected_input = (
            f"test_org/test_system/test_project/"
            f"{INITIAL_DATA_FOLDER_NAME}/work_5.json"
        )
        self.assertEqual(result.input_filepath, expected_input)

        # Check output paths
        expected_output = (
            f"test_org/test_system/test_project/"
            f"{DATA_CLEANUP_APP_FOLDER_NAME}/parsed_items_5.pkl"
        )
        self.assertEqual(len(result.output_filepaths), 1)
        self.assertEqual(result.output_filepaths[0], expected_output)

    def test_cleanup_data_paths(self):
        """Test that CLEANUP_DATA action generates correct paths"""
        result = define_paths_for_input_output_files(
            CLEANUP_DATA,
            self.test_sync_unit,
            self.test_chunk_number
        )

        self.assertIsInstance(result, InputOutputPaths)

        # Check input path
        expected_input = (
            f"test_org/test_system/test_project/"
            f"{DATA_CLEANUP_APP_FOLDER_NAME}/parsed_items_5.pkl"
        )
        self.assertEqual(result.input_filepath, expected_input)

        # Check output paths
        expected_output = (
            f"test_org/test_system/test_project/"
            f"{CLEAN_DATA_FOLDER_NAME}/cleaned_items_5.pkl"
        )
        self.assertEqual(len(result.output_filepaths), 1)
        self.assertEqual(result.output_filepaths[0], expected_output)

    def test_calculate_original_time_deltas_paths(self):
        """Test that CALCULATE_ORIGINAL_TIME_DELTAS action generates
        correct paths with multiple outputs"""
        result = define_paths_for_input_output_files(
            CALCULATE_ORIGINAL_TIME_DELTAS,
            self.test_sync_unit,
            self.test_chunk_number
        )

        self.assertIsInstance(result, InputOutputPaths)

        # Check input path
        expected_input = (
            f"test_org/test_system/test_project/"
            f"{CLEAN_DATA_FOLDER_NAME}/cleaned_items_5.pkl"
        )
        self.assertEqual(result.input_filepath, expected_input)

        # Check output paths - should have 2 outputs
        self.assertEqual(len(result.output_filepaths), 2)
        expected_output_1 = (
            f"test_org/test_system/test_project/"
            f"{DATA_PREPROCESSING_APP_FOLDER_NAME}/created_time_delta_5.pkl"
        )
        expected_output_2 = (
            f"test_org/test_system/test_project/"
            f"{DATA_PREPROCESSING_APP_FOLDER_NAME}/modified_time_delta_5.pkl"
        )
        self.assertEqual(result.output_filepaths[0], expected_output_1)
        self.assertEqual(result.output_filepaths[1], expected_output_2)

    def test_create_merged_dataset_paths(self):
        """Test that CREATE_MERGED_DATASET action generates correct paths"""
        result = define_paths_for_input_output_files(
            CREATE_MERGED_DATASET,
            self.test_sync_unit,
            self.test_chunk_number
        )

        self.assertIsInstance(result, InputOutputPaths)

        # Check input path
        expected_input = (
            f"test_org/test_system/test_project/"
            f"{CLEAN_DATA_FOLDER_NAME}/cleaned_items_5.pkl"
        )
        self.assertEqual(result.input_filepath, expected_input)

        # Check output paths
        expected_output = (
            f"test_org/test_system/test_project/"
            f"{DATA_PREPROCESSING_APP_FOLDER_NAME}/merged_dataset_5.pkl"
        )
        self.assertEqual(len(result.output_filepaths), 1)
        self.assertEqual(result.output_filepaths[0], expected_output)

    def test_calculate_merged_time_deltas_paths(self):
        """Test that CALCULATE_MERGED_TIME_DELTAS action generates
        correct paths"""
        result = define_paths_for_input_output_files(
            CALCULATE_MERGED_TIME_DELTAS,
            self.test_sync_unit,
            self.test_chunk_number
        )

        self.assertIsInstance(result, InputOutputPaths)

        # Check input path
        expected_input = (
            f"test_org/test_system/test_project/"
            f"{DATA_PREPROCESSING_APP_FOLDER_NAME}/merged_dataset_5.pkl"
        )
        self.assertEqual(result.input_filepath, expected_input)

        # Check output paths
        expected_output = (
            f"test_org/test_system/test_project/"
            f"{DATA_PREPROCESSING_APP_FOLDER_NAME}/merged_time_delta_5.pkl"
        )
        self.assertEqual(len(result.output_filepaths), 1)
        self.assertEqual(result.output_filepaths[0], expected_output)

    def test_different_chunk_numbers(self):
        """Test that different chunk numbers generate different paths"""
        result_1 = define_paths_for_input_output_files(PARSE_JSON, self.test_sync_unit, 1)
        result_10 = define_paths_for_input_output_files(PARSE_JSON, self.test_sync_unit, 10)
        result_100 = define_paths_for_input_output_files(PARSE_JSON, self.test_sync_unit, 100)

        self.assertIn("work_1.json", result_1.input_filepath)
        self.assertIn("work_10.json", result_10.input_filepath)
        self.assertIn("work_100.json", result_100.input_filepath)

        self.assertIn("parsed_items_1.pkl", result_1.output_filepaths[0])
        self.assertIn("parsed_items_10.pkl", result_10.output_filepaths[0])
        self.assertIn("parsed_items_100.pkl", result_100.output_filepaths[0])

    def test_different_sync_units(self):
        """Test that different sync units generate different paths"""
        sync_unit_1 = SyncUnit("org1", "system1", "project1")
        sync_unit_2 = SyncUnit("org2", "system2", "project2")

        result_1 = define_paths_for_input_output_files(PARSE_JSON, sync_unit_1, 1)
        result_2 = define_paths_for_input_output_files(PARSE_JSON, sync_unit_2, 1)

        self.assertIn("org1/system1/project1", result_1.input_filepath)
        self.assertIn("org2/system2/project2", result_2.input_filepath)

        self.assertNotEqual(result_1.input_filepath, result_2.input_filepath)
        self.assertNotEqual(result_1.output_filepaths[0], result_2.output_filepaths[0])

    def test_invalid_action_returns_value_error(self):
        """Test that an invalid action returns ValueError"""
        result = define_paths_for_input_output_files(
            "invalid_action",
            self.test_sync_unit,
            self.test_chunk_number
        )

        self.assertIsInstance(result, ValueError)
        self.assertIn("invalid_action", str(result))
        self.assertIn("not mapped", str(result))


class TestBuildFullAppFlowGraph(unittest.TestCase):
    """Test the build_full_app_flow_graph function"""

    def setUp(self):
        """Set up test fixtures"""
        self.graph = build_full_app_flow_graph()

    def test_graph_is_directed(self):
        """Test that the returned graph is a directed graph"""
        self.assertIsInstance(self.graph, nx.DiGraph)

    def test_graph_is_dag(self):
        """Test that the graph is a Directed Acyclic Graph (DAG)"""
        self.assertTrue(nx.is_directed_acyclic_graph(self.graph))

    def test_graph_has_correct_number_of_nodes(self):
        """Test that the graph has the expected number of nodes"""
        # Expected nodes:
        # 1. work_{int}.json (INITIAL_DATA_FILE_NAME_FORMAT)
        # 2. parsed_items_{int}.pkl (PARSED_DATA_FILE_NAME_FORMAT)
        # 3. cleaned_items_{int}.pkl (CLEANED_ITEMS_FILE_NAME_FORMAT)
        # 4. created_time_delta_{int}.pkl (CREATED_TIME_DELTA_FILE_NAME_FORMAT)
        # 5. modified_time_delta_{int}.pkl (MODIFIED_TIME_DELTA_FILE_NAME_FORMAT)
        # 6. merged_dataset_{int}.pkl (MERGED_DATASET_FILE_NAME_FORMAT)
        # 7. merged_time_delta_{int}.pkl (MERGED_TIME_DELTA_FILE_NAME_FORMAT)
        # 8. merged_events_per_minute_{int}.pkl (MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT)
        # 9. analysis_per_record_{int}.pkl (ANALYSIS_PER_RECORD_FILE_NAME_FORMAT)
        expected_node_count = 9
        self.assertEqual(self.graph.number_of_nodes(), expected_node_count)

    def test_graph_has_correct_number_of_edges(self):
        """Test that the graph has the expected number of edges"""
        # Expected edges:
        # 1. work_{int}.json -> parsed_items_{int}.pkl (PARSE_JSON)
        # 2. parsed_items_{int}.pkl -> cleaned_items_{int}.pkl (CLEANUP_DATA)
        # 3. cleaned_items_{int}.pkl -> created_time_delta_{int}.pkl (CALCULATE_ORIGINAL_TIME_DELTAS)
        # 4. cleaned_items_{int}.pkl -> modified_time_delta_{int}.pkl (CALCULATE_ORIGINAL_TIME_DELTAS)
        # 5. cleaned_items_{int}.pkl -> merged_dataset_{int}.pkl (CREATE_MERGED_DATASET)
        # 6. merged_dataset_{int}.pkl -> merged_time_delta_{int}.pkl (CALCULATE_MERGED_TIME_DELTAS)
        # 7. merged_dataset_{int}.pkl -> merged_events_per_minute_{int}.pkl (CALCULATE_MERGED_EVENTS_PER_MINUTE)
        # 8. cleaned_items_{int}.pkl -> analysis_per_record_{int}.pkl (ANALYSIS_PER_RECORD)
        expected_edge_count = 8
        self.assertEqual(self.graph.number_of_edges(), expected_edge_count)

    def test_initial_data_file_is_in_graph(self):
        """Test that the initial data file is a node in the graph"""
        self.assertIn(INITIAL_DATA_FILE_NAME_FORMAT, self.graph.nodes())

    def test_all_expected_nodes_present(self):
        """Test that all expected file nodes are present in the graph"""
        expected_nodes = [
            INITIAL_DATA_FILE_NAME_FORMAT,
            PARSED_DATA_FILE_NAME_FORMAT,
            CLEANED_ITEMS_FILE_NAME_FORMAT,
            CREATED_TIME_DELTA_FILE_NAME_FORMAT,
            MODIFIED_TIME_DELTA_FILE_NAME_FORMAT,
            MERGED_DATASET_FILE_NAME_FORMAT,
            MERGED_TIME_DELTA_FILE_NAME_FORMAT
        ]

        for node in expected_nodes:
            with self.subTest(node=node):
                self.assertIn(node, self.graph.nodes())

    def test_parse_json_edge_exists(self):
        """Test that the PARSE_JSON edge exists from initial to parsed data"""
        self.assertTrue(
            self.graph.has_edge(
                INITIAL_DATA_FILE_NAME_FORMAT,
                PARSED_DATA_FILE_NAME_FORMAT
            )
        )

        # Check edge has correct action attribute
        edge_data = self.graph.get_edge_data(
            INITIAL_DATA_FILE_NAME_FORMAT,
            PARSED_DATA_FILE_NAME_FORMAT
        )
        self.assertEqual(edge_data['action'], PARSE_JSON)

    def test_cleanup_data_edge_exists(self):
        """Test that the CLEANUP_DATA edge exists from parsed to cleaned data"""
        self.assertTrue(
            self.graph.has_edge(
                PARSED_DATA_FILE_NAME_FORMAT,
                CLEANED_ITEMS_FILE_NAME_FORMAT
            )
        )

        # Check edge has correct action attribute
        edge_data = self.graph.get_edge_data(
            PARSED_DATA_FILE_NAME_FORMAT,
            CLEANED_ITEMS_FILE_NAME_FORMAT
        )
        self.assertEqual(edge_data['action'], CLEANUP_DATA)

    def test_calculate_original_time_deltas_edges_exist(self):
        """Test that CALCULATE_ORIGINAL_TIME_DELTAS creates two edges"""
        # Edge to created_time_delta
        self.assertTrue(
            self.graph.has_edge(
                CLEANED_ITEMS_FILE_NAME_FORMAT,
                CREATED_TIME_DELTA_FILE_NAME_FORMAT
            )
        )

        # Edge to modified_time_delta
        self.assertTrue(
            self.graph.has_edge(
                CLEANED_ITEMS_FILE_NAME_FORMAT,
                MODIFIED_TIME_DELTA_FILE_NAME_FORMAT
            )
        )

        # Check both edges have correct action attribute
        edge_data_1 = self.graph.get_edge_data(
            CLEANED_ITEMS_FILE_NAME_FORMAT,
            CREATED_TIME_DELTA_FILE_NAME_FORMAT
        )
        self.assertEqual(edge_data_1['action'], CALCULATE_ORIGINAL_TIME_DELTAS)

        edge_data_2 = self.graph.get_edge_data(
            CLEANED_ITEMS_FILE_NAME_FORMAT,
            MODIFIED_TIME_DELTA_FILE_NAME_FORMAT
        )
        self.assertEqual(edge_data_2['action'], CALCULATE_ORIGINAL_TIME_DELTAS)

    def test_create_merged_dataset_edge_exists(self):
        """Test that the CREATE_MERGED_DATASET edge exists"""
        self.assertTrue(
            self.graph.has_edge(
                CLEANED_ITEMS_FILE_NAME_FORMAT,
                MERGED_DATASET_FILE_NAME_FORMAT
            )
        )

        # Check edge has correct action attribute
        edge_data = self.graph.get_edge_data(
            CLEANED_ITEMS_FILE_NAME_FORMAT,
            MERGED_DATASET_FILE_NAME_FORMAT
        )
        self.assertEqual(edge_data['action'], CREATE_MERGED_DATASET)

    def test_calculate_merged_time_deltas_edge_exists(self):
        """Test that the CALCULATE_MERGED_TIME_DELTAS edge exists"""
        self.assertTrue(
            self.graph.has_edge(
                MERGED_DATASET_FILE_NAME_FORMAT,
                MERGED_TIME_DELTA_FILE_NAME_FORMAT
            )
        )

        # Check edge has correct action attribute
        edge_data = self.graph.get_edge_data(
            MERGED_DATASET_FILE_NAME_FORMAT,
            MERGED_TIME_DELTA_FILE_NAME_FORMAT
        )
        self.assertEqual(edge_data['action'], CALCULATE_MERGED_TIME_DELTAS)

    def test_initial_file_has_no_predecessors(self):
        """Test that the initial data file has no incoming edges"""
        predecessors = list(self.graph.predecessors(INITIAL_DATA_FILE_NAME_FORMAT))
        self.assertEqual(len(predecessors), 0)

    def test_initial_file_has_successors(self):
        """Test that the initial data file has outgoing edges"""
        successors = list(self.graph.successors(INITIAL_DATA_FILE_NAME_FORMAT))
        self.assertGreater(len(successors), 0)
        self.assertIn(PARSED_DATA_FILE_NAME_FORMAT, successors)

    def test_cleaned_items_has_multiple_successors(self):
        """Test that cleaned_items file has multiple outgoing edges"""
        successors = list(self.graph.successors(CLEANED_ITEMS_FILE_NAME_FORMAT))
        # Should have 4 successors: created_time_delta, modified_time_delta, merged_dataset, analysis_per_record
        self.assertEqual(len(successors), 4)
        self.assertIn(CREATED_TIME_DELTA_FILE_NAME_FORMAT, successors)
        self.assertIn(MODIFIED_TIME_DELTA_FILE_NAME_FORMAT, successors)
        self.assertIn(MERGED_DATASET_FILE_NAME_FORMAT, successors)
        self.assertIn(ANALYSIS_PER_RECORD_FILE_NAME_FORMAT, successors)

    def test_leaf_nodes_have_no_successors(self):
        """Test that leaf nodes (final outputs) have no outgoing edges"""
        leaf_nodes = [
            CREATED_TIME_DELTA_FILE_NAME_FORMAT,
            MODIFIED_TIME_DELTA_FILE_NAME_FORMAT,
            MERGED_TIME_DELTA_FILE_NAME_FORMAT,
            ANALYSIS_PER_RECORD_FILE_NAME_FORMAT
        ]

        for node in leaf_nodes:
            with self.subTest(node=node):
                successors = list(self.graph.successors(node))
                self.assertEqual(len(successors), 0)

    def test_all_edges_have_action_attribute(self):
        """Test that all edges have an 'action' attribute"""
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            with self.subTest(edge=f"{source} -> {target}"):
                self.assertIn('action', data)
                self.assertIsNotNone(data['action'])
                self.assertIsInstance(data['action'], str)

    def test_topological_sort_possible(self):
        """Test that a topological sort can be performed on the graph"""
        # This should not raise an exception for a DAG
        try:
            topo_order = list(nx.topological_sort(self.graph))
            # Initial file should be first in topological order
            self.assertEqual(topo_order[0], INITIAL_DATA_FILE_NAME_FORMAT)
        except nx.NetworkXError:
            self.fail("Graph is not a DAG - topological sort failed")

    def test_graph_connectivity(self):
        """Test that all nodes are reachable from the initial file"""
        # Convert to undirected for connectivity check
        undirected = self.graph.to_undirected()
        self.assertTrue(nx.is_connected(undirected))

    def test_no_self_loops(self):
        """Test that the graph has no self-loops"""
        self.assertEqual(nx.number_of_selfloops(self.graph), 0)


class TestExtractFileNameFormat(unittest.TestCase):
    """Test the extract_file_name_format function"""

    def test_extract_from_full_path(self):
        """Test extracting format from a full file path"""
        path = "data/ecosphere/zendesk/ecosphere_zendesk/data_cleanup/parsed_items_1.pkl"
        result = extract_file_name_format(path)
        self.assertEqual(result, PARSED_DATA_FILE_NAME_FORMAT)

    def test_extract_from_filename_only(self):
        """Test extracting format from just a filename"""
        result = extract_file_name_format("work_5.json")
        self.assertEqual(result, INITIAL_DATA_FILE_NAME_FORMAT)

    def test_extract_parsed_items(self):
        """Test extracting parsed_items format"""
        result = extract_file_name_format("parsed_items_10.pkl")
        self.assertEqual(result, PARSED_DATA_FILE_NAME_FORMAT)

    def test_extract_cleaned_items(self):
        """Test extracting cleaned_items format"""
        result = extract_file_name_format("cleaned_items_42.pkl")
        self.assertEqual(result, CLEANED_ITEMS_FILE_NAME_FORMAT)

    def test_extract_created_time_delta(self):
        """Test extracting created_time_delta format"""
        result = extract_file_name_format("created_time_delta_7.pkl")
        self.assertEqual(result, CREATED_TIME_DELTA_FILE_NAME_FORMAT)

    def test_extract_modified_time_delta(self):
        """Test extracting modified_time_delta format"""
        result = extract_file_name_format("modified_time_delta_3.pkl")
        self.assertEqual(result, MODIFIED_TIME_DELTA_FILE_NAME_FORMAT)

    def test_extract_merged_dataset(self):
        """Test extracting merged_dataset format"""
        result = extract_file_name_format("merged_dataset_99.pkl")
        self.assertEqual(result, MERGED_DATASET_FILE_NAME_FORMAT)

    def test_extract_merged_time_delta(self):
        """Test extracting merged_time_delta format"""
        result = extract_file_name_format("merged_time_delta_15.pkl")
        self.assertEqual(result, MERGED_TIME_DELTA_FILE_NAME_FORMAT)

    def test_extract_with_different_chunk_numbers(self):
        """Test that different chunk numbers are recognized"""
        test_cases = [
            ("work_1.json", INITIAL_DATA_FILE_NAME_FORMAT),
            ("work_100.json", INITIAL_DATA_FILE_NAME_FORMAT),
            ("work_9999.json", INITIAL_DATA_FILE_NAME_FORMAT),
        ]

        for filename, expected_format in test_cases:
            with self.subTest(filename=filename):
                result = extract_file_name_format(filename)
                self.assertEqual(result, expected_format)

    def test_unrecognized_file_returns_none(self):
        """Test that unrecognized files return None"""
        test_cases = [
            "unknown_file.txt",
            "work.json",  # Missing number
            "work_abc.json",  # Non-numeric chunk
            "parsed_items.pkl",  # Missing number
            "random_data_5.pkl",  # Unknown format
        ]

        for filename in test_cases:
            with self.subTest(filename=filename):
                result = extract_file_name_format(filename)
                self.assertIsNone(result)

    def test_extract_from_complex_path(self):
        """Test extracting from a complex nested path"""
        path = "data/org/system/project/folder/subfolder/cleaned_items_25.pkl"
        result = extract_file_name_format(path)
        self.assertEqual(result, CLEANED_ITEMS_FILE_NAME_FORMAT)


class TestGetFilePipeline(unittest.TestCase):
    """Test the get_file_pipeline function"""

    def test_pipeline_for_initial_file(self):
        """Test pipeline for the initial data file"""
        pipeline = get_file_pipeline("work_1.json")

        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, nx.DiGraph)

        # Initial file should have no predecessors
        predecessors = list(pipeline.predecessors(INITIAL_DATA_FILE_NAME_FORMAT))
        self.assertEqual(len(predecessors), 0)

        # Should have the initial file in the graph
        self.assertIn(INITIAL_DATA_FILE_NAME_FORMAT, pipeline.nodes())

    def test_pipeline_for_parsed_items(self):
        """Test pipeline for parsed_items file"""
        pipeline = get_file_pipeline("parsed_items_5.pkl")

        self.assertIsNotNone(pipeline)

        # Should contain the file itself
        self.assertIn(PARSED_DATA_FILE_NAME_FORMAT, pipeline.nodes())

        # Should contain its predecessor (work file)
        self.assertIn(INITIAL_DATA_FILE_NAME_FORMAT, pipeline.nodes())

        # Should contain its successor (cleaned_items)
        self.assertIn(CLEANED_ITEMS_FILE_NAME_FORMAT, pipeline.nodes())

    def test_pipeline_for_cleaned_items(self):
        """Test pipeline for cleaned_items file (has multiple successors)"""
        pipeline = get_file_pipeline("cleaned_items_10.pkl")

        self.assertIsNotNone(pipeline)

        # Should contain the file itself
        self.assertIn(CLEANED_ITEMS_FILE_NAME_FORMAT, pipeline.nodes())

        # Should contain all predecessors
        self.assertIn(INITIAL_DATA_FILE_NAME_FORMAT, pipeline.nodes())
        self.assertIn(PARSED_DATA_FILE_NAME_FORMAT, pipeline.nodes())

        # Should contain all successors (3 direct successors)
        self.assertIn(CREATED_TIME_DELTA_FILE_NAME_FORMAT, pipeline.nodes())
        self.assertIn(MODIFIED_TIME_DELTA_FILE_NAME_FORMAT, pipeline.nodes())
        self.assertIn(MERGED_DATASET_FILE_NAME_FORMAT, pipeline.nodes())

    def test_pipeline_for_leaf_node(self):
        """Test pipeline for a leaf node (final output file)"""
        pipeline = get_file_pipeline("created_time_delta_3.pkl")

        self.assertIsNotNone(pipeline)

        # Should contain the file itself
        self.assertIn(CREATED_TIME_DELTA_FILE_NAME_FORMAT, pipeline.nodes())

        # Should contain all predecessors
        self.assertIn(INITIAL_DATA_FILE_NAME_FORMAT, pipeline.nodes())
        self.assertIn(PARSED_DATA_FILE_NAME_FORMAT, pipeline.nodes())
        self.assertIn(CLEANED_ITEMS_FILE_NAME_FORMAT, pipeline.nodes())

        # Leaf node should have no successors
        successors = list(pipeline.successors(CREATED_TIME_DELTA_FILE_NAME_FORMAT))
        self.assertEqual(len(successors), 0)

    def test_pipeline_for_merged_dataset(self):
        """Test pipeline for merged_dataset file"""
        pipeline = get_file_pipeline("merged_dataset_7.pkl")

        self.assertIsNotNone(pipeline)

        # Should contain the file itself
        self.assertIn(MERGED_DATASET_FILE_NAME_FORMAT, pipeline.nodes())

        # Should contain all predecessors
        self.assertIn(INITIAL_DATA_FILE_NAME_FORMAT, pipeline.nodes())
        self.assertIn(PARSED_DATA_FILE_NAME_FORMAT, pipeline.nodes())
        self.assertIn(CLEANED_ITEMS_FILE_NAME_FORMAT, pipeline.nodes())

        # Should contain its successor
        self.assertIn(MERGED_TIME_DELTA_FILE_NAME_FORMAT, pipeline.nodes())

    def test_pipeline_with_full_path(self):
        """Test pipeline extraction from a full file path"""
        full_path = "data/ecosphere/zendesk/ecosphere_zendesk/data_cleanup/parsed_items_1.pkl"
        pipeline = get_file_pipeline(full_path)

        self.assertIsNotNone(pipeline)
        self.assertIn(PARSED_DATA_FILE_NAME_FORMAT, pipeline.nodes())

    def test_pipeline_for_unrecognized_file(self):
        """Test that unrecognized files return None"""
        pipeline = get_file_pipeline("unknown_file.txt")
        self.assertIsNone(pipeline)

    def test_pipeline_is_dag(self):
        """Test that the returned pipeline is a DAG"""
        pipeline = get_file_pipeline("cleaned_items_5.pkl")

        self.assertIsNotNone(pipeline)
        self.assertTrue(nx.is_directed_acyclic_graph(pipeline))

    def test_pipeline_preserves_edge_attributes(self):
        """Test that the pipeline preserves action attributes on edges"""
        pipeline = get_file_pipeline("cleaned_items_5.pkl")

        self.assertIsNotNone(pipeline)

        # Check that edges have action attributes
        for edge in pipeline.edges(data=True):
            source, target, data = edge
            self.assertIn('action', data)
            self.assertIsNotNone(data['action'])

    def test_pipeline_connectivity(self):
        """Test that all nodes in the pipeline are connected"""
        pipeline = get_file_pipeline("merged_time_delta_1.pkl")

        self.assertIsNotNone(pipeline)

        # Convert to undirected for connectivity check
        undirected = pipeline.to_undirected()
        self.assertTrue(nx.is_connected(undirected))

    def test_pipeline_contains_full_graph_for_middle_file(self):
        """Test that a middle file's pipeline contains the entire graph"""
        pipeline = get_file_pipeline("cleaned_items_5.pkl")

        self.assertIsNotNone(pipeline)

        # For cleaned_items, the pipeline should contain all 8 nodes
        # (it's in the middle and has paths to all other nodes)
        self.assertEqual(pipeline.number_of_nodes(), 9)

    def test_pipeline_for_different_files_same_format(self):
        """Test that files with the same format produce the same pipeline structure"""
        pipeline_1 = get_file_pipeline("parsed_items_5.pkl")
        pipeline_2 = get_file_pipeline("parsed_items_10.pkl")

        # Both should return the same result
        self.assertIsNotNone(pipeline_1)
        self.assertIsNotNone(pipeline_2)

        # Should have the same nodes (same structure)
        self.assertEqual(
            set(pipeline_1.nodes()),
            set(pipeline_2.nodes())
        )

    def test_pipeline_different_chunk_numbers(self):
        """Test that different chunk numbers produce the same pipeline structure"""
        pipeline_1 = get_file_pipeline("work_1.json")
        pipeline_100 = get_file_pipeline("work_100.json")

        self.assertIsNotNone(pipeline_1)
        self.assertIsNotNone(pipeline_100)

        # Should have the same structure (same nodes)
        self.assertEqual(
            set(pipeline_1.nodes()),
            set(pipeline_100.nodes())
        )


class TestExtractChunkNumber(unittest.TestCase):
    """Test the extract_chunk_number function"""

    def test_extract_from_work_file(self):
        """Test extracting chunk number from work file"""
        self.assertEqual(extract_chunk_number("work_5.json"), 5)
        self.assertEqual(extract_chunk_number("work_1.json"), 1)
        self.assertEqual(extract_chunk_number("work_100.json"), 100)

    def test_extract_from_parsed_items(self):
        """Test extracting chunk number from parsed_items file"""
        self.assertEqual(extract_chunk_number("parsed_items_10.pkl"), 10)
        self.assertEqual(extract_chunk_number("parsed_items_42.pkl"), 42)

    def test_extract_from_cleaned_items(self):
        """Test extracting chunk number from cleaned_items file"""
        self.assertEqual(extract_chunk_number("cleaned_items_7.pkl"), 7)

    def test_extract_from_full_path(self):
        """Test extracting chunk number from full path"""
        path = "data/ecosphere/zendesk/ecosphere_zendesk/data_cleanup/parsed_items_15.pkl"
        self.assertEqual(extract_chunk_number(path), 15)

    def test_extract_from_time_delta_files(self):
        """Test extracting chunk number from time delta files"""
        self.assertEqual(extract_chunk_number("created_time_delta_3.pkl"), 3)
        self.assertEqual(extract_chunk_number("modified_time_delta_8.pkl"), 8)
        self.assertEqual(extract_chunk_number("merged_time_delta_12.pkl"), 12)

    def test_extract_from_merged_dataset(self):
        """Test extracting chunk number from merged dataset file"""
        self.assertEqual(extract_chunk_number("merged_dataset_20.pkl"), 20)

    def test_file_without_chunk_number(self):
        """Test that files without chunk numbers return None"""
        self.assertIsNone(extract_chunk_number("work.json"))
        self.assertIsNone(extract_chunk_number("parsed_items.pkl"))
        self.assertIsNone(extract_chunk_number("unknown_file.txt"))

    def test_file_with_non_numeric_chunk(self):
        """Test that files with non-numeric chunks return None"""
        self.assertIsNone(extract_chunk_number("work_abc.json"))

    def test_large_chunk_numbers(self):
        """Test extracting large chunk numbers"""
        self.assertEqual(extract_chunk_number("work_9999.json"), 9999)
        self.assertEqual(extract_chunk_number("parsed_items_123456.pkl"), 123456)


class TestConstructFilePathForFormat(unittest.TestCase):
    """Test the _construct_file_path_for_format function"""

    def setUp(self):
        """Set up test fixtures"""
        self.sync_unit = SyncUnit("test_org", "test_system", "test_project")

    def test_construct_initial_data_path(self):
        """Test constructing path for initial data file"""
        result = _construct_file_path_for_format(
            INITIAL_DATA_FILE_NAME_FORMAT,
            self.sync_unit,
            5
        )
        self.assertEqual(result, "test_org/test_system/test_project/initial_data/work_5.json")

    def test_construct_parsed_items_path(self):
        """Test constructing path for parsed items file"""
        result = _construct_file_path_for_format(
            PARSED_DATA_FILE_NAME_FORMAT,
            self.sync_unit,
            10
        )
        self.assertEqual(result, "test_org/test_system/test_project/data_cleanup/parsed_items_10.pkl")

    def test_construct_cleaned_items_path(self):
        """Test constructing path for cleaned items file"""
        result = _construct_file_path_for_format(
            CLEANED_ITEMS_FILE_NAME_FORMAT,
            self.sync_unit,
            7
        )
        self.assertEqual(result, "test_org/test_system/test_project/clean_data/cleaned_items_7.pkl")

    def test_construct_time_delta_paths(self):
        """Test constructing paths for time delta files"""
        result_created = _construct_file_path_for_format(
            CREATED_TIME_DELTA_FILE_NAME_FORMAT,
            self.sync_unit,
            3
        )
        self.assertEqual(
            result_created,
            "test_org/test_system/test_project/data_preprocessing/created_time_delta_3.pkl"
        )

        result_modified = _construct_file_path_for_format(
            MODIFIED_TIME_DELTA_FILE_NAME_FORMAT,
            self.sync_unit,
            3
        )
        self.assertEqual(
            result_modified,
            "test_org/test_system/test_project/data_preprocessing/modified_time_delta_3.pkl"
        )

    def test_construct_merged_dataset_path(self):
        """Test constructing path for merged dataset file"""
        result = _construct_file_path_for_format(
            MERGED_DATASET_FILE_NAME_FORMAT,
            self.sync_unit,
            15
        )
        self.assertEqual(
            result,
            "test_org/test_system/test_project/data_preprocessing/merged_dataset_15.pkl"
        )

    def test_construct_merged_time_delta_path(self):
        """Test constructing path for merged time delta file"""
        result = _construct_file_path_for_format(
            MERGED_TIME_DELTA_FILE_NAME_FORMAT,
            self.sync_unit,
            20
        )
        self.assertEqual(
            result,
            "test_org/test_system/test_project/data_preprocessing/merged_time_delta_20.pkl"
        )

    def test_unknown_file_format_raises_error(self):
        """Test that unknown file format raises ValueError"""
        with self.assertRaises(ValueError):
            _construct_file_path_for_format(
                "unknown_format_{int}.txt",
                self.sync_unit,
                5
            )

    def test_different_chunk_numbers(self):
        """Test that different chunk numbers produce different paths"""
        result_1 = _construct_file_path_for_format(
            INITIAL_DATA_FILE_NAME_FORMAT,
            self.sync_unit,
            1
        )
        result_100 = _construct_file_path_for_format(
            INITIAL_DATA_FILE_NAME_FORMAT,
            self.sync_unit,
            100
        )

        self.assertNotEqual(result_1, result_100)
        self.assertIn("work_1.json", result_1)
        self.assertIn("work_100.json", result_100)


def _create_test_pipeline_data(sync_unit):
    """Create a minimal but complete pipeline dataset for a sync unit.

    Creates chunk 1 with:
    - initial_data/work_1.json (line-delimited JSON)
    - data_cleanup/parsed_items_1.pkl + data_cleanup_file_stats.json
    - clean_data/cleaned_items_1.pkl (tracked in data_cleanup stats)
    - data_preprocessing/* pickles + data_preprocessing_file_stats.json
    """
    from storage_access import StorageAccess
    import json

    sa = StorageAccess()
    su_path = sync_unit.get_path_for_sync_unit()

    # --- initial_data/work_1.json (line-delimited JSON, not a JSON array) ---
    initial_folder = f"{su_path}/{INITIAL_DATA_FOLDER_NAME}"
    sa.create_folder(initial_folder)
    records = [
        {"id": "1", "created_date": "2024-01-01T10:00:00Z", "modified_date": "2024-01-01T11:00:00Z"},
        {"id": "2", "created_date": "2024-01-01T10:00:30Z", "modified_date": "2024-01-01T11:30:00Z"},
        {"id": "3", "created_date": "2024-01-01T10:01:00Z", "modified_date": "2024-01-01T12:00:00Z"},
        {"id": "4", "created_date": "2024-01-01T10:01:15Z", "modified_date": "2024-01-01T12:15:00Z"},
    ]
    work_content = "\n".join(json.dumps(r, indent=2) for r in records)
    work_path = f"{initial_folder}/work_1.json"
    sa.write_to_storage(work_path, work_content)

    # --- Shared timestamps ---
    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    ts = [base, base + timedelta(seconds=30), base + timedelta(minutes=1), base + timedelta(minutes=1, seconds=15)]

    # --- parsed_items_1.pkl ---
    parsed_df = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "created_date": ts,
        "modified_date": [t + timedelta(hours=1) for t in ts],
    })
    cleanup_folder = f"{su_path}/{DATA_CLEANUP_APP_FOLDER_NAME}"
    sa.create_folder(cleanup_folder)
    parsed_path = f"{cleanup_folder}/parsed_items_1.pkl"
    sa.save_dataframe_as_picklefile(parsed_df, parsed_path)

    # --- cleaned_items_1.pkl ---
    cleaned_df = parsed_df.copy()
    clean_folder = f"{su_path}/{CLEAN_DATA_FOLDER_NAME}"
    sa.create_folder(clean_folder)
    cleaned_path = f"{clean_folder}/cleaned_items_1.pkl"
    sa.save_dataframe_as_picklefile(cleaned_df, cleaned_path)

    # --- data_cleanup_file_stats.json ---
    cleanup_stats = [
        {
            "file_name": f"{su_path}/{DATA_CLEANUP_APP_FOLDER_NAME}/parsed_items_1.pkl",
            "size": 100, "last_modified": None,
            "created_date": "2024-01-01 12:00:00",
            "input_file": f"{su_path}/{INITIAL_DATA_FOLDER_NAME}/work_1.json",
            "output_file": f"{su_path}/{DATA_CLEANUP_APP_FOLDER_NAME}/parsed_items_1.pkl",
            "created_by_function": "parse_json",
        },
        {
            "file_name": f"{su_path}/{CLEAN_DATA_FOLDER_NAME}/cleaned_items_1.pkl",
            "size": 100, "last_modified": None,
            "created_date": "2024-01-01 12:01:00",
            "input_file": f"{su_path}/{DATA_CLEANUP_APP_FOLDER_NAME}/parsed_items_1.pkl",
            "output_file": f"{su_path}/{CLEAN_DATA_FOLDER_NAME}/cleaned_items_1.pkl",
            "created_by_function": "clean_data",
        },
    ]
    sa.write_json_to_storage(f"{cleanup_folder}/{DATA_CLEANUP_FILE_STATS_FILE_NAME}", cleanup_stats)

    # --- data_preprocessing pickles ---
    preproc_folder = f"{su_path}/{DATA_PREPROCESSING_APP_FOLDER_NAME}"
    sa.create_folder(preproc_folder)

    created_td_df = parsed_df.copy()
    created_td_df["created_date_delta"] = [0.0, 0.5, 1.0, 1.25]
    sa.save_dataframe_as_picklefile(created_td_df, f"{preproc_folder}/created_time_delta_1.pkl")

    modified_td_df = parsed_df.copy()
    modified_td_df["modified_date_delta"] = [0.0, 30.0, 60.0, 75.0]
    sa.save_dataframe_as_picklefile(modified_td_df, f"{preproc_folder}/modified_time_delta_1.pkl")

    merged_df = pd.DataFrame({
        "id": ["1", "2", "3", "4", "1", "2", "3", "4"],
        "events": ts + [t + timedelta(hours=1) for t in ts],
    })
    sa.save_dataframe_as_picklefile(merged_df, f"{preproc_folder}/merged_dataset_1.pkl")

    merged_td_df = merged_df.copy()
    merged_td_df["events_delta"] = [0.0, 0.5, 1.0, 1.25, 60.0, 60.5, 61.0, 61.25]
    sa.save_dataframe_as_picklefile(merged_td_df, f"{preproc_folder}/merged_time_delta_1.pkl")

    epm_df = pd.DataFrame({
        "minute": [base, base + timedelta(minutes=1)],
        "events_per_minute": [2, 2],
    })
    sa.save_dataframe_as_picklefile(epm_df, f"{preproc_folder}/merged_events_per_minute_1.pkl")

    # --- data_preprocessing_file_stats.json ---
    preproc_stats = [
        {
            "file_name": f"{preproc_folder}/created_time_delta_1.pkl",
            "size": 0, "last_modified": None, "created_date": "2024-01-01 12:02:00",
            "input_file": f"{su_path}/{CLEAN_DATA_FOLDER_NAME}/cleaned_items_1.pkl",
            "output_file": f"{preproc_folder}/created_time_delta_1.pkl",
            "created_by_function": "calculate_original_time_deltas",
        },
        {
            "file_name": f"{preproc_folder}/modified_time_delta_1.pkl",
            "size": 0, "last_modified": None, "created_date": "2024-01-01 12:02:00",
            "input_file": f"{su_path}/{CLEAN_DATA_FOLDER_NAME}/cleaned_items_1.pkl",
            "output_file": f"{preproc_folder}/modified_time_delta_1.pkl",
            "created_by_function": "calculate_original_time_deltas",
        },
        {
            "file_name": f"{preproc_folder}/merged_dataset_1.pkl",
            "size": 0, "last_modified": None, "created_date": "2024-01-01 12:03:00",
            "input_file": f"{su_path}/{CLEAN_DATA_FOLDER_NAME}/cleaned_items_1.pkl",
            "output_file": f"{preproc_folder}/merged_dataset_1.pkl",
            "created_by_function": "create_merged_dataset",
        },
        {
            "file_name": f"{preproc_folder}/merged_time_delta_1.pkl",
            "size": 0, "last_modified": None, "created_date": "2024-01-01 12:04:00",
            "input_file": f"{preproc_folder}/merged_dataset_1.pkl",
            "output_file": f"{preproc_folder}/merged_time_delta_1.pkl",
            "created_by_function": "calculate_merged_time_deltas",
        },
        {
            "file_name": f"{preproc_folder}/merged_events_per_minute_1.pkl",
            "size": 0, "last_modified": None, "created_date": "2024-01-01 12:05:00",
            "input_file": f"{preproc_folder}/merged_dataset_1.pkl",
            "output_file": f"{preproc_folder}/merged_events_per_minute_1.pkl",
            "created_by_function": "calculate_merged_events_per_minute",
        },
    ]
    sa.write_json_to_storage(
        f"{preproc_folder}/{DATA_PREPROCESSING_FILE_STATS_FILE_NAME}",
        preproc_stats,
    )


def _cleanup_test_pipeline_data(sync_unit):
    """Remove all pipeline data created by _create_test_pipeline_data."""
    from storage_access import StorageAccess
    sa = StorageAccess()
    full_path = sa.create_full_path_string(sync_unit.get_path_for_sync_unit())
    for subfolder in [
        INITIAL_DATA_FOLDER_NAME,
        DATA_CLEANUP_APP_FOLDER_NAME,
        CLEAN_DATA_FOLDER_NAME,
        DATA_PREPROCESSING_APP_FOLDER_NAME,
    ]:
        folder_path = os.path.join(full_path, subfolder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)


class TestValidatePipelineFileStats(unittest.TestCase):
    """Test the validate_pipeline_file_stats function"""

    def setUp(self):
        """Set up test fixtures with synthetic pipeline data"""
        self.sync_unit = SyncUnit("test_org", "test_system", "test_project")
        _create_test_pipeline_data(self.sync_unit)

    def tearDown(self):
        """Clean up test data"""
        _cleanup_test_pipeline_data(self.sync_unit)

    def test_validate_existing_file(self):
        """Test validation for an existing file with stats"""
        validation = validate_pipeline_file_stats("parsed_items_1.pkl", self.sync_unit)

        self.assertIsNotNone(validation)
        self.assertIn(PARSED_DATA_FILE_NAME_FORMAT, validation)

        parsed_result = validation[PARSED_DATA_FILE_NAME_FORMAT]
        self.assertTrue(parsed_result.exists)
        self.assertTrue(parsed_result.has_stats)
        self.assertIsNotNone(parsed_result.file_path)

    def test_validate_nonexistent_file(self):
        """Test validation for a file that doesn't exist"""
        # Use chunk 999 which doesn't exist
        validation = validate_pipeline_file_stats("parsed_items_999.pkl", self.sync_unit)

        self.assertIsNotNone(validation)

        # All files should not exist for chunk 999
        for file_format, result in validation.items():
            self.assertFalse(result.exists)
            self.assertIn("File does not exist", result.issues)

    def test_validate_unrecognized_file(self):
        """Test that unrecognized files return None"""
        validation = validate_pipeline_file_stats("unknown_file.txt", self.sync_unit)
        self.assertIsNone(validation)

    def test_validation_includes_all_pipeline_files(self):
        """Test that validation includes all files in the pipeline"""
        validation = validate_pipeline_file_stats("cleaned_items_1.pkl", self.sync_unit)

        self.assertIsNotNone(validation)

        # Should include all 9 file formats in the pipeline
        self.assertEqual(len(validation), 9)

        expected_formats = [
            INITIAL_DATA_FILE_NAME_FORMAT,
            PARSED_DATA_FILE_NAME_FORMAT,
            CLEANED_ITEMS_FILE_NAME_FORMAT,
            CREATED_TIME_DELTA_FILE_NAME_FORMAT,
            MODIFIED_TIME_DELTA_FILE_NAME_FORMAT,
            MERGED_DATASET_FILE_NAME_FORMAT,
            MERGED_TIME_DELTA_FILE_NAME_FORMAT,
        ]

        for fmt in expected_formats:
            self.assertIn(fmt, validation)

    def test_validation_result_structure(self):
        """Test that validation results have the expected structure"""
        validation = validate_pipeline_file_stats("work_1.json", self.sync_unit)

        self.assertIsNotNone(validation)

        work_result = validation[INITIAL_DATA_FILE_NAME_FORMAT]

        self.assertTrue(hasattr(work_result, 'file_path'))
        self.assertTrue(hasattr(work_result, 'exists'))
        self.assertTrue(hasattr(work_result, 'has_stats'))
        self.assertTrue(hasattr(work_result, 'expected_input'))
        self.assertTrue(hasattr(work_result, 'actual_input'))
        self.assertTrue(hasattr(work_result, 'input_matches'))
        self.assertTrue(hasattr(work_result, 'expected_outputs'))
        self.assertTrue(hasattr(work_result, 'actual_outputs'))
        self.assertTrue(hasattr(work_result, 'output_matches'))
        self.assertTrue(hasattr(work_result, 'issues'))

    def test_validation_checks_input_relationships(self):
        """Test that validation checks input file relationships"""
        validation = validate_pipeline_file_stats("parsed_items_1.pkl", self.sync_unit)

        self.assertIsNotNone(validation)

        parsed_result = validation[PARSED_DATA_FILE_NAME_FORMAT]

        if parsed_result.has_stats:
            self.assertIsNotNone(parsed_result.expected_input)
            self.assertIn("work_1.json", parsed_result.expected_input)

    def test_validation_checks_output_relationships(self):
        """Test that validation checks output file relationships"""
        validation = validate_pipeline_file_stats("parsed_items_1.pkl", self.sync_unit)

        self.assertIsNotNone(validation)

        parsed_result = validation[PARSED_DATA_FILE_NAME_FORMAT]

        if parsed_result.has_stats:
            self.assertIsInstance(parsed_result.expected_outputs, list)

    def test_validation_with_different_chunk_numbers(self):
        """Test validation works with different chunk numbers"""
        validation_1 = validate_pipeline_file_stats("work_1.json", self.sync_unit)
        validation_2 = validate_pipeline_file_stats("work_2.json", self.sync_unit)

        self.assertIsNotNone(validation_1)
        self.assertIsNotNone(validation_2)

        # Both should have the same structure (same file formats)
        self.assertEqual(set(validation_1.keys()), set(validation_2.keys()))

    def test_validation_issues_list(self):
        """Test that validation issues are properly tracked"""
        validation = validate_pipeline_file_stats("parsed_items_1.pkl", self.sync_unit)

        self.assertIsNotNone(validation)

        for result in validation.values():
            self.assertIsInstance(result.issues, list)


class TestCalculatePipelineMetadata(unittest.TestCase):
    """Test the calculate_pipeline_metadata function"""

    def setUp(self):
        """Set up test fixtures with synthetic pipeline data"""
        self.sync_unit = SyncUnit("test_org", "test_system", "test_project")
        _create_test_pipeline_data(self.sync_unit)

    def tearDown(self):
        """Clean up test data"""
        _cleanup_test_pipeline_data(self.sync_unit)

    def test_calculate_metadata_returns_dict(self):
        """Test that calculate_pipeline_metadata returns a dictionary"""
        metadata = calculate_pipeline_metadata(self.sync_unit)
        self.assertIsInstance(metadata, dict)

    def test_metadata_includes_chunk_numbers(self):
        """Test that metadata includes chunk numbers for each initial file"""
        metadata = calculate_pipeline_metadata(self.sync_unit)

        self.assertGreater(len(metadata), 0)

        for pipeline_data in metadata.values():
            self.assertIsInstance(pipeline_data.chunk_number, int)

    def test_metadata_includes_files_dict(self):
        """Test that metadata includes files dictionary"""
        metadata = calculate_pipeline_metadata(self.sync_unit)

        for pipeline_data in metadata.values():
            self.assertIsInstance(pipeline_data.files, dict)

    def test_metadata_includes_missing_files_list(self):
        """Test that metadata includes missing files list"""
        metadata = calculate_pipeline_metadata(self.sync_unit)

        for pipeline_data in metadata.values():
            self.assertIsInstance(pipeline_data.missing_files, list)

    def test_file_metadata_structure(self):
        """Test that file metadata has the expected structure"""
        metadata = calculate_pipeline_metadata(self.sync_unit)

        self.assertGreater(len(metadata), 0)

        first_pipeline = next(iter(metadata.values()))

        for file_format, file_meta in first_pipeline.files.items():
            self.assertTrue(hasattr(file_meta, 'file_path'))
            self.assertTrue(hasattr(file_meta, 'exists'))
            self.assertTrue(hasattr(file_meta, 'total_rows'))
            self.assertTrue(hasattr(file_meta, 'unique_ids'))
            self.assertTrue(hasattr(file_meta, 'date_columns'))
            self.assertTrue(hasattr(file_meta, 'error'))

    def test_existing_file_has_metadata(self):
        """Test that existing files have calculated metadata"""
        metadata = calculate_pipeline_metadata(self.sync_unit)

        self.assertGreater(len(metadata), 0)

        for pipeline_data in metadata.values():
            if pipeline_data.chunk_number == 1:
                parsed_format = PARSED_DATA_FILE_NAME_FORMAT
                if parsed_format in pipeline_data.files:
                    parsed_meta = pipeline_data.files[parsed_format]
                    if parsed_meta.exists:
                        self.assertIsNotNone(parsed_meta.total_rows)
                        self.assertIsInstance(parsed_meta.total_rows, int)
                        break

    def test_missing_file_has_error(self):
        """Test that missing files have error message"""
        metadata = calculate_pipeline_metadata(self.sync_unit)

        for pipeline_data in metadata.values():
            for file_meta in pipeline_data.files.values():
                if not file_meta.exists:
                    self.assertIsNotNone(file_meta.error)
                    self.assertEqual(file_meta.error, "File does not exist")

    def test_metadata_for_nonexistent_sync_unit(self):
        """Test metadata calculation for sync unit with no files"""
        empty_sync_unit = SyncUnit("nonexistent", "org", "project")
        metadata = calculate_pipeline_metadata(empty_sync_unit)

        self.assertEqual(metadata, {})


if __name__ == '__main__':
    unittest.main()
