"""
Graphical User Interface for the Orchestrator System.

This Streamlit app provides a user-friendly interface to:
1. Select a sync unit
2. Choose an app
3. Pick an action to perform
4. Execute the action and view results
5. Execute batch operations using config files
"""
import gc
import logging
import json
import os
import subprocess
from pathlib import Path

import streamlit as st
import sync_unit
from app_registry import AppRegistry
from data_cleanup.data_cleanup_app import DataCleanupApp
from data_preprocessing.data_preprocessing_app import DataPreprocessingApp
from model_training.model_training_app import ModelTrainingApp
from model_serving.model_serving_app import ModelServingApp
from model_lifecycle.model_lifecycle_app import ModelLifecycleApp
from rl_training.rl_training_app import RLTrainingApp
from sync_policy_evaluation.sync_policy_evaluation_app import SyncPolicyEvaluationApp
from pipeline_explorer import (
    PipelineExplorerApp,
    get_timeline_plots,
    can_visualize_file,
    get_required_columns,
    OutlierMethod,
    get_outlier_plots,
    can_detect_outliers,
)
from pipeline_explorer.pipeline_data import get_combined_pipeline_data, get_pipeline_page
from pipeline_explorer.mermaid_generator import generate_pipeline_mermaid
from pipeline_explorer.file_viewer import (
    load_pickle_file, get_column_info, get_page, apply_filters_and_sort, ColumnType
)
from global_vars import (
    INITIAL_DATA_FILE_NAME_FORMAT,
    DATASET_COMPARISON_FOLDER_NAME,
    FINGERPRINTS_FILE_NAME,
    PAIRWISE_DISTANCES_FILE_NAME,
    DATASET_CLUSTERS_FILE_NAME,
    RECORD_CLUSTERS_FILE_NAME,
)
from modeling.mlflow_utils import is_mlflow_available
from storage_access import StorageAccess
from dataset_comparison import DatasetComparisonApp
from dataset_comparison.fingerprint import DatasetFingerprint, compute_all_fingerprints
from dataset_comparison.comparison import (
    PairwiseComparison, compute_all_pairwise, load_delta_samples, compute_distance_matrix,
)
from dataset_comparison.clustering import DatasetClusterResult, cluster_datasets, cluster_records
from dataset_comparison import visualization as dc_viz

from orchestrator import Orchestrator
from orchestrator_message import OrchestratorMessage
import log_viewer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_registry() -> AppRegistry:
    """Initialize and populate the app registry with all available apps."""
    registry = AppRegistry()

    registry.register_app(DataCleanupApp())
    registry.register_app(DataPreprocessingApp())
    registry.register_app(PipelineExplorerApp())
    registry.register_app(ModelTrainingApp())
    registry.register_app(ModelServingApp())
    registry.register_app(ModelLifecycleApp())
    registry.register_app(RLTrainingApp())
    registry.register_app(SyncPolicyEvaluationApp())
    registry.register_app(DatasetComparisonApp())

    return registry


def execute_single_action(registry, orchestrator):
    """UI for executing a single action on one sync unit."""

    # Layout: Three columns for the selection process
    col1, col2, col3 = st.columns(3)

    # Step 1: Select Sync Unit
    with col1:
        st.subheader("1️⃣ Select Sync Unit")

        try:
            all_sync_units = sync_unit.get_sync_units()

            if not all_sync_units:
                st.warning("No sync units found in the registry.")
                return

            sync_unit_options = {
                f"{su.organization_id} / {su.external_system_id} / {su.project_id}": su
                for su in all_sync_units
            }

            selected_sync_unit_name = st.selectbox(
                "Choose a sync unit:",
                options=list(sync_unit_options.keys()),
                key="sync_unit_selector"
            )

            selected_sync_unit = sync_unit_options[selected_sync_unit_name]

            with st.expander("Sync Unit Details"):
                st.write(f"**Organization:** {selected_sync_unit.organization_id}")
                st.write(f"**External System:** {selected_sync_unit.external_system_id}")
                st.write(f"**Project:** {selected_sync_unit.project_id}")
                st.write(f"**Path:** {selected_sync_unit.get_path_for_sync_unit()}")

        except Exception as e:
            st.error(f"Error loading sync units: {str(e)}")
            return

    # Step 2: Select App
    with col2:
        st.subheader("2️⃣ Select App")

        app_names = registry.get_app_names()

        if not app_names:
            st.warning("No apps registered in the system.")
            return

        selected_app_name = st.selectbox(
            "Choose an app:",
            options=app_names,
            key="app_selector"
        )

        app = registry.get_app(selected_app_name)
        if app:
            with st.expander("App Details"):
                st.write(f"**App Name:** {app.get_name()}")
                st.write(f"**Available Actions:** {', '.join(app.get_available_actions())}")

    # Step 3: Select Action
    with col3:
        st.subheader("3️⃣ Select Action")

        if selected_app_name:
            actions = registry.get_actions_for_app(selected_app_name)

            if not actions:
                st.warning(f"No actions available for app '{selected_app_name}'.")
                return

            selected_action = st.selectbox(
                "Choose an action:",
                options=actions,
                key="action_selector"
            )
        else:
            st.info("Select an app first.")
            return

    st.markdown("---")

    # Execute button
    st.subheader("4️⃣ Execute")

    col_execute, col_status = st.columns([1, 3])

    with col_execute:
        execute_button = st.button("🚀 Execute Action", type="primary", use_container_width=True)

    if execute_button:
        with st.spinner(f"Executing {selected_action} on {selected_sync_unit.get_name()}..."):
            try:
                message = OrchestratorMessage(
                    sync_unit_obj=selected_sync_unit,
                    action=selected_action,
                    parameters={}
                )

                response = orchestrator.send_message(message)

                st.markdown("---")
                st.subheader("📊 Execution Results")

                if response.success:
                    st.success("✅ Action executed successfully!")
                else:
                    st.error("❌ Action failed!")

                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.metric("Status", "Success" if response.success else "Failed")

                with col_res2:
                    st.metric("Completed", "Yes" if response.is_done else "In Progress")

                st.write("**Message:**")
                st.info(response.message)

                if response.logs:
                    st.write("**Logs:**")
                    for log in response.logs:
                        st.text(log)

                log_path = f"./data/{selected_sync_unit.get_path_for_sync_unit()}/logs/"
                st.write(f"**Log files saved to:** `{log_path}`")

            except Exception as e:
                st.error(f"Error executing action: {str(e)}")
                logger.error("Error in GUI execution", exc_info=True)


def execute_config(registry, orchestrator):
    """UI for executing batch operations from config files."""
    st.subheader("📋 Config-Based Batch Execution")
    st.markdown("Execute actions on multiple sync units using configuration files.")

    # Config file selection
    col1, col2 = st.columns([2, 1])

    with col1:
        config_dir = Path("configs")
        if not config_dir.exists():
            st.warning("No 'configs' folder found. Creating one...")
            config_dir.mkdir(exist_ok=True)

        config_files = list(config_dir.glob("*.json"))

        if not config_files:
            st.info("No config files found in the 'configs' folder.")
            st.markdown("Create a JSON config file in the `configs/` folder. See examples below.")
        else:
            config_file_names = [f.name for f in config_files]
            selected_config = st.selectbox(
                "Select a configuration file:",
                options=config_file_names,
                key="config_selector"
            )

            if selected_config:
                config_path = config_dir / selected_config

                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    st.markdown("**Configuration Preview:**")
                    st.json(config)

                    action = config.get("action", "N/A")
                    apply_to_all = config.get("apply_to_all", False)
                    sync_units_list = config.get("sync_units", [])
                    parameters = config.get("parameters", {})
                    description = config.get("description", "No description")

                    st.markdown("**Execution Summary:**")
                    st.write(f"- **Description:** {description}")
                    st.write(f"- **Action:** `{action}`")

                    if apply_to_all:
                        all_units = sync_unit.get_sync_units()
                        st.write(f"- **Target:** All sync units ({len(all_units)} units)")
                    else:
                        st.write(f"- **Target:** {len(sync_units_list)} specific sync units")
                        with st.expander("View target sync units"):
                            for unit_name in sync_units_list:
                                st.write(f"  - {unit_name}")

                    if parameters:
                        st.write(f"- **Parameters:** {parameters}")

                    # Execution options
                    st.markdown("---")
                    col_exec1, col_exec2 = st.columns(2)

                    with col_exec1:
                        dry_run = st.checkbox("Dry Run (preview only)", value=True, key="dry_run_checkbox")

                    with col_exec2:
                        execute_config_button = st.button("🚀 Execute Config", type="primary", use_container_width=True)

                    # Execute
                    if execute_config_button:
                        st.markdown("---")
                        st.subheader("📊 Execution Results")

                        all_sync_units = sync_unit.get_sync_units()

                        if apply_to_all:
                            units_to_process = all_sync_units
                        else:
                            all_unit_names = {unit.get_name(): unit for unit in all_sync_units}
                            units_to_process = []
                            for name in sync_units_list:
                                if name in all_unit_names:
                                    units_to_process.append(all_unit_names[name])
                                else:
                                    st.warning(f"Sync unit not found: {name}")

                        if not units_to_process:
                            st.error("No valid sync units to process!")
                            return

                        if dry_run:
                            st.info("🔍 **DRY RUN MODE** - No actions will be executed")
                            for unit in units_to_process:
                                st.write(f"Would execute `{action}` on **{unit.get_name()}**")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            results_container = st.container()

                            results = []

                            for i, unit in enumerate(units_to_process):
                                status_text.text(f"Processing {i+1}/{len(units_to_process)}: {unit.get_name()}")
                                progress_bar.progress((i + 1) / len(units_to_process))

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

                                    with results_container:
                                        if response.success:
                                            st.success(f"✅ {unit.get_name()}: {response.message}")
                                        else:
                                            st.error(f"❌ {unit.get_name()}: {response.message}")

                                except Exception as e:
                                    results.append({
                                        "sync_unit": unit.get_name(),
                                        "success": False,
                                        "message": str(e)
                                    })
                                    with results_container:
                                        st.error(f"❌ {unit.get_name()}: {str(e)}")

                            # Summary
                            status_text.text("Execution complete!")
                            st.markdown("---")
                            st.subheader("📈 Summary")

                            successful = sum(1 for r in results if r["success"])
                            failed = len(results) - successful

                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                            with col_sum1:
                                st.metric("Total", len(results))
                            with col_sum2:
                                st.metric("Successful", successful)
                            with col_sum3:
                                st.metric("Failed", failed)

                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON in config file: {e}")
                except Exception as e:
                    st.error(f"Error loading config: {e}")

    with col2:
        st.markdown("**Quick Actions**")

        if st.button("📁 Open configs folder", use_container_width=True):
            st.info(f"Config folder: `{config_dir.absolute()}`")

        if st.button("🔄 Refresh list", use_container_width=True):
            st.rerun()

    # Example configs
    st.markdown("---")
    st.markdown("### 📝 Example Configurations")

    with st.expander("Example 1: Specific Sync Units"):
        st.code('''
{
  "description": "Parse JSON for specific sync units",
  "action": "parse_json",
  "sync_units": [
    "novacore_zendesk_novacore_zendesk",
    "brightpath_servicenow_brightpath_servicenow"
  ],
  "parameters": {}
}
        ''', language="json")

    with st.expander("Example 2: Apply to All"):
        st.code('''
{
  "description": "Parse JSON for all sync units",
  "action": "parse_json",
  "apply_to_all": true,
  "parameters": {}
}
        ''', language="json")

    with st.expander("Example 3: With Parameters"):
        st.code('''
{
  "description": "Clean data with options",
  "action": "cleanup_data",
  "sync_units": ["novacore_zendesk_novacore_zendesk"],
  "parameters": {
    "remove_duplicates": true,
    "remove_empty": true
  }
}
        ''', language="json")


def view_logs():
    """UI for viewing logs from sync units and actions."""
    st.subheader("📋 Log Viewer")
    st.markdown("View and search through logs for sync units and actions.")

    # Layout: Two columns for selection
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1️⃣ Select Sync Unit")

        all_sync_units = sync_unit.get_sync_units()

        if not all_sync_units:
            st.warning("No sync units found.")
            return

        sync_unit_display_names = [
            f"{unit.organization_id} / {unit.external_system_id} / {unit.project_id}"
            for unit in all_sync_units
        ]

        selected_index = st.selectbox(
            "Choose a sync unit:",
            options=range(len(all_sync_units)),
            format_func=lambda i: sync_unit_display_names[i],
            key="log_viewer_sync_unit_selector"
        )

        selected_sync_unit = all_sync_units[selected_index]

        with st.expander("Sync Unit Details"):
            st.write(f"**Organization:** {selected_sync_unit.organization_id}")
            st.write(f"**External System:** {selected_sync_unit.external_system_id}")
            st.write(f"**Project:** {selected_sync_unit.project_id}")

    with col2:
        st.subheader("2️⃣ Select Log Type & File")

        log_files = log_viewer.get_log_files_for_sync_unit(selected_sync_unit)

        # Choose log type
        log_type = st.radio(
            "Log Type:",
            options=["processing", "orchestrator"],
            format_func=lambda x: "Processing Logs" if x == "processing" else "Orchestrator Logs",
            horizontal=True,
            key="log_type_selector"
        )

        available_logs = log_files[log_type]

        if not available_logs:
            st.info(f"No {log_type} logs found for this sync unit.")
            return

        # Action filter (only for processing logs)
        filtered_logs = available_logs
        if log_type == "processing":
            unique_actions = log_viewer.get_unique_actions_from_logs(available_logs)

            if unique_actions:
                action_filter_options = ["All Actions"] + unique_actions

                selected_action_filter = st.selectbox(
                    "Filter by Action:",
                    options=action_filter_options,
                    key="action_filter_selector"
                )

                if selected_action_filter != "All Actions":
                    filtered_logs = [
                        log for log in available_logs
                        if log.action == selected_action_filter
                    ]

                    if not filtered_logs:
                        st.warning(f"No logs found for action '{selected_action_filter}'.")
                        return

                if selected_action_filter != "All Actions":
                    st.caption(f"Showing {len(filtered_logs)} of {len(available_logs)} log files")

        log_file_options = [
            f"{log.file_name} ({log.timestamp.strftime('%Y-%m-%d %H:%M:%S') if log.timestamp else 'Unknown time'})"
            + (f" - {log.action}" if log.action else "")
            for log in filtered_logs
        ]

        selected_log_index = st.selectbox(
            "Select log file:",
            options=range(len(filtered_logs)),
            format_func=lambda i: log_file_options[i],
            key="log_file_selector"
        )

        selected_log_file = filtered_logs[selected_log_index]

        action_name = log_viewer.get_action_from_log_file(selected_log_file)
        if action_name:
            st.info(f"**Action:** {action_name}")

    st.markdown("---")

    # Filters section
    st.subheader("3️⃣ Filters & Search")

    col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])

    with col_filter1:
        search_text = st.text_input(
            "🔍 Search in logs:",
            placeholder="Enter text to search...",
            key="log_search_text"
        )

    with col_filter2:
        level_options = ["All Levels", "INFO", "WARNING", "ERROR", "DEBUG", "CRITICAL"]
        selected_level = st.selectbox(
            "Log Level:",
            options=level_options,
            key="log_level_filter"
        )
        level_filter = None if selected_level == "All Levels" else selected_level

    with col_filter3:
        max_lines = st.number_input(
            "Max lines:",
            min_value=10,
            max_value=10000,
            value=500,
            step=50,
            key="max_lines"
        )

    st.markdown("---")

    st.subheader("4️⃣ Log Content")

    with st.spinner("Loading logs..."):
        log_entries, total_matches = log_viewer.read_log_file(
            selected_log_file,
            search_text=search_text,
            level_filter=level_filter,
            max_lines=int(max_lines)
        )

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Matches", total_matches)
    with col_stat2:
        st.metric("Displayed Lines", len(log_entries))
    with col_stat3:
        if total_matches > len(log_entries):
            st.warning(f"⚠️ Showing first {len(log_entries)} of {total_matches} matches")

    if log_entries:
        # Create a container for the logs with custom styling
        log_container = st.container()

        with log_container:
            for entry in log_entries:
                # Color code by level
                if entry.level == "ERROR":
                    st.error(entry.raw_line)
                elif entry.level == "WARNING":
                    st.warning(entry.raw_line)
                elif entry.level == "INFO":
                    st.text(entry.raw_line)
                else:
                    st.text(entry.raw_line)

        # Download button for full log
        with open(selected_log_file.file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        st.download_button(
            label="📥 Download Full Log File",
            data=log_content,
            file_name=selected_log_file.file_name,
            mime="text/plain",
            key="download_log"
        )
    else:
        st.info("No log entries found matching the filters.")


def _init_pipeline_explorer_state():
    """Initialize pipeline explorer session state if not exists."""
    if 'pe_selected_sync_unit' not in st.session_state:
        st.session_state.pe_selected_sync_unit = None
    if 'pe_open_files' not in st.session_state:
        st.session_state.pe_open_files = []
    if 'pe_current_view' not in st.session_state:
        st.session_state.pe_current_view = "pipeline"
    if 'pe_cached_pipelines' not in st.session_state:
        st.session_state.pe_cached_pipelines = None
    if 'pe_pipeline_page' not in st.session_state:
        st.session_state.pe_pipeline_page = 1
    if 'pe_file_data' not in st.session_state:
        st.session_state.pe_file_data = {}
    if 'pe_file_filters' not in st.session_state:
        st.session_state.pe_file_filters = {}
    if 'pe_file_sort' not in st.session_state:
        st.session_state.pe_file_sort = {}
    if 'pe_file_page' not in st.session_state:
        st.session_state.pe_file_page = {}
    if 'pe_timeline_plots' not in st.session_state:
        st.session_state.pe_timeline_plots = {}  # Cache for timeline plots
    if 'pe_outlier_plots' not in st.session_state:
        st.session_state.pe_outlier_plots = {}  # Cache for outlier plots (key: file_path:method:threshold)


def _open_file(file_path: str, file_format: str):
    """Add a file to the open files list."""
    if file_path not in st.session_state.pe_open_files:
        st.session_state.pe_open_files.append(file_path)
        st.session_state.pe_file_page[file_path] = 1
        st.session_state.pe_file_filters[file_path] = {}
        st.session_state.pe_file_sort[file_path] = {"column": None, "ascending": True}


def _close_file(file_path: str):
    """Remove a file from the open files list and free memory."""
    if file_path in st.session_state.pe_open_files:
        st.session_state.pe_open_files.remove(file_path)
        st.session_state.pe_file_data.pop(file_path, None)
        st.session_state.pe_file_filters.pop(file_path, None)
        st.session_state.pe_file_sort.pop(file_path, None)
        st.session_state.pe_file_page.pop(file_path, None)
        st.session_state.pe_timeline_plots.pop(file_path, None)
        # Clear outlier plots for this file (keys contain file_path)
        outlier_keys_to_remove = [
            k for k in st.session_state.pe_outlier_plots if k.startswith(f"{file_path}:")
        ]
        for k in outlier_keys_to_remove:
            st.session_state.pe_outlier_plots.pop(k, None)
        if st.session_state.pe_current_view == f"file:{file_path}":
            st.session_state.pe_current_view = "pipeline"
        gc.collect()  # Force garbage collection


def _clear_all_file_cache():
    """Clear all cached file data to free memory."""
    st.session_state.pe_file_data = {}
    st.session_state.pe_timeline_plots = {}
    st.session_state.pe_outlier_plots = {}
    st.session_state.pe_file_filters = {}
    st.session_state.pe_file_sort = {}
    st.session_state.pe_file_page = {}
    gc.collect()  # Force garbage collection


def _render_mermaid(mermaid_code: str, height: int = 500):
    """Render a Mermaid diagram using HTML component."""
    html_code = f"""
    <div style="background-color: white; padding: 15px; border-radius: 8px;
                overflow: auto; min-height: {height - 50}px;">
        <pre class="mermaid" style="display: flex; justify-content: center;">
{mermaid_code}
        </pre>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{ useMaxWidth: false, htmlLabels: true }}
        }});
    </script>
    """
    st.components.v1.html(html_code, height=height, scrolling=True)


def _render_pipeline_view(pipelines):
    """Render the pipeline visualization view with pagination."""
    st.markdown("### 📊 Pipeline Overview")

    # Pipeline summary
    total_pipelines = len(pipelines)
    error_pipelines = sum(1 for p in pipelines if p.has_errors())

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total Pipelines", total_pipelines)
    with col_m2:
        st.metric("Pipelines with Issues", error_pipelines)
    with col_m3:
        st.metric("Healthy Pipelines", total_pipelines - error_pipelines)

    # Pagination
    page_size = 10
    page_pipelines, total_pages, current_page = get_pipeline_page(
        pipelines, st.session_state.pe_pipeline_page, page_size
    )

    if total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅️ Previous", disabled=current_page <= 1):
                st.session_state.pe_pipeline_page = current_page - 1
                st.rerun()
        with col_info:
            st.write(f"Page {current_page} of {total_pages}")
        with col_next:
            if st.button("➡️ Next", disabled=current_page >= total_pages):
                st.session_state.pe_pipeline_page = current_page + 1
                st.rerun()

    st.markdown("---")

    # Display each pipeline
    for pipeline in page_pipelines:
        with st.expander(f"📦 Pipeline #{pipeline.chunk_number}", expanded=True):
            # Generate and render Mermaid diagram
            mermaid_code = generate_pipeline_mermaid(pipeline)

            st.markdown("**Pipeline Flow:**")
            _render_mermaid(mermaid_code, height=550)

            # Show file list with metadata
            st.markdown("**Files in Pipeline:**")

            for file_format, file_display in pipeline.files.items():
                is_json = file_display.is_json
                has_error = file_display.has_errors()

                # Status indicator
                if not file_display.exists:
                    status = "⚪"
                    status_text = "Missing"
                elif is_json:
                    status = "🔵"
                    status_text = "JSON"
                elif has_error:
                    status = "🔴"
                    status_text = "Error"
                else:
                    status = "🟢"
                    status_text = "OK"

                file_name = (file_display.file_path.split("/")[-1]
                             if file_display.file_path else file_format)

                # File header row
                col_s, col_n, col_a = st.columns([0.5, 4, 0.5])
                with col_s:
                    st.write(status)
                with col_n:
                    st.markdown(f"**{file_name}** ({status_text})")
                with col_a:
                    if file_display.exists and not is_json and file_display.file_path:
                        btn_key = f"open_{pipeline.chunk_number}_{file_format}"
                        if st.button("📂", key=btn_key, help="Open file"):
                            _open_file(file_display.file_path, file_format)
                            st.session_state.pe_current_view = (
                                f"file:{file_display.file_path}"
                            )
                            st.rerun()

                # File metadata (indented)
                if file_display.exists and not is_json:
                    with st.container():
                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            rows = file_display.total_rows or 0
                            st.caption(f"📊 **Rows:** {rows:,}")
                        with mcol2:
                            uids = file_display.unique_ids or 0
                            st.caption(f"🆔 **Unique IDs:** {uids:,}")
                        with mcol3:
                            if file_display.date_columns:
                                # Show date range from first date column
                                dc = file_display.date_columns[0]
                                col_name = dc.get("column_name", "Date")
                                min_d = dc.get("earliest_date") or "N/A"
                                max_d = dc.get("latest_date") or "N/A"
                                st.caption(f"📅 **{col_name}:** {min_d} to {max_d}")
                            else:
                                st.caption("📅 No date columns")

                        # Show all date columns if multiple
                        if len(file_display.date_columns) > 1:
                            for dc in file_display.date_columns[1:]:
                                col_name = dc.get("column_name", "Date")
                                min_d = dc.get("earliest_date") or "N/A"
                                max_d = dc.get("latest_date") or "N/A"
                                st.caption(f"    📅 **{col_name}:** {min_d} to {max_d}")

                # Show errors/issues
                if file_display.has_errors():
                    if file_display.metadata_error:
                        st.error(f"Metadata Error: {file_display.metadata_error}")
                    for issue in file_display.validation_issues:
                        st.warning(f"Validation Issue: {issue}")

                st.markdown("---")

            # Show missing files
            if pipeline.missing_files:
                st.markdown("**Missing Files:**")
                for missing in pipeline.missing_files:
                    st.caption(f"⚪ {missing}")


def _render_file_view(file_path: str):
    """Render the file exploration view with table, filtering, and sorting."""
    file_name = file_path.split("/")[-1]
    st.markdown(f"### 📄 {file_name}")

    # Sub-tabs for different views
    tab_table, tab_timeline, tab_outliers = st.tabs([
        "📊 Table View", "📈 Timeline Viewer", "🔴 Outlier Detection"
    ])

    # Load file data with memory optimization:
    # - Initially load only visualization columns (for timeline/outlier tabs)
    # - Full DataFrame loaded lazily only when Table View is accessed
    if file_path not in st.session_state.pe_file_data:
        with st.spinner(f"Loading {file_name}..."):
            try:
                # Get visualization columns first (smaller memory footprint)
                viz_cols = get_required_columns(file_path)
                full_df = load_pickle_file(file_path)
                column_info = get_column_info(full_df, file_path)

                # Store only visualization columns initially
                if viz_cols:
                    available_viz_cols = [c for c in viz_cols if c in full_df.columns]
                    viz_df = full_df[available_viz_cols] if available_viz_cols else None
                else:
                    viz_df = None

                st.session_state.pe_file_data[file_path] = {
                    "df": None,  # Lazy-loaded on demand for Table View
                    "viz_df": viz_df,  # Subset for visualization
                    "column_info": column_info,
                    "file_path": file_path,  # Store path for lazy loading
                }
                # Keep full_df reference if viz_df is None (can't subset)
                if viz_df is None:
                    st.session_state.pe_file_data[file_path]["df"] = full_df
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return

    file_data = st.session_state.pe_file_data[file_path]
    column_info = file_data["column_info"]

    # Timeline Viewer - uses viz_df (memory-optimized subset)
    with tab_timeline:
        if can_visualize_file(file_path):
            st.markdown("**Interactive Timeline Visualization**")
            st.caption(
                "Use the range selector buttons or slider to zoom. "
                "Hover over points for details."
            )

            # Check if plots are cached
            if file_path in st.session_state.pe_timeline_plots:
                plots = st.session_state.pe_timeline_plots[file_path]
            else:
                # Generate and cache plots using viz_df (smaller memory footprint)
                viz_df = file_data.get("viz_df") if file_data.get("viz_df") is not None else file_data.get("df")
                if viz_df is not None:
                    with st.spinner("Generating timeline plots..."):
                        plots = get_timeline_plots(viz_df, file_path)
                        st.session_state.pe_timeline_plots[file_path] = plots
                else:
                    plots = []

            if plots:
                for plot_name, fig in plots:
                    st.plotly_chart(
                        fig, use_container_width=True,
                        key=f"plot_{file_path}_{plot_name}"
                    )
            else:
                st.warning("No visualizable columns found in this file.")
        else:
            st.info(
                "📊 **Timeline visualization not available for this file type.**\n\n"
                "Timeline plots are available for:\n"
                "- `cleaned_items_*.pkl` - Event count per minute\n"
                "- `created_time_delta_*.pkl` - Created date vs time delta\n"
                "- `modified_time_delta_*.pkl` - Modified date vs time delta\n"
                "- `merged_time_delta_*.pkl` - Events vs time delta"
            )

    # Outlier Detection
    with tab_outliers:
        if can_detect_outliers(file_path):
            st.markdown("**Outlier Detection & Visualization**")

            # Method selection
            col_method, col_threshold = st.columns([2, 1])
            with col_method:
                method_names = [m.value for m in OutlierMethod]
                selected_method_name = st.selectbox(
                    "Detection Method",
                    method_names,
                    key=f"outlier_method_{file_path}",
                    help=(
                        "**Z-score**: Standard deviation based (threshold = σ)\n\n"
                        "**Modified Z-score**: MAD-based, robust to non-normal data\n\n"
                        "**IQR**: Interquartile range (threshold/2 = multiplier)\n\n"
                        "**Seasonal Decompose**: Hourly pattern residuals"
                    )
                )
                selected_method = OutlierMethod(selected_method_name)

            with col_threshold:
                default_threshold = 3.5 if selected_method == OutlierMethod.MODIFIED_ZSCORE else 3.0
                threshold = st.number_input(
                    "Threshold",
                    min_value=1.0,
                    max_value=10.0,
                    value=default_threshold,
                    step=0.5,
                    key=f"outlier_threshold_{file_path}",
                )

            # Generate and display outlier plots (with caching)
            # Uses viz_df (memory-optimized subset)
            outlier_cache_key = f"{file_path}:{selected_method.value}:{threshold}"
            if outlier_cache_key in st.session_state.pe_outlier_plots:
                plots = st.session_state.pe_outlier_plots[outlier_cache_key]
            else:
                viz_df = file_data.get("viz_df") if file_data.get("viz_df") is not None else file_data.get("df")
                if viz_df is not None:
                    with st.spinner("Detecting outliers..."):
                        plots = get_outlier_plots(
                            viz_df, file_path, selected_method, threshold
                        )
                        st.session_state.pe_outlier_plots[outlier_cache_key] = plots
                else:
                    plots = []

            if plots:
                for plot_name, fig, outlier_count, total_count in plots:
                    st.plotly_chart(
                        fig, use_container_width=True,
                        key=f"outlier_plot_{file_path}_{plot_name}"
                    )
                    pct = (outlier_count / total_count * 100) if total_count > 0 else 0
                    st.caption(
                        f"🔴 **{outlier_count:,}** outliers detected "
                        f"out of **{total_count:,}** points ({pct:.2f}%)"
                    )
            else:
                st.warning("No analyzable columns found in this file.")
        else:
            st.info(
                "🔴 **Outlier detection not available for this file type.**\n\n"
                "Outlier detection is available for:\n"
                "- `cleaned_items_*.pkl` - Event count outliers\n"
                "- `created_time_delta_*.pkl` - Time delta outliers\n"
                "- `modified_time_delta_*.pkl` - Time delta outliers\n"
                "- `merged_time_delta_*.pkl` - Time delta outliers"
            )

    # Table View (main functionality) - requires full DataFrame
    with tab_table:
        # Lazy load full DataFrame only when Table View is accessed
        if file_data.get("df") is None:
            with st.spinner("Loading full data for table view..."):
                try:
                    full_df = load_pickle_file(file_path)
                    st.session_state.pe_file_data[file_path]["df"] = full_df
                except Exception as e:
                    st.error(f"Error loading full data: {e}")
                    return

        df = st.session_state.pe_file_data[file_path]["df"]

        # File summary
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Rows", len(df))
        with col_s2:
            st.metric("Columns", len(df.columns))
        with col_s3:
            id_cols = [c for c in column_info if c.column_type == ColumnType.ID]
            if id_cols:
                st.metric("Unique IDs", df[id_cols[0].name].nunique())

        st.markdown("---")

        # Filtering UI
        st.markdown("**🔍 Filters:**")
        filters = st.session_state.pe_file_filters.get(file_path, {})
        new_filters = {}

        filter_cols = st.columns(min(4, len(column_info)))
        col_idx = 0

        for col_info in column_info:
            if col_info.column_type == ColumnType.TIME_DELTA:
                continue  # Time delta columns are sort-only

            with filter_cols[col_idx % len(filter_cols)]:
                if col_info.column_type == ColumnType.ID:
                    val = st.text_input(
                        f"🔤 {col_info.name}",
                        value=filters.get(col_info.name, {}).get("text", ""),
                        key=f"filter_{file_path}_{col_info.name}"
                    )
                    if val:
                        new_filters[col_info.name] = {"type": "text", "text": val}

                elif col_info.column_type == ColumnType.DATE:
                    st.markdown(f"📅 **{col_info.name}**")
                    # Get min/max dates from the data for defaults
                    col_data = df[col_info.name].dropna()
                    if len(col_data) > 0:
                        from datetime import date
                        min_date = col_data.min()
                        max_date = col_data.max()
                        # Convert to date objects for date_input
                        if hasattr(min_date, 'date'):
                            min_date = min_date.date()
                        if hasattr(max_date, 'date'):
                            max_date = max_date.date()

                        # Validate min_date and max_date are actual date objects
                        if not isinstance(min_date, date) or not isinstance(max_date, date):
                            st.caption("Date filter not available")
                        else:
                            # Get existing filter values or use data range
                            existing = filters.get(col_info.name, {})
                            start_val = existing.get("start")
                            end_val = existing.get("end")

                            # Ensure start_val and end_val are valid date objects
                            if start_val is not None and hasattr(start_val, 'date'):
                                start_val = start_val.date()
                            if end_val is not None and hasattr(end_val, 'date'):
                                end_val = end_val.date()

                            # Use min/max as fallback if filter values are invalid
                            if not isinstance(start_val, date):
                                start_val = min_date
                            if not isinstance(end_val, date):
                                end_val = max_date

                            # Date range picker
                            date_range = st.date_input(
                                "Date range",
                                value=(start_val, end_val),
                                min_value=min_date,
                                max_value=max_date,
                                key=f"filter_date_{file_path}_{col_info.name}",
                                label_visibility="collapsed"
                            )

                            # Store filter if user changed from default range
                            if isinstance(date_range, tuple) and len(date_range) == 2:
                                start, end = date_range
                                if start != min_date or end != max_date:
                                    new_filters[col_info.name] = {
                                        "type": "date_range",
                                        "start": start,
                                        "end": end
                                    }
                    else:
                        st.caption("No date data available")

            col_idx += 1

        st.session_state.pe_file_filters[file_path] = new_filters

        # Sorting
        st.markdown("**📊 Sort:**")
        sort_cols = st.columns(2)
        sort_state = st.session_state.pe_file_sort.get(
            file_path, {"column": None, "ascending": True}
        )

        with sort_cols[0]:
            col_names = [c.name for c in column_info]
            sort_idx = 0
            if sort_state["column"] and sort_state["column"] in col_names:
                sort_idx = col_names.index(sort_state["column"]) + 1
            sort_col = st.selectbox(
                "Sort by",
                options=["None"] + col_names,
                index=sort_idx,
                key=f"sort_col_{file_path}"
            )
        with sort_cols[1]:
            sort_asc = st.checkbox(
                "Ascending", value=sort_state["ascending"], key=f"sort_asc_{file_path}"
            )

        st.session_state.pe_file_sort[file_path] = {
            "column": None if sort_col == "None" else sort_col,
            "ascending": sort_asc
        }

        st.markdown("---")

        # Apply filters and sort
        filtered_df = df.copy()

        for col_name, filter_val in new_filters.items():
            if filter_val.get("type") == "text" and filter_val.get("text"):
                mask = filtered_df[col_name].astype(str).str.contains(
                    filter_val["text"], case=False, na=False
                )
                filtered_df = filtered_df[mask]
            elif filter_val.get("type") == "date_range":
                start = filter_val.get("start")
                end = filter_val.get("end")
                if start and end:
                    # Convert date to datetime for comparison
                    import pandas as pd
                    start_dt = pd.Timestamp(start)
                    end_dt = pd.Timestamp(end) + pd.Timedelta(days=1)
                    # Handle timezone-aware columns
                    col_data = filtered_df[col_name]
                    if col_data.dt.tz is not None:
                        start_dt = start_dt.tz_localize(col_data.dt.tz)
                        end_dt = end_dt.tz_localize(col_data.dt.tz)
                    mask = (col_data >= start_dt) & (col_data < end_dt)
                    filtered_df = filtered_df[mask]

        if sort_col != "None":
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)

        # Pagination
        page_size = 50
        total_rows = len(filtered_df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        current_page = st.session_state.pe_file_page.get(file_path, 1)
        current_page = max(1, min(current_page, total_pages))

        st.write(f"Showing {total_rows} rows (filtered from {len(df)})")

        if total_pages > 1:
            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
            with col_p1:
                if st.button("⬅️ Prev", key=f"prev_{file_path}",
                             disabled=current_page <= 1):
                    st.session_state.pe_file_page[file_path] = current_page - 1
                    st.rerun()
            with col_p2:
                st.write(f"Page {current_page} of {total_pages}")
            with col_p3:
                if st.button("➡️ Next", key=f"next_{file_path}",
                             disabled=current_page >= total_pages):
                    st.session_state.pe_file_page[file_path] = current_page + 1
                    st.rerun()

        # Display table
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered_df.iloc[start_idx:end_idx]

        st.dataframe(page_df, use_container_width=True)


def _scan_model_runs(sync_unit_obj):
    """Scan artifact directories for a sync unit and load training metadata.

    Returns a list of dicts (one per trained model), sorted newest first.
    """
    sa = StorageAccess()
    models_dir = os.path.join(
        sa.create_full_path_string(sync_unit_obj.get_path_for_sync_unit()),
        "models",
    )

    if not os.path.isdir(models_dir):
        return []

    runs = []
    for entry in os.scandir(models_dir):
        if not entry.is_dir():
            continue
        meta_path = os.path.join(entry.path, "training_metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["_artifact_dir"] = entry.path
            meta["_dir_name"] = entry.name
            runs.append(meta)
        except Exception:
            continue

    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs


def _discover_metric_names(runs):
    """Discover all unique metric names across runs, grouped by split."""
    splits: dict[str, set[str]] = {}  # e.g. {"val": {"mae", "rmse"}, ...}
    for r in runs:
        for key in r.get("metrics", {}):
            parts = key.split("_", 1)
            if len(parts) == 2:
                split, name = parts
                splits.setdefault(split, set()).add(name)
    return splits


def _fmt(v, precision=4):
    """Format a metric value for display."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def model_results_tab():
    """UI for browsing trained models and launching MLflow."""
    import pandas as pd

    st.subheader("Model Results")

    # --- MLflow section (always visible at the top) ---
    mlflow_available = is_mlflow_available()

    col_mlflow_status, col_mlflow_action = st.columns([3, 1])
    with col_mlflow_status:
        if mlflow_available:
            st.info(
                "MLflow is installed. Use the button to open the tracking UI "
                "(make sure `mlflow ui` is running, or start it from here)."
            )
        else:
            st.warning(
                "MLflow is not installed. Install with `pip install mlflow` "
                "to enable experiment tracking and the MLflow UI."
            )
    with col_mlflow_action:
        st.write("")
        if mlflow_available:
            if st.button("Open MLflow UI", use_container_width=True, type="primary"):
                try:
                    subprocess.Popen(
                        ["mlflow", "ui", "--port", "5000"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    st.success("MLflow UI started at http://localhost:5000")
                except Exception as e:
                    st.error(f"Failed to start MLflow UI: {e}")

    st.markdown("---")

    # --- Sync unit selector ---
    all_sync_units = sync_unit.get_sync_units()
    if not all_sync_units:
        st.warning("No sync units found.")
        return

    sync_unit_options = {
        f"{su.organization_id} / {su.external_system_id} / {su.project_id}": su
        for su in all_sync_units
    }

    selected_name = st.selectbox(
        "Select Sync Unit:",
        options=list(sync_unit_options.keys()),
        key="mr_sync_unit_selector",
    )
    selected_sync_unit = sync_unit_options[selected_name]

    # --- Load runs ---
    runs = _scan_model_runs(selected_sync_unit)

    if not runs:
        st.info(
            f"No trained models found for **{selected_sync_unit.get_name()}**. "
            "Train models first via the Single Action or Batch Execution tab."
        )
        return

    st.markdown(f"**{len(runs)}** trained model(s) found.")

    # --- Dynamic summary table ---
    split_metrics = _discover_metric_names(runs)

    table_rows = []
    for r in runs:
        metrics = r.get("metrics", {})
        cv = r.get("cv")
        combined = cv.get("combined_metrics", {}) if cv else {}

        row: dict = {
            "Model": r.get("model_type", "?"),
            "Template": r.get("template_name") or "-",
            "Task": r.get("task_type", "?"),
            "Train": r.get("train_indices_count", "?"),
        }

        # Single-split metrics — dynamically discovered
        for split in ("val", "test"):
            for name in sorted(split_metrics.get(split, [])):
                key = f"{split}_{name}"
                row[f"{split.title()} {name.upper()}"] = metrics.get(key)

        # CV mean columns (if any run has CV)
        if combined:
            for name in sorted(split_metrics.get("val", [])):
                cv_key = f"cv_mean_val_{name}"
                val = combined.get(cv_key)
                std_key = f"cv_std_val_{name}"
                std = combined.get(std_key)
                if val is not None and std is not None:
                    row[f"CV {name.upper()}"] = f"{val:.3f} +/- {std:.3f}"
                elif val is not None:
                    row[f"CV {name.upper()}"] = f"{val:.3f}"

        row["Trained At"] = r.get("created_at", "?")
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # --- Detail expanders ---
    st.markdown("---")
    st.markdown("### Run Details")

    for i, r in enumerate(runs):
        metrics = r.get("metrics", {})
        training_info = r.get("training_info", {})
        cv = r.get("cv")
        label = (
            f"{r.get('model_type', '?')} "
            f"({r.get('template_name') or 'no template'}) "
            f"- {r.get('created_at', '?')}"
        )

        with st.expander(label, expanded=(i == 0)):
            # --- Single-split metrics: dynamic cards ---
            st.markdown("**Single-Split Metrics**")
            metric_keys = [
                k for k in metrics if isinstance(metrics[k], (int, float))
            ]
            if metric_keys:
                cols = st.columns(min(len(metric_keys), 6))
                for idx, key in enumerate(sorted(metric_keys)):
                    with cols[idx % len(cols)]:
                        st.metric(key, _fmt(metrics[key]))

            # --- Cross-validation results ---
            if cv:
                st.markdown("---")
                st.markdown(
                    f"**Time-Series Cross-Validation** ({cv.get('n_folds', '?')} folds)"
                )

                # Combined metrics as cards
                combined = cv.get("combined_metrics", {})
                mean_keys = sorted(
                    k for k in combined if k.startswith("cv_mean_")
                )
                if mean_keys:
                    cv_cols = st.columns(min(len(mean_keys), 4))
                    for idx, mk in enumerate(mean_keys):
                        name = mk.replace("cv_mean_", "")
                        std_key = mk.replace("cv_mean_", "cv_std_")
                        std_val = combined.get(std_key)
                        display = _fmt(combined[mk])
                        delta = (
                            f"+/- {_fmt(std_val)}"
                            if std_val is not None else None
                        )
                        with cv_cols[idx % len(cv_cols)]:
                            st.metric(
                                f"CV {name}",
                                display,
                                delta=delta,
                                delta_color="off",
                            )

                # Per-fold table
                fold_metrics_list = cv.get("fold_metrics", [])
                if fold_metrics_list:
                    fold_rows = []
                    for fi, fm in enumerate(fold_metrics_list):
                        fold_row: dict = {"Fold": fi + 1}
                        for k in sorted(fm):
                            if isinstance(fm[k], (int, float)):
                                fold_row[k] = round(fm[k], 4)
                        fold_rows.append(fold_row)

                    st.dataframe(
                        pd.DataFrame(fold_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

            # --- Training info ---
            st.markdown("---")
            st.markdown("**Training Info**")
            if training_info:
                info_keys = [
                    k for k in training_info
                    if k not in ("order", "seasonal_order")
                ]
                info_cols = st.columns(min(len(info_keys), 4)) if info_keys else []
                for idx, k in enumerate(info_keys):
                    with info_cols[idx % len(info_cols)]:
                        st.write(f"{k}: `{training_info[k]}`")

            # Config
            st.markdown("**Model Config**")
            st.json(r.get("model_config", {}))

            # Splits
            st.markdown("**Data Splits**")
            split_cols = st.columns(3)
            with split_cols[0]:
                st.write(f"Train: {r.get('train_indices_count', '?')}")
            with split_cols[1]:
                st.write(f"Val: {r.get('val_indices_count', '?')}")
            with split_cols[2]:
                st.write(f"Test: {r.get('test_indices_count', '?')}")

            st.caption(f"Artifacts: `{r.get('_artifact_dir', '?')}`")


def pipeline_explorer_tab():
    """UI for exploring and validating pipelines."""
    _init_pipeline_explorer_state()

    st.subheader("🔍 Pipeline Explorer")
    st.markdown("Visualize, explore, and validate data processing pipelines.")

    # Sync unit selection
    all_sync_units = sync_unit.get_sync_units()
    if not all_sync_units:
        st.warning("No sync units found.")
        return

    sync_unit_options = {
        f"{su.organization_id} / {su.external_system_id} / {su.project_id}": su
        for su in all_sync_units
    }

    col_select, col_action = st.columns([3, 1])

    with col_select:
        selected_name = st.selectbox(
            "Select Sync Unit:",
            options=list(sync_unit_options.keys()),
            key="pe_sync_unit_selector"
        )
        selected_sync_unit = sync_unit_options[selected_name]

    with col_action:
        st.write("")
        st.write("")
        if st.button("🔄 Load/Refresh Pipeline", use_container_width=True):
            st.session_state.pe_selected_sync_unit = selected_sync_unit
            st.session_state.pe_cached_pipelines = None
            st.session_state.pe_open_files = []
            st.session_state.pe_current_view = "pipeline"
            st.session_state.pe_pipeline_page = 1

    # Check if sync unit is selected
    if st.session_state.pe_selected_sync_unit is None:
        st.info("👆 Select a sync unit and click 'Load/Refresh Pipeline' to begin.")
        return

    # Load pipelines if not cached
    if st.session_state.pe_cached_pipelines is None:
        with st.spinner("Loading pipeline data..."):
            try:
                pipelines = get_combined_pipeline_data(st.session_state.pe_selected_sync_unit)
                st.session_state.pe_cached_pipelines = pipelines
            except Exception as e:
                st.error(f"Error loading pipeline data: {e}")
                return

    pipelines = st.session_state.pe_cached_pipelines

    if not pipelines:
        st.warning("No pipelines found for this sync unit.")
        return

    st.markdown("---")

    # Search by ID placeholder
    with st.expander("🔎 Search by ID (Coming Soon)", expanded=False):
        st.info("🚧 **Search by ID - Coming Soon**")
        st.markdown("""
        This feature will allow you to:
        - Search for a specific ID across all files in the pipeline
        - See all occurrences and transformations of the ID
        - Track an item's journey through the data processing pipeline

        *Development in progress...*
        """)
        st.text_input(
            "Enter ID to search:",
            disabled=True,
            placeholder="Feature not yet implemented"
        )

    st.markdown("---")

    # Sidebar for open files + main content
    sidebar_col, main_col = st.columns([1, 4])

    with sidebar_col:
        st.markdown("### 📂 Open Files")

        if st.button("📊 Pipeline View", use_container_width=True,
                     type="primary" if st.session_state.pe_current_view == "pipeline" else "secondary"):
            st.session_state.pe_current_view = "pipeline"
            st.rerun()

        if st.session_state.pe_open_files:
            st.markdown("---")
            for file_path in st.session_state.pe_open_files:
                file_name = file_path.split("/")[-1]
                is_active = st.session_state.pe_current_view == f"file:{file_path}"

                col_name, col_close = st.columns([4, 1])
                with col_name:
                    if st.button(f"📄 {file_name}", key=f"open_{file_path}",
                                 use_container_width=True,
                                 type="primary" if is_active else "secondary"):
                        st.session_state.pe_current_view = f"file:{file_path}"
                        st.rerun()
                with col_close:
                    if st.button("✖", key=f"close_{file_path}"):
                        _close_file(file_path)
                        st.rerun()
        else:
            st.caption("Click on a file in the pipeline view to open it here.")

        # Clear cache button to free memory
        if st.session_state.pe_file_data or st.session_state.pe_timeline_plots:
            st.markdown("---")
            if st.button("🗑️ Clear Cache", use_container_width=True):
                _clear_all_file_cache()
                st.success("Cache cleared!")
                st.rerun()

    with main_col:
        if st.session_state.pe_current_view == "pipeline":
            _render_pipeline_view(pipelines)
        elif st.session_state.pe_current_view.startswith("file:"):
            file_path = st.session_state.pe_current_view[5:]
            _render_file_view(file_path)


def _init_dc_state():
    """Initialize session state for Dataset Comparison tab.

    Loads previously persisted results from disk so they survive page refreshes.
    """
    if "dc_fingerprints" not in st.session_state:
        st.session_state.dc_fingerprints = _load_dc_fingerprints()
    if "dc_comparisons" not in st.session_state:
        st.session_state.dc_comparisons = _load_dc_comparisons()
    if "dc_delta_samples" not in st.session_state:
        st.session_state.dc_delta_samples = None
    if "dc_dataset_cluster" not in st.session_state:
        st.session_state.dc_dataset_cluster = _load_dc_dataset_cluster()
    if "dc_record_cluster" not in st.session_state:
        st.session_state.dc_record_cluster = _load_dc_record_cluster()


def _dc_storage_path(file_name: str) -> str:
    """Return the storage-relative path for a dataset comparison file."""
    return f"{DATASET_COMPARISON_FOLDER_NAME}/{file_name}"


def _ensure_dc_folder() -> None:
    """Create the dataset_comparison folder in storage if it doesn't exist."""
    sa = StorageAccess()
    folder = sa.bucket_path / DATASET_COMPARISON_FOLDER_NAME
    folder.mkdir(parents=True, exist_ok=True)


def _load_dc_fingerprints():
    """Load fingerprints from disk, or return None."""
    sa = StorageAccess()
    path = _dc_storage_path(FINGERPRINTS_FILE_NAME)
    data = sa.read_json_from_storage(path)
    if data:
        return [DatasetFingerprint.from_dict(d) for d in data]
    return None


def _save_dc_fingerprints(fingerprints):
    """Persist fingerprints to disk."""
    _ensure_dc_folder()
    sa = StorageAccess()
    data = [fp.to_dict() for fp in fingerprints]
    sa.write_json_to_storage(_dc_storage_path(FINGERPRINTS_FILE_NAME), data)


def _load_dc_comparisons():
    """Load pairwise comparisons from disk, or return None."""
    sa = StorageAccess()
    path = _dc_storage_path(PAIRWISE_DISTANCES_FILE_NAME)
    data = sa.read_json_from_storage(path)
    if data:
        return [PairwiseComparison.from_dict(d) for d in data]
    return None


def _save_dc_comparisons(comparisons):
    """Persist pairwise comparisons to disk."""
    _ensure_dc_folder()
    sa = StorageAccess()
    data = [c.to_dict() for c in comparisons]
    sa.write_json_to_storage(_dc_storage_path(PAIRWISE_DISTANCES_FILE_NAME), data)


def _load_dc_dataset_cluster():
    """Load dataset cluster result from disk, or return None."""
    sa = StorageAccess()
    path = _dc_storage_path(DATASET_CLUSTERS_FILE_NAME)
    data = sa.read_json_from_storage(path)
    if data:
        return DatasetClusterResult.from_dict(data)
    return None


def _save_dc_dataset_cluster(result):
    """Persist dataset cluster result to disk."""
    _ensure_dc_folder()
    sa = StorageAccess()
    sa.write_json_to_storage(
        _dc_storage_path(DATASET_CLUSTERS_FILE_NAME), result.to_dict()
    )


def _load_dc_record_cluster():
    """Load record cluster result from disk, or return None."""
    import pickle
    sa = StorageAccess()
    path = sa.bucket_path / _dc_storage_path(RECORD_CLUSTERS_FILE_NAME)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_dc_record_cluster(result):
    """Persist record cluster result to disk as pickle."""
    import pickle
    _ensure_dc_folder()
    sa = StorageAccess()
    path = sa.bucket_path / _dc_storage_path(RECORD_CLUSTERS_FILE_NAME)
    with open(path, "wb") as f:
        pickle.dump(result, f)


def dataset_comparison_tab():
    """Dataset Comparison tab with 4 sub-tabs."""
    _init_dc_state()

    st.header("Dataset Comparison & Clustering")
    st.markdown(
        "Compare sync unit datasets, find similarities, and discover behavioral archetypes."
    )

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Fingerprints",
        "Comparison",
        "Dataset Clustering",
        "Record Clustering",
    ])

    with sub_tab1:
        _dc_fingerprints_tab()

    with sub_tab2:
        _dc_comparison_tab()

    with sub_tab3:
        _dc_dataset_clustering_tab()

    with sub_tab4:
        _dc_record_clustering_tab()


def _dc_fingerprints_tab():
    """Sub-tab 1: Fingerprints - radar charts and heatmaps."""
    if st.button("Compute Fingerprints", key="dc_compute_fp"):
        with st.spinner("Computing fingerprints for all sync units..."):
            fingerprints = compute_all_fingerprints()
            st.session_state.dc_fingerprints = fingerprints
            _save_dc_fingerprints(fingerprints)

    fingerprints = st.session_state.dc_fingerprints
    if fingerprints is None:
        st.info("Click 'Compute Fingerprints' to analyze all sync units.")
        return

    st.success(f"Computed fingerprints for {len(fingerprints)} sync units")

    # Multi-select for filtering
    all_names = [fp.sync_unit_name for fp in fingerprints]
    selected = st.multiselect(
        "Select datasets to compare",
        options=all_names,
        default=all_names,
        key="dc_fp_select",
    )
    selected_fps = [fp for fp in fingerprints if fp.sync_unit_name in selected]

    if len(selected_fps) < 2:
        st.warning("Select at least 2 datasets to compare.")
        return

    # Summary table
    st.subheader("Summary Table")
    import pandas as pd
    table_data = []
    for fp in selected_fps:
        row = {"Dataset": fp.sync_unit_name, "Org": fp.org, "System": fp.system}
        row.update({k: round(v, 2) for k, v in fp.features.items()})
        table_data.append(row)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # Radar chart
    st.subheader("Radar Chart")
    fig_radar = dc_viz.create_radar_chart(selected_fps)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Feature heatmap
    st.subheader("Feature Heatmap")
    fig_heatmap = dc_viz.create_feature_heatmap(selected_fps)
    st.plotly_chart(fig_heatmap, use_container_width=True)


def _dc_comparison_tab():
    """Sub-tab 2: Pairwise comparison metrics."""
    fingerprints = st.session_state.dc_fingerprints
    if fingerprints is None:
        st.info("Compute fingerprints first (in the Fingerprints tab).")
        return

    if st.button("Compute Pairwise Comparisons", key="dc_compute_pw"):
        with st.spinner("Computing pairwise metrics and KS-tests..."):
            all_sus = sync_unit.get_sync_units()
            delta_samples = load_delta_samples(all_sus)
            comparisons = compute_all_pairwise(fingerprints, delta_samples)
            st.session_state.dc_comparisons = comparisons
            st.session_state.dc_delta_samples = delta_samples
            _save_dc_comparisons(comparisons)

    comparisons = st.session_state.dc_comparisons
    if comparisons is None:
        st.info("Click 'Compute Pairwise Comparisons' to analyze distances between datasets.")
        return

    st.success(f"Computed {len(comparisons)} pairwise comparisons")

    # Distance matrix heatmap
    st.subheader("Distance Matrix")
    metric = st.radio(
        "Metric",
        ["euclidean_distance", "cosine_similarity", "ks_statistic", "dow_jsd", "hod_jsd"],
        horizontal=True,
        key="dc_dist_metric",
    )
    fig_dist = dc_viz.create_distance_matrix_heatmap(fingerprints, comparisons, metric=metric)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Temporal profile overlays
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hour-of-Day Profile")
        fig_hod = dc_viz.create_temporal_profile_overlay(fingerprints, "hod")
        st.plotly_chart(fig_hod, use_container_width=True)
    with col2:
        st.subheader("Day-of-Week Profile")
        fig_dow = dc_viz.create_temporal_profile_overlay(fingerprints, "dow")
        st.plotly_chart(fig_dow, use_container_width=True)

    # KS-test results table
    st.subheader("KS-Test Results")
    fig_ks = dc_viz.create_ks_test_table(comparisons)
    st.plotly_chart(fig_ks, use_container_width=True)

    # Delta distribution overlay
    delta_samples = st.session_state.dc_delta_samples
    if delta_samples:
        st.subheader("Time-Delta Distribution Overlay")
        all_names = [fp.sync_unit_name for fp in fingerprints]
        sel_pair = st.multiselect(
            "Select datasets to overlay",
            options=all_names,
            default=all_names[:2] if len(all_names) >= 2 else all_names,
            key="dc_delta_select",
        )
        if sel_pair:
            fig_delta = dc_viz.create_delta_histogram_overlay(delta_samples, sel_pair)
            st.plotly_chart(fig_delta, use_container_width=True)


def _dc_dataset_clustering_tab():
    """Sub-tab 3: Dataset-level clustering."""
    fingerprints = st.session_state.dc_fingerprints
    if fingerprints is None:
        st.info("Compute fingerprints first (in the Fingerprints tab).")
        return

    col1, col2 = st.columns(2)
    with col1:
        method = st.radio("Method", ["hierarchical", "kmeans"], key="dc_ds_method")
    with col2:
        max_k = max(2, len(fingerprints) - 1)
        n_clusters = st.slider("Number of clusters", 2, max_k, min(3, max_k), key="dc_ds_k")

    if st.button("Cluster Datasets", key="dc_cluster_ds"):
        with st.spinner("Clustering datasets..."):
            result = cluster_datasets(fingerprints, method=method, n_clusters=n_clusters)
            st.session_state.dc_dataset_cluster = result
            _save_dc_dataset_cluster(result)

    result = st.session_state.dc_dataset_cluster
    if result is None:
        st.info("Click 'Cluster Datasets' to group sync units by similarity.")
        return

    st.success(
        f"Clustered into {result.n_clusters} groups "
        f"(silhouette: {result.silhouette:.3f})"
    )

    # Dendrogram (hierarchical only)
    if result.linkage_matrix is not None:
        st.subheader("Dendrogram")
        fig_dendro = dc_viz.create_dendrogram(fingerprints, result.linkage_matrix)
        st.plotly_chart(fig_dendro, use_container_width=True)

    # 2D scatter
    st.subheader("Cluster Scatter")
    proj = st.radio(
        "Projection",
        ["pca", "tsne"] if result.tsne_2d else ["pca"],
        horizontal=True,
        key="dc_ds_proj",
    )
    fig_scatter = dc_viz.create_dataset_cluster_scatter(fingerprints, result, projection=proj)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Cluster assignments
    st.subheader("Cluster Assignments")
    fig_sil = dc_viz.create_silhouette_plot(fingerprints, result)
    st.plotly_chart(fig_sil, use_container_width=True)

    # Feature importance
    if result.feature_importances:
        st.subheader("Feature Importance (What Drives the Clustering)")
        fig_imp = dc_viz.create_feature_importance_bar(result.feature_importances)
        st.plotly_chart(fig_imp, use_container_width=True)

    # System signature analysis
    st.subheader("System Signature Analysis")
    group_by = st.radio(
        "Group by", ["system", "org"], horizontal=True, key="dc_sig_group"
    )
    fig_sig = dc_viz.create_system_signature_boxplots(fingerprints, group_by=group_by)
    st.plotly_chart(fig_sig, use_container_width=True)


def _dc_record_clustering_tab():
    """Sub-tab 4: Record-level clustering."""
    col1, col2, col3 = st.columns(3)
    with col1:
        n_clusters = st.slider("Number of clusters", 2, 20, 8, key="dc_rec_k")
    with col2:
        method = st.radio(
            "Method", ["minibatch_kmeans", "dbscan"], key="dc_rec_method"
        )
    with col3:
        max_records = st.number_input(
            "Max records", min_value=1000, max_value=100000, value=50000, step=5000,
            key="dc_rec_max",
        )

    if st.button("Cluster Records", key="dc_cluster_rec"):
        with st.spinner("Loading and clustering records across all sync units..."):
            all_sus = sync_unit.get_sync_units()
            result = cluster_records(
                all_sus, n_clusters=n_clusters, method=method, max_total=max_records
            )
            st.session_state.dc_record_cluster = result
            _save_dc_record_cluster(result)

    result = st.session_state.dc_record_cluster
    if result is None or result.record_labels is None:
        st.info("Click 'Cluster Records' to find behavioral archetypes across all datasets.")
        return

    st.success(
        f"Clustered {result.sample_size} records into {result.n_clusters} groups "
        f"(silhouette: {result.silhouette:.3f})"
    )
    if result.was_subsampled:
        st.info(f"Subsampled from {result.total_records} total records")

    # Color-by selector
    color_by = st.radio(
        "Color by",
        ["cluster_id", "org", "system", "project", "sync_unit"],
        horizontal=True,
        key="dc_rec_color",
    )

    # 2D scatter
    st.subheader("Record Cluster Scatter")
    proj = st.radio(
        "Projection",
        ["pca", "tsne"] if result.tsne_2d is not None else ["pca"],
        horizontal=True,
        key="dc_rec_proj",
    )
    fig_scatter = dc_viz.create_record_cluster_scatter(
        result, color_by=color_by, projection=proj
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Cluster composition
    st.subheader("Cluster Composition")
    fig_comp = dc_viz.create_cluster_composition_bar(result)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Archetype profiles (for kmeans only)
    if result.cluster_centers is not None:
        st.subheader("Cluster Archetypes")
        cluster_id = st.selectbox(
            "Select cluster",
            options=list(range(result.n_clusters)),
            key="dc_rec_archetype",
        )
        fig_arch = dc_viz.create_archetype_radar(result, cluster_id)
        st.plotly_chart(fig_arch, use_container_width=True)

    # Feature importance
    if result.feature_importances:
        st.subheader("Feature Importance")
        fig_imp = dc_viz.create_feature_importance_bar(
            result.feature_importances,
            title="What Drives Record Clustering",
        )
        st.plotly_chart(fig_imp, use_container_width=True)


def main():
    """Main function for the Streamlit GUI."""
    st.set_page_config(
        page_title="Orchestrator Control Panel",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 Orchestrator Control Panel")
    st.markdown("---")

    if 'registry' not in st.session_state:
        st.session_state.registry = initialize_registry()

    registry = st.session_state.registry

    orchestrator = Orchestrator(
        app_registry=registry,
        enable_sync_unit_logging=True
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Single Action",
        "📋 Batch Execution",
        "📋 Log Viewer",
        "🔍 Pipeline Explorer",
        "📊 Model Results",
        "🔬 Dataset Comparison",
    ])

    with tab1:
        execute_single_action(registry, orchestrator)

    with tab2:
        execute_config(registry, orchestrator)

    with tab3:
        view_logs()

    with tab4:
        pipeline_explorer_tab()

    with tab5:
        model_results_tab()

    with tab6:
        dataset_comparison_tab()


if __name__ == "__main__":
    main()
