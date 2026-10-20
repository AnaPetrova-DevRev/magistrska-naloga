"""
Mermaid Generator Module

Generates Mermaid diagram strings for pipeline visualization.
"""
import logging
from typing import Optional

from app_flow_definitions import build_full_app_flow_graph

from .pipeline_data import PipelineDisplayData, FileDisplayData

logger = logging.getLogger(__name__)


# Node style classes for Mermaid
STYLE_OK = "fill:#10b981,stroke:#059669,color:#fff"  # Green - file exists, no errors
STYLE_ERROR = "fill:#ef4444,stroke:#dc2626,color:#fff"  # Red - has errors/issues
STYLE_MISSING = "fill:#9ca3af,stroke:#6b7280,color:#fff"  # Gray - file missing
STYLE_JSON = "fill:#3b82f6,stroke:#2563eb,color:#fff"  # Blue - JSON file (non-clickable)


def _sanitize_node_id(file_format: str) -> str:
    """
    Convert a file format to a valid Mermaid node ID.

    Args:
        file_format: The file format string (e.g., "parsed_items_{int}.pkl")

    Returns:
        Valid Mermaid node ID (alphanumeric with underscores)
    """
    # Replace special characters with underscores
    node_id = file_format.replace("{", "").replace("}", "")
    node_id = node_id.replace(".", "_").replace("-", "_")
    return node_id


def _get_display_name(file_format: str, chunk_number: int) -> str:
    """
    Get a user-friendly display name for a file format.

    Args:
        file_format: The file format string
        chunk_number: The chunk number for this pipeline

    Returns:
        Display name with chunk number substituted
    """
    return file_format.replace("{int}", str(chunk_number))


def _get_node_style(file_display: Optional[FileDisplayData]) -> str:
    """
    Determine the style for a node based on its status.

    Args:
        file_display: FileDisplayData object (may be None for missing files)

    Returns:
        Mermaid style string
    """
    if file_display is None or not file_display.exists:
        return STYLE_MISSING

    if file_display.is_json:
        return STYLE_JSON

    if file_display.has_errors():
        return STYLE_ERROR

    return STYLE_OK


def generate_pipeline_mermaid(pipeline: PipelineDisplayData) -> str:
    """
    Generate a Mermaid diagram string for a pipeline.

    The diagram shows:
    - All files in the pipeline as nodes
    - Edges showing the flow from input to output
    - Color coding: green (OK), red (errors), gray (missing), blue (JSON)

    Args:
        pipeline: PipelineDisplayData object

    Returns:
        Mermaid diagram definition string
    """
    # Get the full flow graph to understand relationships
    full_graph = build_full_app_flow_graph()

    lines = ["graph TD"]
    styles = []
    node_ids = {}

    # Create nodes for each file in the pipeline
    for file_format, file_display in pipeline.files.items():
        node_id = _sanitize_node_id(file_format)
        node_ids[file_format] = node_id

        display_name = _get_display_name(file_format, pipeline.chunk_number)
        style = _get_node_style(file_display)

        # Use different shapes for different file types
        if file_display.is_json:
            # JSON files: hexagon shape (non-clickable indicator)
            lines.append(f'    {node_id}{{{{"{display_name}"}}}}')
        elif file_display.has_errors():
            # Files with errors: diamond shape
            lines.append(f'    {node_id}{{"{display_name}"}}')
        else:
            # Normal files: rounded rectangle
            lines.append(f'    {node_id}("{display_name}")')

        styles.append(f"    style {node_id} {style}")

    # Add nodes for missing files
    for file_format in pipeline.missing_files:
        if file_format not in node_ids:
            node_id = _sanitize_node_id(file_format)
            node_ids[file_format] = node_id
            display_name = _get_display_name(file_format, pipeline.chunk_number)

            # Missing files: dashed rectangle
            lines.append(f'    {node_id}["{display_name}"]')
            styles.append(f"    style {node_id} {STYLE_MISSING}")

    # Add edges based on the flow graph
    for file_format in list(pipeline.files.keys()) + pipeline.missing_files:
        if file_format in full_graph:
            for successor in full_graph.successors(file_format):
                if successor in node_ids:
                    src_id = node_ids[file_format]
                    dst_id = node_ids[successor]

                    # Get the action name for the edge label
                    edge_data = full_graph.get_edge_data(file_format, successor)
                    action = edge_data.get("action", "") if edge_data else ""

                    if action:
                        lines.append(f'    {src_id} -->|{action}| {dst_id}')
                    else:
                        lines.append(f'    {src_id} --> {dst_id}')

    # Add all styles at the end
    lines.extend(styles)

    return "\n".join(lines)
