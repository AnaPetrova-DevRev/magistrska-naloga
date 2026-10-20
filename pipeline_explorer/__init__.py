"""
Pipeline Explorer Module

This module provides functionality for visualizing, exploring, and validating
the data processing pipeline. It includes:
- File viewing with pagination, filtering, and sorting
- Pipeline metadata calculation and visualization
- Mermaid diagram generation for pipeline visualization
- Validation of pipeline file relationships
- Timeline visualization for data exploration
- Outlier detection and visualization
"""
from .pipeline_explorer_app import PipelineExplorerApp
from .timeline_viewer import (
    get_timeline_plots,
    can_visualize_file,
    get_required_columns,
    create_timeline_scatter,
    create_event_count_scatter,
)
from .outlier_detection import (
    OutlierMethod,
    get_outlier_plots,
    can_detect_outliers,
    detect_outliers,
    create_outlier_scatter,
)

__all__ = [
    'PipelineExplorerApp',
    'get_timeline_plots',
    'can_visualize_file',
    'get_required_columns',
    'create_timeline_scatter',
    'create_event_count_scatter',
    'OutlierMethod',
    'get_outlier_plots',
    'can_detect_outliers',
    'detect_outliers',
    'create_outlier_scatter',
]
