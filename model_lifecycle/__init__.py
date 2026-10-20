"""Model lifecycle management application for the orchestrator.

This package provides the ModelLifecycleApp for managing model versions,
stages, and deployments via MLflow Model Registry.
"""

from model_lifecycle.model_lifecycle_app import ModelLifecycleApp

__all__ = ["ModelLifecycleApp"]
