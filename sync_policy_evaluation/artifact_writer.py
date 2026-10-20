"""Writes evaluation artifacts to disk.

Produces decisions.parquet, summary_metrics.json, and policy_metadata.json
in a structured directory for dashboard consumption.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def write_evaluation_artifacts(
    eval_dir: str,
    decisions_df: pd.DataFrame,
    summary_metrics: Dict[str, Any],
    policy_metadata: Dict[str, Any],
) -> None:
    """Write all evaluation artifacts to eval_dir.

    Args:
        eval_dir: Output directory (created if needed).
        decisions_df: Minute-by-minute decision DataFrame.
        summary_metrics: Aggregate metrics dict.
        policy_metadata: Policy and config metadata dict.
    """
    os.makedirs(eval_dir, exist_ok=True)

    decisions_path = os.path.join(eval_dir, "decisions.parquet")
    decisions_df.to_parquet(decisions_path, index=False)
    logger.info("Wrote %d decisions to %s", len(decisions_df), decisions_path)

    summary_path = os.path.join(eval_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    meta_path = os.path.join(eval_dir, "policy_metadata.json")
    policy_metadata.setdefault("evaluated_at", datetime.now(UTC).isoformat())
    with open(meta_path, "w") as f:
        json.dump(policy_metadata, f, indent=2, default=str)
