"""
Visualization Module for Dataset Comparison

Generates Plotly figures for:
- Fingerprint radar charts and heatmaps
- Pairwise distance matrix heatmaps
- Temporal profile overlays (hour-of-day, day-of-week)
- Dendrograms and cluster scatter plots
- Record-level clustering scatter plots and composition charts
"""
import logging
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram

from .fingerprint import DatasetFingerprint
from .comparison import PairwiseComparison
from .clustering import DatasetClusterResult, RecordClusterResult

logger = logging.getLogger(__name__)

# Color palette for sync units / clusters
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    "#FF97FF", "#FECB52",
]

# Color mapping for organizations
ORG_COLORS = {
    "brightpath": "#636EFA",
    "ecosphere": "#EF553B",
    "novacore": "#00CC96",
}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _get_color(idx: int) -> str:
    return COLORS[idx % len(COLORS)]


def _get_org_color(org: str) -> str:
    return ORG_COLORS.get(org, "#888888")


# Fingerprint visualizations


def create_radar_chart(
    fingerprints: list[DatasetFingerprint],
    feature_subset: Optional[list[str]] = None,
    max_features: int = 12,
) -> go.Figure:
    """Create an overlaid radar chart of normalized fingerprint features.

    Args:
        fingerprints: List of fingerprints to display.
        feature_subset: Specific features to show. If None, selects top by variance.
        max_features: Maximum features on the radar (too many clutters the chart).
    """
    if not fingerprints:
        return go.Figure()

    all_names = fingerprints[0].get_feature_names()

    if feature_subset is None:
        # Select features with highest variance (most discriminative)
        vectors = np.array([fp.to_feature_vector(all_names) for fp in fingerprints])
        variances = np.var(vectors, axis=0)
        top_indices = np.argsort(variances)[::-1][:max_features]
        feature_subset = [all_names[i] for i in top_indices]

    # Normalize features to [0, 1] for radar display
    vectors = np.array([
        [fp.features.get(f, 0.0) for f in feature_subset]
        for fp in fingerprints
    ])

    mins = vectors.min(axis=0)
    maxs = vectors.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (vectors - mins) / ranges

    fig = go.Figure()

    for i, fp in enumerate(fingerprints):
        values = normalized[i].tolist()
        values.append(values[0])  # Close the polygon
        categories = feature_subset + [feature_subset[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            name=fp.sync_unit_name,
            line=dict(color=_get_org_color(fp.org)),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Dataset Fingerprint Radar",
        showlegend=True,
        height=600,
    )
    return fig


def create_feature_heatmap(
    fingerprints: list[DatasetFingerprint],
    feature_subset: Optional[list[str]] = None,
) -> go.Figure:
    """Create a heatmap of fingerprint features (rows=datasets, cols=features).

    Values are standardized (z-scores) for cross-feature comparison.
    """
    if not fingerprints:
        return go.Figure()

    feature_names = feature_subset or fingerprints[0].get_feature_names()
    su_names = [fp.sync_unit_name for fp in fingerprints]

    vectors = np.array([
        [fp.features.get(f, 0.0) for f in feature_names]
        for fp in fingerprints
    ])

    # Standardize per feature
    means = vectors.mean(axis=0)
    stds = vectors.std(axis=0)
    stds[stds == 0] = 1
    z_scores = (vectors - means) / stds

    fig = go.Figure(data=go.Heatmap(
        z=z_scores,
        x=feature_names,
        y=su_names,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(vectors, 2),
        texttemplate="%{text}",
        hovertemplate="Dataset: %{y}<br>Feature: %{x}<br>Raw: %{text}<br>Z-score: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="Feature Heatmap (z-score normalized)",
        xaxis_title="Feature",
        yaxis_title="Dataset",
        height=max(400, len(fingerprints) * 50 + 200),
        xaxis=dict(tickangle=45),
    )
    return fig


# Comparison visualizations


def create_distance_matrix_heatmap(
    fingerprints: list[DatasetFingerprint],
    comparisons: list[PairwiseComparison],
    metric: str = "euclidean_distance",
) -> go.Figure:
    """Create an annotated heatmap of pairwise distances."""
    n = len(fingerprints)
    names = [fp.sync_unit_name for fp in fingerprints]
    name_to_idx = {name: i for i, name in enumerate(names)}

    matrix = np.zeros((n, n))
    for comp in comparisons:
        i = name_to_idx.get(comp.su_a)
        j = name_to_idx.get(comp.su_b)
        if i is not None and j is not None:
            val = getattr(comp, metric, 0.0)
            matrix[i, j] = val
            matrix[j, i] = val

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=names,
        y=names,
        colorscale="Viridis",
        text=np.round(matrix, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
    ))

    title_label = metric.replace("_", " ").title()
    fig.update_layout(
        title=f"Pairwise {title_label}",
        height=500,
        width=600,
    )
    return fig


def create_temporal_profile_overlay(
    fingerprints: list[DatasetFingerprint],
    profile_type: str = "hod",
) -> go.Figure:
    """Create a grouped bar chart overlaying temporal profiles.

    Args:
        fingerprints: List of fingerprints.
        profile_type: "hod" for hour-of-day (24 bars) or "dow" for day-of-week (7 bars).
    """
    if profile_type == "dow":
        categories = DAY_NAMES
        get_profile = lambda fp: fp.dow_profile if fp.dow_profile else [1/7]*7
        title = "Day-of-Week Activity Profile"
        xaxis_title = "Day of Week"
    else:
        categories = [f"{h:02d}" for h in range(24)]
        get_profile = lambda fp: fp.hod_profile if fp.hod_profile else [1/24]*24
        title = "Hour-of-Day Activity Profile"
        xaxis_title = "Hour"

    fig = go.Figure()

    for i, fp in enumerate(fingerprints):
        profile = get_profile(fp)
        fig.add_trace(go.Bar(
            x=categories,
            y=profile,
            name=fp.sync_unit_name,
            marker_color=_get_org_color(fp.org),
            opacity=0.7,
        ))

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Fraction of Activity",
        height=500,
        showlegend=True,
    )
    return fig


def create_delta_histogram_overlay(
    delta_samples: dict[str, np.ndarray],
    su_names: list[str],
    log_x: bool = True,
) -> go.Figure:
    """Create overlaid histograms of time-delta distributions."""
    fig = go.Figure()

    for i, name in enumerate(su_names):
        if name not in delta_samples:
            continue
        deltas = delta_samples[name]
        if log_x:
            deltas = deltas[deltas > 0]
            deltas = np.log10(deltas)

        fig.add_trace(go.Histogram(
            x=deltas,
            name=name,
            opacity=0.5,
            nbinsx=50,
        ))

    fig.update_layout(
        barmode="overlay",
        title="Time-Delta Distribution" + (" (log10 scale)" if log_x else ""),
        xaxis_title="log10(seconds)" if log_x else "Seconds",
        yaxis_title="Count",
        height=500,
    )
    return fig


def create_ks_test_table(comparisons: list[PairwiseComparison]) -> go.Figure:
    """Create a table figure of KS-test results sorted by statistic."""
    sorted_comps = sorted(comparisons, key=lambda c: c.ks_statistic, reverse=True)

    fig = go.Figure(data=go.Table(
        header=dict(
            values=["Dataset A", "Dataset B", "KS Statistic", "p-value", "Significant?"],
            fill_color="lightgrey",
            align="left",
        ),
        cells=dict(
            values=[
                [c.su_a for c in sorted_comps],
                [c.su_b for c in sorted_comps],
                [f"{c.ks_statistic:.4f}" for c in sorted_comps],
                [f"{c.ks_pvalue:.4e}" for c in sorted_comps],
                ["Yes" if c.ks_pvalue < 0.05 else "No" for c in sorted_comps],
            ],
            align="left",
        ),
    ))

    fig.update_layout(title="KS-Test Results (Time-Delta Distributions)", height=400)
    return fig


# Dataset clustering visualizations


def create_dendrogram(
    fingerprints: list[DatasetFingerprint],
    linkage_matrix: list,
) -> go.Figure:
    """Create a dendrogram from hierarchical clustering results."""
    labels = [fp.sync_unit_name for fp in fingerprints]
    linkage_arr = np.array(linkage_matrix)

    # Use scipy to compute dendrogram layout
    dendro = dendrogram(linkage_arr, labels=labels, no_plot=True)

    fig = go.Figure()

    # Draw the dendrogram lines
    for i in range(len(dendro["icoord"])):
        x = dendro["icoord"][i]
        y = dendro["dcoord"][i]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color="#636EFA", width=2),
            showlegend=False,
        ))

    fig.update_layout(
        title="Dataset Dendrogram (Ward Linkage)",
        xaxis=dict(
            ticktext=dendro["ivl"],
            tickvals=list(range(5, len(dendro["ivl"]) * 10, 10)),
            tickangle=45,
        ),
        yaxis_title="Distance",
        height=500,
    )
    return fig


def create_dataset_cluster_scatter(
    fingerprints: list[DatasetFingerprint],
    cluster_result: DatasetClusterResult,
    projection: str = "pca",
) -> go.Figure:
    """Create a 2D scatter plot of dataset clusters.

    Points sized by total_events (log scale), colored by cluster.
    """
    coords = cluster_result.pca_2d if projection == "pca" else cluster_result.tsne_2d
    if coords is None:
        coords = cluster_result.pca_2d

    if coords is None:
        return go.Figure()

    fig = go.Figure()

    for i, fp in enumerate(fingerprints):
        xy = coords.get(fp.sync_unit_name, [0, 0])
        cluster_id = cluster_result.labels.get(fp.sync_unit_name, 0)
        total_events = fp.features.get("total_events", 100)
        marker_size = max(10, min(40, np.log10(total_events + 1) * 8))

        fig.add_trace(go.Scatter(
            x=[xy[0]],
            y=[xy[1]],
            mode="markers+text",
            marker=dict(
                size=marker_size,
                color=_get_color(cluster_id),
                line=dict(width=1, color="black"),
            ),
            text=[fp.sync_unit_name],
            textposition="top center",
            name=f"Cluster {cluster_id}",
            showlegend=False,
            hovertemplate=(
                f"<b>{fp.sync_unit_name}</b><br>"
                f"Org: {fp.org}<br>"
                f"System: {fp.system}<br>"
                f"Cluster: {cluster_id}<br>"
                f"Events: {total_events:,.0f}<br>"
                "<extra></extra>"
            ),
        ))

    proj_label = "PCA" if projection == "pca" else "t-SNE"
    fig.update_layout(
        title=f"Dataset Clusters ({proj_label} projection)",
        xaxis_title=f"{proj_label} 1",
        yaxis_title=f"{proj_label} 2",
        height=500,
    )
    return fig


def create_silhouette_plot(
    fingerprints: list[DatasetFingerprint],
    cluster_result: DatasetClusterResult,
) -> go.Figure:
    """Create a bar chart of per-cluster silhouette information."""
    # Group by cluster
    clusters = {}
    for name, cid in cluster_result.labels.items():
        clusters.setdefault(cid, []).append(name)

    fig = go.Figure()

    names = []
    colors = []
    for cid in sorted(clusters.keys()):
        for name in clusters[cid]:
            names.append(name)
            colors.append(_get_color(cid))

    fig.add_trace(go.Bar(
        x=names,
        y=[1] * len(names),  # Uniform height, colored by cluster
        marker_color=colors,
        text=[f"Cluster {cluster_result.labels[n]}" for n in names],
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Cluster Assignments (Overall Silhouette: {cluster_result.silhouette:.3f})",
        xaxis_title="Dataset",
        yaxis=dict(visible=False),
        height=300,
    )
    return fig


def create_feature_importance_bar(
    importances: dict[str, float],
    title: str = "Feature Importance",
    max_features: int = 10,
) -> go.Figure:
    """Create a horizontal bar chart of feature importances."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:max_features]

    names = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]

    fig = go.Figure(data=go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color="#636EFA",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        height=max(300, len(names) * 30 + 100),
    )
    return fig


def create_system_signature_boxplots(
    fingerprints: list[DatasetFingerprint],
    group_by: str = "system",
    features: Optional[list[str]] = None,
    max_features: int = 6,
) -> go.Figure:
    """Create box plots of features grouped by org or system.

    Shows whether behavior is driven more by system type or organization.
    """
    if not fingerprints:
        return go.Figure()

    if features is None:
        all_names = fingerprints[0].get_feature_names()
        vectors = np.array([fp.to_feature_vector(all_names) for fp in fingerprints])
        variances = np.var(vectors, axis=0)
        top_indices = np.argsort(variances)[::-1][:max_features]
        features = [all_names[i] for i in top_indices]

    n_features = len(features)
    fig = make_subplots(
        rows=1, cols=n_features,
        subplot_titles=features,
        shared_yaxes=False,
    )

    groups = {}
    for fp in fingerprints:
        key = fp.system if group_by == "system" else fp.org
        groups.setdefault(key, []).append(fp)

    for col, feat in enumerate(features, start=1):
        for group_name, group_fps in groups.items():
            values = [fp.features.get(feat, 0.0) for fp in group_fps]
            fig.add_trace(
                go.Box(
                    y=values,
                    name=group_name,
                    marker_color=_get_org_color(group_name) if group_by == "org" else None,
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )

    fig.update_layout(
        title=f"Feature Distribution by {group_by.title()}",
        height=400,
        showlegend=True,
    )
    return fig


# Record clustering visualizations


def create_record_cluster_scatter(
    result: RecordClusterResult,
    color_by: str = "cluster_id",
    projection: str = "pca",
    max_points: int = 2000,
) -> go.Figure:
    """Create a 2D scatter plot of record clusters.

    Uses Scattergl for performance with large datasets.

    Args:
        result: RecordClusterResult with projections and labels.
        color_by: Column name to color by: "cluster_id", "org", "system", "project".
        projection: "pca" or "tsne".
        max_points: Maximum points to plot (downsample if larger).
    """
    coords = result.pca_2d if projection == "pca" else result.tsne_2d
    if coords is None:
        coords = result.pca_2d
    if coords is None or result.record_labels is None:
        return go.Figure()

    labels = result.record_labels
    n = len(labels)

    # Downsample if needed
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        idx = np.sort(idx)
        coords = coords[idx]
        labels = labels.iloc[idx].reset_index(drop=True)

    fig = go.Figure()

    color_col = color_by if color_by in labels.columns else "cluster_id"
    unique_vals = labels[color_col].unique()

    sorted_vals = sorted(unique_vals)
    for i, val in enumerate(sorted_vals):
        mask = labels[color_col] == val
        x = coords[mask, 0]
        y = coords[mask, 1]

        if color_col == "cluster_id":
            color = _get_color(int(val))
        elif color_col == "org":
            color = _get_org_color(str(val))
        else:
            color = _get_color(i)

        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.6),
            name=str(val),
        ))

    proj_label = "PCA" if projection == "pca" else "t-SNE"
    fig.update_layout(
        title=f"Record Clusters (colored by {color_col}, {proj_label})",
        xaxis_title=f"{proj_label} 1",
        yaxis_title=f"{proj_label} 2",
        height=600,
        showlegend=True,
    )
    return fig


def create_cluster_composition_bar(result: RecordClusterResult) -> go.Figure:
    """Create a stacked bar chart showing sync unit composition per cluster."""
    if result.record_labels is None:
        return go.Figure()

    labels = result.record_labels
    composition = labels.groupby(["cluster_id", "sync_unit"]).size().unstack(fill_value=0)

    fig = go.Figure()

    for su in composition.columns:
        fig.add_trace(go.Bar(
            x=[f"Cluster {c}" for c in composition.index],
            y=composition[su].values,
            name=su,
        ))

    fig.update_layout(
        barmode="stack",
        title="Cluster Composition by Sync Unit",
        xaxis_title="Cluster",
        yaxis_title="Number of Records",
        height=500,
    )
    return fig


def create_archetype_radar(
    result: RecordClusterResult,
    cluster_id: int,
    max_features: int = 10,
) -> go.Figure:
    """Create a radar chart for a single cluster's centroid (archetype profile)."""
    if result.cluster_centers is None or result.feature_names is None:
        return go.Figure()

    if cluster_id >= len(result.cluster_centers):
        return go.Figure()

    center = result.cluster_centers[cluster_id]

    # Select top features by absolute magnitude
    abs_vals = np.abs(center)
    top_idx = np.argsort(abs_vals)[::-1][:max_features]

    names = [result.feature_names[i] for i in top_idx]
    values = [float(center[i]) for i in top_idx]

    # Normalize to [0, 1] for display
    min_v = min(values)
    max_v = max(values)
    rng = max_v - min_v if max_v != min_v else 1
    norm_values = [(v - min_v) / rng for v in values]
    norm_values.append(norm_values[0])
    names.append(names[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=norm_values,
        theta=names,
        fill="toself",
        fillcolor=f"rgba({100}, {100}, {255}, 0.2)",
        line=dict(color="#636EFA"),
    ))

    fig.update_layout(
        title=f"Cluster {cluster_id} Archetype Profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500,
    )
    return fig
