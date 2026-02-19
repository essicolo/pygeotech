"""Automated classification of borehole intervals into stratigraphic units.

Uses scikit-learn clustering algorithms (K-Means, Gaussian Mixture Models,
Agglomerative Clustering) with optional stratigraphic ordering constraints.

Reference methods:

* K-Means dynamic clustering for lithology (ScienceDirect, 2021)
* Semi-supervised HMRF clustering (Wang et al., Eng. Geology 2018)
* LogTrans stratigraphic ordering (Lark et al., J. Appl. Geophys. 2003)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cluster_logs(
    boreholes: Any,
    n_units: int,
    method: str = "gmm",
    features: list[str] | None = None,
    depth_weight: float = 0.5,
    stratigraphic_order: list[str] | None = None,
    random_state: int = 42,
) -> list[str]:
    """Classify borehole intervals into stratigraphic units.

    Each interval is represented by its measured properties (e.g. N_SPT,
    water content, grain-size fractions).  Clustering is performed in
    the standardised feature space, optionally including normalised
    mid-depth to encourage depth-coherent units.

    After clustering, labels are ordered from shallowest to deepest
    by mean elevation.  If *stratigraphic_order* is provided, the
    cluster labels are mapped to the given names.

    Args:
        boreholes: A :class:`~pygeotech.stratigraphy.borehole.BoreholeSet`.
        n_units: Number of stratigraphic units to identify.
        method: Clustering algorithm — ``"kmeans"``, ``"gmm"``
            (Gaussian Mixture), or ``"agglomerative"``.
        features: Property names to use as clustering features.
            If ``None``, all numeric properties present in the first
            interval are used.
        depth_weight: Weight for the normalised mid-depth feature
            (0 = ignore depth, 1 = full weight).
        stratigraphic_order: Optional list of unit names from top to
            bottom.  Must have length *n_units*.  Clusters are mapped
            to these names by mean elevation.
        random_state: Random seed for reproducibility.

    Returns:
        List of unit names in stratigraphic order (top → bottom).
        The ``layer.unit`` attribute is set on every interval in place.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "cluster_logs requires scikit-learn.  "
            "Install with: pip install scikit-learn"
        ) from exc

    # ------------------------------------------------------------------
    # 1. Auto-detect features
    # ------------------------------------------------------------------
    if features is None:
        features = _auto_detect_features(boreholes)

    # ------------------------------------------------------------------
    # 2. Build feature matrix
    # ------------------------------------------------------------------
    X_raw, labels = boreholes.feature_matrix(features)

    # Replace NaN with column median (simple imputation)
    for col in range(X_raw.shape[1]):
        mask = np.isnan(X_raw[:, col])
        if mask.any():
            median = np.nanmedian(X_raw[:, col])
            X_raw[mask, col] = median

    # Standardise
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Add normalised mid-depth as a feature
    if depth_weight > 0:
        depths = np.array([
            boreholes[_bh_index(boreholes, bh_id)].layers[li].z_mid
            for bh_id, li in labels
        ])
        depths_norm = (depths - depths.mean()) / (depths.std() + 1e-30)
        X = np.column_stack([X, depths_norm * depth_weight])

    # ------------------------------------------------------------------
    # 3. Cluster
    # ------------------------------------------------------------------
    if method == "kmeans":
        model = KMeans(n_clusters=n_units, random_state=random_state, n_init=10)
        cluster_ids = model.fit_predict(X)
    elif method == "gmm":
        model = GaussianMixture(
            n_components=n_units, random_state=random_state, n_init=3
        )
        cluster_ids = model.fit_predict(X)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_units)
        cluster_ids = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}")

    # ------------------------------------------------------------------
    # 4. Order clusters by mean elevation (top → bottom)
    # ------------------------------------------------------------------
    depths_per_interval = np.array([
        boreholes[_bh_index(boreholes, bh_id)].layers[li].z_mid
        for bh_id, li in labels
    ])
    cluster_mean_z = {}
    for cid in range(n_units):
        mask = cluster_ids == cid
        if mask.any():
            cluster_mean_z[cid] = depths_per_interval[mask].mean()
        else:
            cluster_mean_z[cid] = -1e30

    # Sort cluster IDs by mean elevation, highest first
    ordered_cids = sorted(cluster_mean_z, key=lambda c: -cluster_mean_z[c])

    # ------------------------------------------------------------------
    # 5. Assign unit names
    # ------------------------------------------------------------------
    if stratigraphic_order is not None:
        if len(stratigraphic_order) != n_units:
            raise ValueError(
                f"stratigraphic_order has {len(stratigraphic_order)} names "
                f"but n_units={n_units}"
            )
        name_map = {cid: name for cid, name in zip(ordered_cids, stratigraphic_order)}
    else:
        name_map = {cid: f"unit_{rank}" for rank, cid in enumerate(ordered_cids)}

    unit_names_ordered = [name_map[cid] for cid in ordered_cids]

    # ------------------------------------------------------------------
    # 6. Write labels back to borehole layers
    # ------------------------------------------------------------------
    for (bh_id, li), cid in zip(labels, cluster_ids):
        bh_idx = _bh_index(boreholes, bh_id)
        boreholes[bh_idx].layers[li].unit = name_map[cid]

    return unit_names_ordered


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _auto_detect_features(boreholes: Any) -> list[str]:
    """Detect numeric property names from the first non-empty interval."""
    for bh in boreholes:
        for layer in bh.layers:
            if layer.properties:
                return sorted(layer.properties.keys())
    return []


def _bh_index(boreholes: Any, bh_id: str) -> int:
    """Find index of a borehole by ID."""
    for i, bh in enumerate(boreholes):
        if bh.id == bh_id:
            return i
    raise KeyError(f"Borehole {bh_id!r} not found")
