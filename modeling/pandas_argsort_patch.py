"""Monkey-patch for pandas nargsort bug on Python 3.14 + pandas 2.3.

Root cause: boolean indexing on a large (>270K) numpy datetime64 array
corrupts the boolean mask in-place.  In ``nargsort``, the code does::

    non_nans   = items[~mask]    # corrupts mask!
    non_nan_idx = idx[~mask]     # ~mask now wrong → empty array

Fix: pre-compute ``~mask`` once and reuse it.

Call ``apply()`` once at startup before any pandas sort operations.
"""

from __future__ import annotations

_applied = False


def apply() -> None:
    """Install the patched nargsort into pandas internals."""
    global _applied
    if _applied:
        return

    import numpy as np
    import pandas.core.arrays.base as base_mod
    import pandas.core.sorting as sorting
    from pandas.core.dtypes.missing import isna
    from pandas.core.sorting import ensure_platform_int

    _original_nargsort = sorting.nargsort

    def _patched_nargsort(items, kind="quicksort", ascending=True,
                          na_position="last", key=None, mask=None):
        # Delegate non-datetime paths to the original implementation
        if key is not None:
            return _original_nargsort(
                items, kind=kind, ascending=ascending,
                na_position=na_position, key=key, mask=mask,
            )

        from pandas.core.construction import extract_array

        # Only intercept the numpy datetime64 path that's buggy
        extracted = extract_array(items, extract_numpy=True)
        if not isinstance(extracted, np.ndarray) or extracted.dtype.kind != "M":
            return _original_nargsort(
                items, kind=kind, ascending=ascending,
                na_position=na_position, key=None, mask=mask,
            )

        # --- patched datetime64 path ---
        if mask is None:
            mask = np.asarray(isna(extracted))

        # Pre-compute ~mask ONCE to avoid corruption (boolean indexing on
        # datetime64 arrays mutates the mask buffer in Python 3.14/numpy).
        not_mask = ~mask.copy()
        idx = np.arange(len(extracted))
        non_nans = extracted[not_mask]
        non_nan_idx = idx[not_mask]
        nan_idx = idx[~not_mask]

        if not ascending:
            non_nans = non_nans[::-1]
            non_nan_idx = non_nan_idx[::-1]

        # Sort via int64 view to also sidestep any datetime argsort issues
        indexer = non_nan_idx[non_nans.view("i8").argsort(kind=kind)]

        if not ascending:
            indexer = indexer[::-1]

        if na_position == "last":
            indexer = np.concatenate([indexer, nan_idx])
        elif na_position == "first":
            indexer = np.concatenate([nan_idx, indexer])
        else:
            raise ValueError(f"invalid na_position: {na_position}")

        return ensure_platform_int(indexer)

    # Patch both the module and any cached imports
    sorting.nargsort = _patched_nargsort
    base_mod.nargsort = _patched_nargsort

    _applied = True
