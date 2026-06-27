from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np


def compare_current_pkls(
    pkl_a: str | Path,
    pkl_b: str | Path,
    *,
    metrics: tuple[str, ...] = ("pearson", "cosine", "nrmse", "max_abs_error"),
    resample: bool = True,
) -> dict[str, np.ndarray]:
    """Compare corresponding current curves from two nightrun pickle files.

    The function expects each pickle to contain 36 current curves, either as:
    - your CURBD dict format with ``data["currents_curves"][(target, source)]``
    - a numpy-like array with shape (36, n_samples)
    - a numpy-like array with shape (6, 6, n_samples)
    - a list/tuple of 36 one-dimensional curves
    - a dict whose keys are ``(target, source)`` and values are current curves

    Returns
    -------
    dict[str, np.ndarray]
        One 6x6 matrix per metric. Lower is better for error metrics
        such as ``nrmse`` and ``max_abs_error``; higher is better for
        similarity metrics such as ``pearson`` and ``cosine``.
    """

    curves_a = _load_36_curves(pkl_a)
    curves_b = _load_36_curves(pkl_b)

    metric_fns = _metric_functions()
    unknown = set(metrics) - set(metric_fns)
    if unknown:
        raise ValueError(f"Unknown metric(s): {sorted(unknown)}")

    out = {name: np.full(36, np.nan, dtype=float) for name in metrics}

    for idx, (curve_a, curve_b) in enumerate(zip(curves_a, curves_b)):
        a = _as_1d_float_array(curve_a)
        b = _as_1d_float_array(curve_b)

        if resample and a.size != b.size:
            b = _resample_to_length(b, a.size)
        elif a.size != b.size:
            raise ValueError(
                f"Curve {idx} has different lengths: {a.size} vs {b.size}. "
                "Use resample=True to compare them on the same grid."
            )

        valid = np.isfinite(a) & np.isfinite(b)
        a = a[valid]
        b = b[valid]

        if a.size < 2:
            continue

        for name in metrics:
            out[name][idx] = metric_fns[name](a, b)

    return {name: values.reshape(6, 6) for name, values in out.items()}


def _load_36_curves(path: str | Path) -> list[np.ndarray]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "currents_curves" in data:
            data = data["currents_curves"]

        if _has_target_source_keys(data):
            return [
                _as_1d_float_array(data[(target, source)])
                for target in range(6)
                for source in range(6)
            ]

        data = list(data.values())

    arr = np.asarray(data, dtype=object)

    if arr.shape[:2] == (6, 6):
        curves = [arr[i, j] for i in range(6) for j in range(6)]
    elif arr.shape[0] == 36:
        curves = [arr[i] for i in range(36)]
    else:
        raise ValueError(
            f"Could not find 36 curves in {path}. Got object with shape {arr.shape}."
        )

    return [_as_1d_float_array(curve) for curve in curves]


def _has_target_source_keys(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return all((target, source) in data for target in range(6) for source in range(6))


def _as_1d_float_array(curve: Any) -> np.ndarray:
    arr = np.asarray(curve, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Each current curve must be 1D after squeeze; got {arr.shape}.")
    return arr


def _resample_to_length(curve: np.ndarray, n_samples: int) -> np.ndarray:
    if curve.size == n_samples:
        return curve
    if curve.size < 2 or n_samples < 2:
        return np.full(n_samples, np.nan)

    old_x = np.linspace(0.0, 1.0, curve.size)
    new_x = np.linspace(0.0, 1.0, n_samples)
    return np.interp(new_x, old_x, curve)


def _metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "pearson": _pearson,
        "cosine": _cosine,
        "nrmse": _nrmse,
        "rmse": _rmse,
        "mae": _mae,
        "max_abs_error": _max_abs_error,
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    scale = np.nanmax(a) - np.nanmin(a)
    if scale == 0:
        scale = np.nanstd(a)
    if scale == 0:
        return np.nan
    return _rmse(a, b) / float(scale)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))
