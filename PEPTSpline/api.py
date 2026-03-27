from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ._core import TrajectoryFit, fit_trajectory, resample_uniform_space, resample_uniform_time


DEFAULT_COLUMNS = ("time", "x", "y", "z", "error")
FOUR_COLUMN_DEFAULTS = ("time", "x", "y", "z")


@dataclass(slots=True)
class FitResult:
    """Small public wrapper around the internal trajectory fit."""

    _fit: TrajectoryFit

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._fit.diagnostics

    @property
    def settings(self):
        return self._fit.settings

    @property
    def segments(self):
        return self._fit.segments

    @property
    def preprocessing(self) -> dict[str, Any]:
        return self._fit.preprocessing

    def evaluate(self, time: float | Iterable[float] | np.ndarray) -> np.ndarray:
        return np.asarray(self._fit.evaluate(time), dtype=float)

    def velocity(self, time: float | Iterable[float] | np.ndarray) -> np.ndarray:
        return np.asarray(self._fit.derivative(time), dtype=float)

    def resample_time(self, dt_out: float | None = None) -> pd.DataFrame:
        step = self._fit.settings.dt_out if dt_out is None else float(dt_out)
        return resample_uniform_time(self._fit, dt_out=step)

    def resample_space(self, ds_mm: float) -> pd.DataFrame:
        return resample_uniform_space(self._fit, ds_mm=float(ds_mm))


def fit(
    data: np.ndarray | pd.DataFrame | Mapping[str, Any] | str | Path,
    *,
    columns: Iterable[str] | None = None,
    default_error: float = 1.0,
    **kwargs: Any,
) -> FitResult:
    prepared = _prepare_input(data, columns=columns, default_error=default_error)
    return FitResult(fit_trajectory(prepared, **kwargs))


def fit_csv(path: str | Path, **kwargs: Any) -> FitResult:
    return FitResult(fit_trajectory(path, **kwargs))


def _prepare_input(
    data: np.ndarray | pd.DataFrame | Mapping[str, Any] | str | Path,
    *,
    columns: Iterable[str] | None,
    default_error: float,
):
    if isinstance(data, (str, Path)):
        return data
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, Mapping):
        frame = pd.DataFrame(data)
    else:
        frame = _array_to_frame(np.asarray(data), columns=columns, default_error=default_error)
    if "error" not in frame.columns and "sigma" not in frame.columns and "err" not in frame.columns:
        frame = frame.copy()
        frame["error"] = float(default_error)
    return frame


def _array_to_frame(array: np.ndarray, *, columns: Iterable[str] | None, default_error: float) -> pd.DataFrame:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}.")
    if array.shape[1] not in (4, 5):
        raise ValueError(
            "Expected array input with 4 or 5 columns: "
            "[time, x, y, z] or [time, x, y, z, error]."
        )
    if columns is None:
        names = list(DEFAULT_COLUMNS if array.shape[1] == 5 else FOUR_COLUMN_DEFAULTS)
    else:
        names = list(columns)
        if len(names) != array.shape[1]:
            raise ValueError(f"Expected {array.shape[1]} column names, got {len(names)}.")
    frame = pd.DataFrame(array, columns=names)
    if array.shape[1] == 4 and "error" not in frame.columns and "sigma" not in frame.columns and "err" not in frame.columns:
        frame["error"] = float(default_error)
    return frame
