from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy import linalg, optimize, sparse
from scipy.interpolate import BSpline, PchipInterpolator


REQUIRED_COLUMNS = ("time", "x", "y", "z", "error")
ProgressFn = Callable[[str], None]


@dataclass(slots=True)
class FitSettings:
    degree: int = 3
    robust: bool = True
    huber_delta: float = 1.5
    dt_out: float = 0.020
    sigma_floor: float | None = None
    fallback_sigma: float | None = None
    knot_spacing_factor: float = 5.0
    min_control_points: int = 10
    max_control_points: int = 200
    lambda_bounds: tuple[float, float] = (1.0e-8, 1.0e8)
    lambda_grid_size: int = 41
    lambda_selection_max_samples: int | None = 20000
    max_irls_iter: int = 20
    irls_tol: float = 1.0e-6
    weight_cap_quantile: float = 0.99
    error_linear_min_weight: float = 1.0
    window_edge_min_weight: float = 1.0
    segment_gap_factor: float | None = None
    segment_gap_abs: float | None = None
    fit_window_duration: float | None = None
    fit_window_overlap: float = 0.0
    arc_tol_scale: float = 0.01
    arc_max_depth: int = 20
    trust_error_column: bool = True


@dataclass(slots=True)
class SegmentFit:
    segment_id: int
    degree: int
    time_start: float
    time_end: float
    time_scale: float
    knots: np.ndarray
    control_points: np.ndarray
    lambda_value: float
    edf_per_axis: float
    wrss: float
    reduced_chi_square: float | None
    irls_iterations: int
    sigma: np.ndarray
    measurement_weights: np.ndarray
    robust_weights: np.ndarray
    times: np.ndarray
    observations: np.ndarray
    fitted_samples: np.ndarray
    residual_norm: np.ndarray

    _spline: BSpline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._spline = BSpline(self.knots, self.control_points, self.degree, axis=0, extrapolate=False)

    def _to_unit_time(self, time: np.ndarray | float) -> np.ndarray:
        if self.time_scale <= 0.0:
            return np.zeros_like(np.asarray(time, dtype=float))
        return (np.asarray(time, dtype=float) - self.time_start) / self.time_scale

    def evaluate(self, time: np.ndarray | float) -> np.ndarray:
        u = self._to_unit_time(time)
        return np.asarray(self._spline(u), dtype=float)

    def derivative(self, time: np.ndarray | float) -> np.ndarray:
        u = self._to_unit_time(time)
        deriv = np.asarray(self._spline.derivative()(u), dtype=float)
        if self.time_scale > 0.0:
            deriv /= self.time_scale
        return deriv


@dataclass(slots=True)
class TrajectoryFit:
    settings: FitSettings
    segments: list[SegmentFit]
    diagnostics: dict[str, Any]
    preprocessing: dict[str, Any]

    _segment_starts: np.ndarray = field(init=False, repr=False)
    _segment_ends: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._segment_starts = np.array([segment.time_start for segment in self.segments], dtype=float)
        self._segment_ends = np.array([segment.time_end for segment in self.segments], dtype=float)

    def __iter__(self):
        yield self
        yield self.diagnostics

    def evaluate(self, time: np.ndarray | float) -> np.ndarray:
        query = np.atleast_1d(np.asarray(time, dtype=float))
        out = np.empty((query.size, 3), dtype=float)
        for idx, value in enumerate(query):
            active = _active_segments(self, float(value))
            out[idx] = _blend_segments(active, float(value), derivative=False)
        if np.ndim(time) == 0:
            return out[0]
        return out

    def derivative(self, time: np.ndarray | float) -> np.ndarray:
        query = np.atleast_1d(np.asarray(time, dtype=float))
        out = np.empty((query.size, 3), dtype=float)
        for idx, value in enumerate(query):
            active = _active_segments(self, float(value))
            out[idx] = _blend_segments(active, float(value), derivative=True)
        if np.ndim(time) == 0:
            return out[0]
        return out


def load_trajectory_table(data: pd.DataFrame | Mapping[str, Any] | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, Mapping):
        frame = pd.DataFrame(data)
    else:
        path = Path(data)
        frame = _read_csv_flex(path)
    frame = _standardize_columns(frame)
    frame = frame.loc[:, list(REQUIRED_COLUMNS)]
    return frame


def fit_trajectory(
    data: pd.DataFrame | Mapping[str, Any] | str | Path,
    *,
    degree: int = 3,
    robust: bool = True,
    huber_delta: float = 1.5,
    dt_out: float = 0.020,
    sigma_floor: float | None = None,
    fallback_sigma: float | None = None,
    knot_spacing_factor: float = 5.0,
    min_control_points: int = 10,
    max_control_points: int = 200,
    lambda_bounds: tuple[float, float] = (1.0e-8, 1.0e8),
    lambda_grid_size: int = 41,
    lambda_selection_max_samples: int | None = 20000,
    max_irls_iter: int = 20,
    irls_tol: float = 1.0e-6,
    weight_cap_quantile: float = 0.99,
    error_linear_min_weight: float = 1.0,
    window_edge_min_weight: float = 1.0,
    segment_gap_factor: float | None = None,
    segment_gap_s: float | None = None,
    fit_window_duration: float | None = None,
    fit_window_overlap: float = 0.0,
    arc_tol_scale: float = 0.01,
    arc_max_depth: int = 20,
    trust_error_column: bool = True,
    n_jobs: int = 1,
    progress_fn: ProgressFn | None = None,
) -> TrajectoryFit:
    _emit_progress(progress_fn, "Loading trajectory table.")
    settings = FitSettings(
        degree=degree,
        robust=robust,
        huber_delta=huber_delta,
        dt_out=dt_out,
        sigma_floor=sigma_floor,
        fallback_sigma=fallback_sigma,
        knot_spacing_factor=knot_spacing_factor,
        min_control_points=min_control_points,
        max_control_points=max_control_points,
        lambda_bounds=lambda_bounds,
        lambda_grid_size=lambda_grid_size,
        lambda_selection_max_samples=lambda_selection_max_samples,
        max_irls_iter=max_irls_iter,
        irls_tol=irls_tol,
        weight_cap_quantile=weight_cap_quantile,
        error_linear_min_weight=error_linear_min_weight,
        window_edge_min_weight=window_edge_min_weight,
        segment_gap_factor=segment_gap_factor,
        segment_gap_abs=segment_gap_s,
        fit_window_duration=fit_window_duration,
        fit_window_overlap=fit_window_overlap,
        arc_tol_scale=arc_tol_scale,
        arc_max_depth=arc_max_depth,
        trust_error_column=trust_error_column,
    )
    frame = load_trajectory_table(data)
    _emit_progress(progress_fn, f"Loaded {len(frame)} rows. Preprocessing trajectory data.")
    cleaned, preprocessing = _preprocess_frame(frame, settings)
    segments_idx = _split_segments(cleaned["time"].to_numpy(dtype=float), settings)
    total_segments = len(segments_idx)
    progress_every = max(1, (total_segments + 9) // 10) if total_segments > 10 else 1
    _emit_progress(
        progress_fn,
        f"Preprocessing complete: {len(cleaned)} usable rows across {total_segments} segment(s).",
    )
    segments = _fit_all_segments(
        cleaned,
        segments_idx,
        settings,
        total_segments=total_segments,
        progress_every=progress_every,
        n_jobs=n_jobs,
        progress_fn=progress_fn,
    )
    fit = TrajectoryFit(settings=settings, segments=segments, diagnostics={}, preprocessing=preprocessing)
    diagnostics = _build_global_diagnostics(cleaned, fit, settings, preprocessing)
    fit.diagnostics = diagnostics
    _emit_progress(progress_fn, "Fit complete.")
    return fit


def resample_uniform_time(fit: TrajectoryFit, dt_out: float = 0.020) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for start, end in _coverage_intervals(fit.segments):
        grid = _build_time_grid(start, end, dt_out)
        values = fit.evaluate(grid)
        chunk = pd.DataFrame(
            {
                "time": grid,
                "x": values[:, 0],
                "y": values[:, 1],
                "z": values[:, 2],
            }
        )
        rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=["time", "x", "y", "z"])
    return pd.concat(rows, ignore_index=True)


def resample_uniform_space(
    fit: TrajectoryFit,
    ds_mm: float,
    arc_tol_mm: float | None = None,
) -> pd.DataFrame:
    if ds_mm <= 0.0:
        raise ValueError("ds_mm must be positive.")
    tol = arc_tol_mm if arc_tol_mm is not None else fit.settings.arc_tol_scale * ds_mm
    rows: list[pd.DataFrame] = []
    s_offset = 0.0
    for start, end in _coverage_intervals(fit.segments):
        table = _build_arc_length_table_fit(fit, start, end, tol, fit.settings.arc_max_depth)
        s_values = table[:, 1]
        total_length = float(s_values[-1])
        if total_length <= max(1.0e-9, 0.1 * ds_mm):
            time_values = np.array([start], dtype=float)
            points = fit.evaluate(time_values)
            chunk = pd.DataFrame(
                {
                    "s": np.array([s_offset], dtype=float),
                    "time": time_values,
                    "x": points[:, 0],
                    "y": points[:, 1],
                    "z": points[:, 2],
                }
            )
            rows.append(chunk)
            continue
        target_s = np.arange(0.0, total_length + 0.5 * ds_mm, ds_mm, dtype=float)
        if not np.isclose(target_s[-1], total_length):
            target_s = np.append(target_s, total_length)
        target_s = np.clip(target_s, 0.0, total_length)
        unique_s, unique_t = _strictly_increasing_xy(table[:, 1], table[:, 0])
        if unique_s.size == 1:
            time_values = np.full_like(target_s, fill_value=start)
        else:
            inverse = PchipInterpolator(unique_s, unique_t, extrapolate=False)
            time_values = np.asarray(inverse(target_s), dtype=float)
        points = fit.evaluate(time_values)
        chunk = pd.DataFrame(
            {
                "s": target_s + s_offset,
                "time": time_values,
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
            }
        )
        rows.append(chunk)
        s_offset += total_length
    if not rows:
        return pd.DataFrame(columns=["s", "time", "x", "y", "z"])
    return pd.concat(rows, ignore_index=True)


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    dt_out: float = 0.020,
    ds_mm: float = 1.0,
    *,
    time_output: str | Path | None = None,
    space_output: str | Path | None = None,
    diagnostics_output: str | Path | None = None,
    lambda_lower: float | None = None,
    lambda_upper: float | None = None,
    segment_gap: float | None = None,
    fit_window_duration: float | None = None,
    fit_window_overlap: float | None = None,
    progress_fn: ProgressFn | None = None,
    **fit_kwargs: Any,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if lambda_lower is not None or lambda_upper is not None:
        lo = lambda_lower if lambda_lower is not None else 1.0e-8
        hi = lambda_upper if lambda_upper is not None else 1.0e8
        fit_kwargs["lambda_bounds"] = (lo, hi)
    if segment_gap is not None and "segment_gap_s" not in fit_kwargs:
        fit_kwargs["segment_gap_s"] = segment_gap
    if fit_window_duration is not None and "fit_window_duration" not in fit_kwargs:
        fit_kwargs["fit_window_duration"] = fit_window_duration
    if fit_window_overlap is not None and "fit_window_overlap" not in fit_kwargs:
        fit_kwargs["fit_window_overlap"] = fit_window_overlap
    fit = fit_trajectory(input_path, dt_out=dt_out, progress_fn=progress_fn, **fit_kwargs)
    diagnostics = fit.diagnostics
    _emit_progress(progress_fn, "Resampling onto uniform time grid.")
    uniform_time = resample_uniform_time(fit, dt_out=dt_out)
    _emit_progress(progress_fn, "Resampling onto uniform space grid.")
    uniform_space = resample_uniform_space(fit, ds_mm=ds_mm)
    time_path = Path(time_output) if time_output is not None else output_dir / "trajectory_uniform_time.csv"
    space_path = Path(space_output) if space_output is not None else output_dir / "trajectory_uniform_space.csv"
    diag_path = Path(diagnostics_output) if diagnostics_output is not None else output_dir / "trajectory_diagnostics.json"
    uniform_time.to_csv(time_path, index=False)
    uniform_space.to_csv(space_path, index=False)
    diag_path.write_text(json.dumps(_to_builtin(diagnostics), indent=2))
    _emit_progress(progress_fn, f"Wrote outputs to {output_dir}.")
    return {
        "uniform_time": time_path,
        "uniform_space": space_path,
        "diagnostics": diag_path,
        "uniform_time_csv": time_path,
        "uniform_space_csv": space_path,
        "diagnostics_json": diag_path,
    }


def plot_diagnostics(fit: TrajectoryFit, segment_id: int = 0):  # pragma: no cover - optional helper
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plot_diagnostics().") from exc

    segment = fit.segments[segment_id]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(segment.observations[:, 0], segment.observations[:, 1], segment.observations[:, 2], ".", alpha=0.4, label="raw")
    ax3d.plot(
        segment.fitted_samples[:, 0],
        segment.fitted_samples[:, 1],
        segment.fitted_samples[:, 2],
        "-",
        linewidth=2.0,
        label="fit",
    )
    ax3d.set_title("Trajectory")
    ax3d.legend(loc="best")

    axes[0, 1].plot(segment.times, segment.residual_norm)
    axes[0, 1].set_title("Residual Norm")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("mm")

    axes[1, 0].plot(segment.times, segment.robust_weights)
    axes[1, 0].set_title("Robust Weights")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("weight")

    table = _build_arc_length_table(segment, max(fit.settings.arc_tol_scale, 1.0e-3), fit.settings.arc_max_depth)
    axes[1, 1].plot(table[:, 0], table[:, 1])
    axes[1, 1].set_title("Arc Length Mapping")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("s")
    fig.tight_layout()
    return fig


def _read_csv_flex(path: Path) -> pd.DataFrame:
    header_try = pd.read_csv(path)
    if set(REQUIRED_COLUMNS).issubset(header_try.columns):
        return header_try
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        engine="python",
        comment="#",
        names=list(REQUIRED_COLUMNS),
    )


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: col.strip().lstrip("#").strip().lower() for col in frame.columns}
    frame = frame.rename(columns=rename_map)
    alias_map = {"t": "time", "err": "error", "sigma": "error"}
    for source, target in alias_map.items():
        if source in frame.columns and target not in frame.columns:
            frame[target] = frame[source]
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame


def _preprocess_frame(frame: pd.DataFrame, settings: FitSettings) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric = frame.copy()
    for col in REQUIRED_COLUMNS:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    numeric = numeric.dropna(subset=["time", "x", "y", "z"]).sort_values("time").reset_index(drop=True)
    input_rows = len(frame)
    if numeric.empty:
        raise ValueError("No usable trajectory rows remain after dropping NaN values.")
    error_values = numeric["error"].to_numpy(dtype=float)
    positive_error = error_values[np.isfinite(error_values) & (error_values > 0.0)]
    fallback_sigma = settings.fallback_sigma
    if fallback_sigma is None:
        fallback_sigma = _estimate_fallback_sigma(numeric[["x", "y", "z"]].to_numpy(dtype=float))
    sigma_floor = settings.sigma_floor
    if sigma_floor is None:
        if positive_error.size > 0:
            sigma_floor = max(1.0e-6, 0.05 * float(np.median(positive_error)))
        else:
            sigma_floor = max(1.0e-6, 0.05 * fallback_sigma)
    sigma = np.where(np.isfinite(error_values) & (error_values > 0.0), error_values, fallback_sigma)
    sigma = np.maximum(sigma, sigma_floor)
    numeric["error"] = sigma
    duplicate_count = int(numeric["time"].duplicated().sum())
    if duplicate_count:
        numeric = _merge_duplicate_timestamps(numeric)
    time_values = numeric["time"].to_numpy(dtype=float)
    if np.any(np.diff(time_values) <= 0.0):
        raise ValueError("Time must be strictly increasing after preprocessing.")
    preprocessing = {
        "input_rows": input_rows,
        "rows_after_dropna": int(len(frame.dropna(subset=["time", "x", "y", "z"]))),
        "rows_after_preprocess": int(len(numeric)),
        "dropped_rows": int(input_rows - len(numeric)),
        "duplicate_timestamps_merged": duplicate_count,
        "sigma_floor": float(sigma_floor),
        "fallback_sigma": float(fallback_sigma),
        "error_linear_min_weight": float(settings.error_linear_min_weight),
        "window_edge_min_weight": float(settings.window_edge_min_weight),
        "assumes_isotropic_error": True,
    }
    return numeric.reset_index(drop=True), preprocessing


def _estimate_fallback_sigma(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 1.0
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if np.allclose(diffs, 0.0):
        return 1.0
    return max(1.0e-3, float(np.median(diffs) / np.sqrt(2.0)))


def _merge_duplicate_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for time_value, group in frame.groupby("time", sort=True):
        sigma = group["error"].to_numpy(dtype=float)
        weights = 1.0 / np.maximum(sigma, 1.0e-12) ** 2
        position = group[["x", "y", "z"]].to_numpy(dtype=float)
        merged_pos = np.average(position, axis=0, weights=weights)
        merged_sigma = max(float(np.sqrt(1.0 / weights.sum())), float(np.min(sigma)))
        rows.append(
            {
                "time": float(time_value),
                "x": float(merged_pos[0]),
                "y": float(merged_pos[1]),
                "z": float(merged_pos[2]),
                "error": merged_sigma,
            }
        )
    return pd.DataFrame(rows)


def _split_segments(time_values: np.ndarray, settings: FitSettings) -> list[np.ndarray]:
    if time_values.size == 0:
        return []
    if time_values.size == 1:
        return [np.array([0], dtype=int)]
    dt = np.diff(time_values)
    gap_threshold = np.inf
    if settings.segment_gap_factor is not None:
        positive_dt = dt[dt > 0.0]
        if positive_dt.size:
            gap_threshold = min(gap_threshold, settings.segment_gap_factor * float(np.median(positive_dt)))
    if settings.segment_gap_abs is not None:
        gap_threshold = min(gap_threshold, settings.segment_gap_abs)
    base_breaks = [0]
    if np.isfinite(gap_threshold):
        base_breaks.extend((np.flatnonzero(dt > gap_threshold) + 1).tolist())
    base_breaks.append(len(time_values))

    segments: list[np.ndarray] = []
    for base_start, base_end in zip(base_breaks[:-1], base_breaks[1:]):
        base_idx = np.arange(base_start, base_end, dtype=int)
        if base_idx.size == 0:
            continue
        if settings.fit_window_duration is None:
            segments.append(base_idx)
            continue
        segments.extend(_windowize_indices(time_values, base_idx, settings.fit_window_duration, settings.fit_window_overlap))
    return segments


def _fit_segment(
    frame: pd.DataFrame,
    settings: FitSettings,
    segment_id: int,
    *,
    total_segments: int,
    progress_every: int,
    progress_fn: ProgressFn | None = None,
) -> SegmentFit:
    times = frame["time"].to_numpy(dtype=float)
    observations = frame[["x", "y", "z"]].to_numpy(dtype=float)
    sigma = frame["error"].to_numpy(dtype=float)
    degree = min(settings.degree, max(0, len(frame) - 1))
    time_start = float(times[0])
    time_end = float(times[-1])
    time_scale = max(time_end - time_start, 1.0)
    u = (times - time_start) / time_scale if time_end > time_start else np.zeros_like(times)
    n_control = _choose_control_points(u, degree, len(times), settings)
    knots = _make_open_knot_vector(n_control, degree)
    design = _bspline_design_matrix(u, knots, degree)
    penalty = _second_difference_penalty(n_control)
    measurement_weights = _measurement_weights(times, sigma, settings.weight_cap_quantile)
    measurement_weights = _apply_linear_error_weighting(
        measurement_weights,
        sigma,
        settings.error_linear_min_weight,
    )
    measurement_weights = _apply_window_center_weighting(
        measurement_weights,
        u,
        settings.window_edge_min_weight,
    )
    lambda_design = design
    lambda_observations = observations
    lambda_sigma = sigma
    lambda_weights = measurement_weights
    detailed_progress = total_segments <= 8 or segment_id in {0, total_segments - 1}
    if settings.lambda_selection_max_samples is not None and len(frame) > settings.lambda_selection_max_samples:
        sample_idx = np.linspace(
            0,
            len(frame) - 1,
            settings.lambda_selection_max_samples,
            dtype=int,
        )
        lambda_design = design[sample_idx]
        lambda_observations = observations[sample_idx]
        lambda_sigma = sigma[sample_idx]
        lambda_weights = measurement_weights[sample_idx]
    if detailed_progress:
        if settings.lambda_selection_max_samples is not None and len(frame) > settings.lambda_selection_max_samples:
            _emit_progress(
                progress_fn,
                (
                    f"Segment {segment_id + 1}/{total_segments}: selecting lambda on {len(sample_idx)} sampled rows "
                    f"out of {len(frame)}."
                ),
            )
        _emit_progress(
            progress_fn,
            (
                f"Segment {segment_id + 1}/{total_segments}: fitting {len(frame)} samples, "
                f"degree={degree}, control_points={n_control}. Selecting lambda."
            ),
        )
    lambda_value, edf_guess = _choose_lambda(
        lambda_design,
        lambda_observations,
        lambda_sigma,
        lambda_weights,
        penalty,
        settings,
        progress_fn=progress_fn,
        segment_id=segment_id,
        total_segments=total_segments,
        detailed_progress=detailed_progress,
    )
    if detailed_progress:
        _emit_progress(
            progress_fn,
            f"Segment {segment_id + 1}/{total_segments}: lambda={lambda_value:.6g}. Running robust fit.",
        )
    control_points, robust_weights, irls_iterations = _robust_fit(
        design,
        observations,
        sigma,
        measurement_weights,
        penalty,
        lambda_value,
        settings,
        progress_fn=progress_fn,
        segment_id=segment_id,
        total_segments=total_segments,
        detailed_progress=detailed_progress,
    )
    fitted = design @ control_points
    residual = fitted - observations
    residual_norm = np.linalg.norm(residual, axis=1)
    wrss = float(np.sum((residual_norm / sigma) ** 2))
    final_gram, _ = _normal_equations(design, observations, measurement_weights * robust_weights)
    edf_per_axis = _effective_degrees_of_freedom(final_gram, lambda_value, penalty)
    dof = max(3.0 * len(times) - 3.0 * edf_per_axis, 1.0)
    reduced_chi_square = wrss / dof if settings.trust_error_column else None
    if detailed_progress:
        _emit_progress(
            progress_fn,
            (
                f"Segment {segment_id + 1}/{total_segments}: done after {irls_iterations} IRLS iteration(s), "
                f"WRSS={wrss:.3f}."
            ),
        )
    elif (segment_id + 1) % progress_every == 0 or segment_id + 1 == total_segments:
        _emit_progress(progress_fn, f"Processed {segment_id + 1}/{total_segments} fit segment(s).")
    return SegmentFit(
        segment_id=segment_id,
        degree=degree,
        time_start=time_start,
        time_end=time_end,
        time_scale=time_scale,
        knots=knots,
        control_points=control_points,
        lambda_value=float(lambda_value),
        edf_per_axis=float(edf_per_axis if np.isfinite(edf_per_axis) else edf_guess),
        wrss=wrss,
        reduced_chi_square=None if reduced_chi_square is None else float(reduced_chi_square),
        irls_iterations=irls_iterations,
        sigma=sigma,
        measurement_weights=measurement_weights,
        robust_weights=robust_weights,
        times=times,
        observations=observations,
        fitted_samples=fitted,
        residual_norm=residual_norm,
    )


def _fit_all_segments(
    cleaned: pd.DataFrame,
    segments_idx: list[np.ndarray],
    settings: FitSettings,
    *,
    total_segments: int,
    progress_every: int,
    n_jobs: int,
    progress_fn: ProgressFn | None = None,
) -> list[SegmentFit]:
    if total_segments == 0:
        return []

    workers = max(1, int(n_jobs))
    if workers == 1 or total_segments == 1:
        segments: list[SegmentFit] = []
        for segment_id, idx in enumerate(segments_idx):
            segment_frame = cleaned.iloc[idx].reset_index(drop=True)
            segments.append(
                _fit_segment(
                    segment_frame,
                    settings,
                    segment_id,
                    total_segments=total_segments,
                    progress_every=progress_every,
                    progress_fn=progress_fn,
                )
            )
        return segments

    chunk_count = min(workers, total_segments)
    segments: list[SegmentFit | None] = [None] * total_segments
    completed = 0
    boundaries = np.linspace(0, total_segments, chunk_count + 1, dtype=int)
    chunk_ranges = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(chunk_count) if boundaries[i] < boundaries[i + 1]]

    def submit_all(executor: ProcessPoolExecutor | ThreadPoolExecutor) -> dict[Future[list[tuple[int, dict[str, Any]]]], tuple[int, int]]:
        pending: dict[Future[list[tuple[int, dict[str, Any]]]], tuple[int, int]] = {}
        for start_segment_id, stop_segment_id in chunk_ranges:
            future = executor.submit(
                _fit_segment_chunk,
                cleaned,
                segments_idx[start_segment_id:stop_segment_id],
                settings,
                start_segment_id,
                total_segments,
                progress_every,
            )
            pending[future] = (start_segment_id, stop_segment_id)
        return pending

    def drain_pending(pending: dict[Future[list[tuple[int, dict[str, Any]]]], tuple[int, int]]) -> None:
        nonlocal completed
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                start_segment_id, stop_segment_id = pending.pop(future)
                for segment_id, payload in future.result():
                    segments[segment_id] = SegmentFit(**payload)
                completed += stop_segment_id - start_segment_id
                _emit_progress(progress_fn, f"Processed {completed}/{total_segments} fit segment(s).")

    try:
        _emit_progress(progress_fn, f"Fitting {chunk_count} segment chunks in parallel with {workers} worker processes.")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            drain_pending(submit_all(executor))
    except (OSError, PermissionError) as exc:
        _emit_progress(
            progress_fn,
            f"Process-based parallelism unavailable ({exc}). Falling back to {workers} worker threads on chunked segments.",
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            drain_pending(submit_all(executor))

    return [segment for segment in segments if segment is not None]


def _segment_fit_payload(segment: SegmentFit) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "degree": segment.degree,
        "time_start": segment.time_start,
        "time_end": segment.time_end,
        "time_scale": segment.time_scale,
        "knots": segment.knots,
        "control_points": segment.control_points,
        "lambda_value": segment.lambda_value,
        "edf_per_axis": segment.edf_per_axis,
        "wrss": segment.wrss,
        "reduced_chi_square": segment.reduced_chi_square,
        "irls_iterations": segment.irls_iterations,
        "sigma": segment.sigma,
        "measurement_weights": segment.measurement_weights,
        "robust_weights": segment.robust_weights,
        "times": segment.times,
        "observations": segment.observations,
        "fitted_samples": segment.fitted_samples,
        "residual_norm": segment.residual_norm,
    }


def _fit_segment_chunk(
    cleaned: pd.DataFrame,
    local_segments_idx: list[np.ndarray],
    settings: FitSettings,
    start_segment_id: int,
    total_segments: int,
    progress_every: int,
) -> list[tuple[int, dict[str, Any]]]:
    fitted_chunk: list[tuple[int, dict[str, Any]]] = []
    for local_offset, idx in enumerate(local_segments_idx):
        segment_id = start_segment_id + local_offset
        segment_frame = cleaned.iloc[idx].reset_index(drop=True)
        segment = _fit_segment(
            segment_frame,
            settings,
            segment_id,
            total_segments=total_segments,
            progress_every=progress_every,
            progress_fn=None,
        )
        fitted_chunk.append((segment_id, _segment_fit_payload(segment)))
    return fitted_chunk


def _choose_control_points(u: np.ndarray, degree: int, n_samples: int, settings: FitSettings) -> int:
    if n_samples <= degree + 1:
        return n_samples
    if u.size < 2:
        return degree + 1
    dt = np.diff(u)
    positive_dt = dt[dt > 0.0]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else 1.0
    span_dt = float((u[-1] - u[0]) / max(n_samples - 1, 1))
    reference_dt = max(median_dt, span_dt)
    target_spacing = max(settings.knot_spacing_factor * reference_dt, 1.0e-6)
    target_spans = max(1, int(np.ceil(1.0 / target_spacing)))
    target = target_spans + degree
    lower = degree + 1
    if n_samples >= settings.min_control_points:
        lower = max(lower, settings.min_control_points)
    upper = max(lower, min(settings.max_control_points, max(degree + 1, n_samples // 2)))
    return int(np.clip(target, lower, upper))


def _make_open_knot_vector(n_control: int, degree: int) -> np.ndarray:
    if n_control <= degree:
        raise ValueError("n_control must exceed degree.")
    n_internal = n_control - degree - 1
    if n_internal > 0:
        internal = np.linspace(0.0, 1.0, n_internal + 2, dtype=float)[1:-1]
    else:
        internal = np.array([], dtype=float)
    return np.concatenate(
        (
            np.zeros(degree + 1, dtype=float),
            internal,
            np.ones(degree + 1, dtype=float),
        )
    )


def _bspline_design_matrix(u: np.ndarray, knots: np.ndarray, degree: int):
    return sparse.csr_matrix(BSpline.design_matrix(u, knots, degree, extrapolate=False))


def _second_difference_penalty(n_control: int) -> np.ndarray:
    if n_control < 3:
        return np.zeros((n_control, n_control), dtype=float)
    d2 = np.zeros((n_control - 2, n_control), dtype=float)
    idx = np.arange(n_control - 2)
    d2[idx, idx] = 1.0
    d2[idx, idx + 1] = -2.0
    d2[idx, idx + 2] = 1.0
    return d2.T @ d2


def _measurement_weights(times: np.ndarray, sigma: np.ndarray, cap_quantile: float) -> np.ndarray:
    support = _sample_support(times)
    weights = support / np.maximum(sigma, 1.0e-12) ** 2
    if 0.0 < cap_quantile < 1.0:
        cap = float(np.quantile(weights, cap_quantile))
        weights = np.minimum(weights, cap)
    return weights


def _apply_linear_error_weighting(weights: np.ndarray, sigma: np.ndarray, min_weight: float) -> np.ndarray:
    if not (0.0 < min_weight <= 1.0):
        raise ValueError("error_linear_min_weight must satisfy 0 < value <= 1.")
    if np.isclose(min_weight, 1.0):
        return weights

    sigma = np.asarray(sigma, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sigma_min = float(np.min(sigma))
    sigma_max = float(np.max(sigma))
    if not np.isfinite(sigma_min) or not np.isfinite(sigma_max) or np.isclose(sigma_min, sigma_max):
        return weights

    alpha = (sigma - sigma_min) / max(sigma_max - sigma_min, 1.0e-12)
    linear_scale = 1.0 - alpha * (1.0 - min_weight)
    return weights * linear_scale


def _apply_window_center_weighting(weights: np.ndarray, u: np.ndarray, min_weight: float) -> np.ndarray:
    if not (0.0 < min_weight <= 1.0):
        raise ValueError("window_edge_min_weight must satisfy 0 < value <= 1.")
    if np.isclose(min_weight, 1.0):
        return weights

    u = np.asarray(u, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if u.size <= 1:
        return weights

    # Smooth symmetric taper: edges get min_weight, window center gets 1.0.
    taper = min_weight + (1.0 - min_weight) * np.sin(np.pi * np.clip(u, 0.0, 1.0)) ** 2
    return weights * taper


def _sample_support(times: np.ndarray) -> np.ndarray:
    if len(times) <= 1:
        return np.ones(len(times), dtype=float)
    dt = np.diff(times)
    support = np.empty(len(times), dtype=float)
    support[0] = dt[0]
    support[-1] = dt[-1]
    if len(times) > 2:
        support[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    support /= np.mean(support)
    return support


def _windowize_indices(
    time_values: np.ndarray,
    base_idx: np.ndarray,
    window_duration: float,
    overlap: float,
) -> list[np.ndarray]:
    if window_duration <= 0.0:
        raise ValueError("fit_window_duration must be positive.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("fit_window_overlap must satisfy 0 <= overlap < 1.")
    if base_idx.size <= 1:
        return [base_idx]

    base_times = time_values[base_idx]
    duration = float(window_duration)
    step = max(duration * (1.0 - overlap), 1.0e-12)
    starts = [float(base_times[0])]
    final_start = max(float(base_times[0]), float(base_times[-1]) - duration)
    while starts[-1] + step < final_start - 1.0e-12:
        starts.append(starts[-1] + step)
    if starts[-1] < final_start - 1.0e-12:
        starts.append(final_start)

    windows: list[np.ndarray] = []
    for start in starts:
        end = start + duration
        left = int(np.searchsorted(base_times, start, side="left"))
        right = int(np.searchsorted(base_times, end, side="right"))
        if right - left <= 0:
            continue
        window = base_idx[left:right]
        if window.size == 0:
            continue
        if windows and np.array_equal(window, windows[-1]):
            continue
        windows.append(window)
    return windows if windows else [base_idx]


def _choose_lambda(
    design,
    observations: np.ndarray,
    sigma: np.ndarray,
    measurement_weights: np.ndarray,
    penalty: np.ndarray,
    settings: FitSettings,
    *,
    progress_fn: ProgressFn | None = None,
    segment_id: int | None = None,
    total_segments: int | None = None,
    detailed_progress: bool = True,
) -> tuple[float, float]:
    lo, hi = settings.lambda_bounds
    grid = np.logspace(np.log10(lo), np.log10(hi), settings.lambda_grid_size)
    gram, rhs = _normal_equations(design, observations, measurement_weights)
    scores_list: list[float] = []
    for idx, value in enumerate(grid, start=1):
        scores_list.append(
            _lambda_objective(value, design, observations, sigma, penalty, settings, gram=gram, rhs=rhs)
        )
        if detailed_progress and (idx == 1 or idx == len(grid) or idx % max(1, len(grid) // 5) == 0):
            if segment_id is not None and total_segments is not None:
                prefix = f"Segment {segment_id + 1}/{total_segments}: "
            elif segment_id is not None:
                prefix = f"Segment {segment_id + 1}: "
            else:
                prefix = ""
            _emit_progress(
                progress_fn,
                f"{prefix}lambda grid {idx}/{len(grid)} evaluated.",
            )
    scores = np.asarray(scores_list, dtype=float)
    best_idx = int(np.argmin(scores))
    if best_idx == 0:
        bounds = (np.log10(grid[0]), np.log10(grid[1]))
    elif best_idx == len(grid) - 1:
        bounds = (np.log10(grid[-2]), np.log10(grid[-1]))
    else:
        bounds = (np.log10(grid[best_idx - 1]), np.log10(grid[best_idx + 1]))

    result = optimize.minimize_scalar(
        lambda x: _lambda_objective(
            10.0**x,
            design,
            observations,
            sigma,
            penalty,
            settings,
            gram=gram,
            rhs=rhs,
        ),
        bounds=bounds,
        method="bounded",
    )
    lambda_value = float(10.0 ** result.x) if result.success else float(grid[best_idx])
    edf = _effective_degrees_of_freedom(gram, lambda_value, penalty)
    return lambda_value, float(edf)


def _lambda_objective(
    lambda_value: float,
    design,
    observations: np.ndarray,
    sigma: np.ndarray,
    penalty: np.ndarray,
    settings: FitSettings,
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
) -> float:
    control = _solve_penalized_from_normal(gram, rhs, lambda_value, penalty)
    fitted = design @ control
    residual_norm = np.linalg.norm(fitted - observations, axis=1)
    wrss = float(np.sum((residual_norm / sigma) ** 2))
    edf_per_axis = _effective_degrees_of_freedom(gram, lambda_value, penalty)
    total_edf = 3.0 * edf_per_axis
    n_obs = 3.0 * len(observations)
    if settings.trust_error_column:
        dof = max(n_obs - total_edf, 1.0)
        reduced_chi = wrss / dof
        return float(np.log(max(reduced_chi, 1.0e-12)) ** 2)
    denom = max((n_obs - total_edf) ** 2, 1.0)
    return wrss / denom


def _robust_fit(
    design,
    observations: np.ndarray,
    sigma: np.ndarray,
    measurement_weights: np.ndarray,
    penalty: np.ndarray,
    lambda_value: float,
    settings: FitSettings,
    *,
    progress_fn: ProgressFn | None = None,
    segment_id: int | None = None,
    total_segments: int | None = None,
    detailed_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not settings.robust:
        control = _solve_penalized(design, observations, measurement_weights, lambda_value, penalty)
        return control, np.ones(len(observations), dtype=float), 1
    robust_weights = np.ones(len(observations), dtype=float)
    control = _solve_penalized(design, observations, measurement_weights, lambda_value, penalty)
    for iteration in range(1, settings.max_irls_iter + 1):
        total_weights = measurement_weights * robust_weights
        updated = _solve_penalized(design, observations, total_weights, lambda_value, penalty)
        fitted = design @ updated
        residual_norm = np.linalg.norm(fitted - observations, axis=1)
        u = residual_norm / np.maximum(sigma, 1.0e-12)
        new_robust = np.where(u <= settings.huber_delta, 1.0, settings.huber_delta / np.maximum(u, 1.0e-12))
        rel_change = np.linalg.norm(updated - control) / max(np.linalg.norm(control), 1.0e-12)
        weight_change = float(np.max(np.abs(new_robust - robust_weights)))
        control = updated
        robust_weights = new_robust
        if detailed_progress:
            if segment_id is not None and total_segments is not None:
                prefix = f"Segment {segment_id + 1}/{total_segments}: "
            elif segment_id is not None:
                prefix = f"Segment {segment_id + 1}: "
            else:
                prefix = ""
            _emit_progress(
                progress_fn,
                f"{prefix}IRLS {iteration}/{settings.max_irls_iter}, rel_change={rel_change:.3e}, weight_change={weight_change:.3e}.",
            )
        if rel_change < settings.irls_tol and weight_change < 10.0 * settings.irls_tol:
            return control, robust_weights, iteration
    return control, robust_weights, settings.max_irls_iter


def _solve_penalized(
    design,
    observations: np.ndarray,
    weights: np.ndarray,
    lambda_value: float,
    penalty: np.ndarray,
) -> np.ndarray:
    gram, rhs = _normal_equations(design, observations, weights)
    return _solve_penalized_from_normal(gram, rhs, lambda_value, penalty)


def _normal_equations(design, observations: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe_weights = np.maximum(weights, 1.0e-18)
    if sparse.issparse(design):
        weighted_design = design.multiply(safe_weights[:, None])
        gram = np.asarray((design.T @ weighted_design).toarray(), dtype=float)
        rhs = np.asarray(design.T @ (observations * safe_weights[:, None]), dtype=float)
        return gram, rhs
    sqrt_w = np.sqrt(safe_weights)
    weighted_design = design * sqrt_w[:, None]
    weighted_obs = observations * sqrt_w[:, None]
    gram = np.asarray(weighted_design.T @ weighted_design, dtype=float)
    rhs = np.asarray(weighted_design.T @ weighted_obs, dtype=float)
    return gram, rhs


def _solve_penalized_from_normal(
    gram: np.ndarray,
    rhs: np.ndarray,
    lambda_value: float,
    penalty: np.ndarray,
) -> np.ndarray:
    lhs = gram + lambda_value * penalty
    try:
        return linalg.solve(lhs, rhs, assume_a="pos", check_finite=False)
    except linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _effective_degrees_of_freedom(
    gram: np.ndarray,
    lambda_value: float,
    penalty: np.ndarray,
) -> float:
    lhs = gram + lambda_value * penalty
    try:
        influence = linalg.solve(lhs, gram, assume_a="pos", check_finite=False)
    except linalg.LinAlgError:
        influence = np.linalg.lstsq(lhs, gram, rcond=None)[0]
    return float(np.trace(influence))


def _build_global_diagnostics(
    cleaned: pd.DataFrame,
    fit: TrajectoryFit,
    settings: FitSettings,
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    segments = fit.segments
    n_segments = len(segments)
    include_segment_details = n_segments <= 1000
    compute_arc_length = n_segments <= 250
    total_arc = 0.0
    segment_info: list[dict[str, Any]] = []
    if compute_arc_length:
        for start, end in _coverage_intervals(segments):
            table = _build_arc_length_table_fit(fit, start, end, max(settings.arc_tol_scale, 1.0e-3), settings.arc_max_depth)
            total_arc += float(table[-1, 1])
    else:
        total_arc = float("nan")

    if include_segment_details:
        for segment in segments:
            arc_length = None
            if compute_arc_length:
                table = _build_arc_length_table(segment, max(settings.arc_tol_scale, 1.0e-3), settings.arc_max_depth)
                arc_length = float(table[-1, 1])
            segment_info.append(
                {
                    "segment_id": segment.segment_id,
                    "n_samples": int(len(segment.times)),
                    "n_control_points": int(len(segment.control_points)),
                    "degree": int(segment.degree),
                    "time_start": float(segment.time_start),
                    "time_end": float(segment.time_end),
                    "lambda": float(segment.lambda_value),
                    "irls_iterations": int(segment.irls_iterations),
                    "weighted_residual_sum": float(segment.wrss),
                    "edf_per_axis": float(segment.edf_per_axis),
                    "edf_total": float(3.0 * segment.edf_per_axis),
                    "reduced_chi_square": None if segment.reduced_chi_square is None else float(segment.reduced_chi_square),
                    "fraction_downweighted": float(np.mean(segment.robust_weights < 0.999)),
                    "total_arc_length": arc_length,
                }
            )
    reduced_chi_values = [seg.reduced_chi_square for seg in segments if seg.reduced_chi_square is not None]
    return {
        "model": {
            "type": "joint_3d_cubic_pspline",
            "robust_loss": "Huber" if settings.robust else "linear",
            "huber_delta": float(settings.huber_delta),
            "penalty": "second_difference_on_control_points",
            "assumes_isotropic_error": True,
        },
        "preprocessing": preprocessing,
        "n_samples": int(len(cleaned)),
        "n_segments": int(n_segments),
        "chosen_lambda": [float(seg.lambda_value) for seg in segments],
        "irls_iterations": [int(seg.irls_iterations) for seg in segments],
        "final_weighted_residual_sum": float(sum(seg.wrss for seg in segments)),
        "effective_degrees_of_freedom_total": float(sum(3.0 * seg.edf_per_axis for seg in segments)),
        "reduced_chi_square": None if not reduced_chi_values else float(np.mean(reduced_chi_values)),
        "fraction_downweighted": float(
            np.mean(np.concatenate([seg.robust_weights for seg in segments]) < 0.999) if segments else 0.0
        ),
        "total_arc_length": None if not compute_arc_length else float(total_arc),
        "diagnostics_note": None if compute_arc_length and include_segment_details else "Arc-length and/or per-segment diagnostics were truncated for performance.",
        "segments": segment_info,
    }


def _build_time_grid(start: float, end: float, dt: float) -> np.ndarray:
    if dt <= 0.0:
        raise ValueError("dt_out must be positive.")
    if np.isclose(start, end):
        return np.array([start], dtype=float)
    grid = np.arange(start, end + 0.5 * dt, dt, dtype=float)
    if grid[-1] > end:
        grid[-1] = end
    if np.isclose(grid[-1], end, atol=1.0e-10, rtol=0.0):
        grid[-1] = end
    return grid


def _build_arc_length_table(segment: SegmentFit, tol_mm: float, max_depth: int) -> np.ndarray:
    spans = np.unique(segment.knots)
    samples: list[tuple[float, np.ndarray]] = []
    start_time = segment.time_start
    samples.append((start_time, segment.evaluate(np.array([start_time]))[0]))
    for left_u, right_u in zip(spans[:-1], spans[1:]):
        if right_u <= left_u:
            continue
        left_t = segment.time_start + left_u * segment.time_scale
        right_t = segment.time_start + right_u * segment.time_scale
        left_p = segment.evaluate(np.array([left_t]))[0]
        right_p = segment.evaluate(np.array([right_t]))[0]
        _subdivide_arc(segment, left_t, right_t, left_p, right_p, tol_mm, max_depth, samples)
    times = np.array([entry[0] for entry in samples], dtype=float)
    points = np.array([entry[1] for entry in samples], dtype=float)
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    points = points[order]
    unique_times, unique_idx = np.unique(times, return_index=True)
    points = points[unique_idx]
    if len(unique_times) == 1:
        return np.column_stack((unique_times, np.array([0.0], dtype=float)))
    seglen = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seglen)))
    return np.column_stack((unique_times, s))


def _subdivide_arc(
    segment: SegmentFit,
    left_t: float,
    right_t: float,
    left_p: np.ndarray,
    right_p: np.ndarray,
    tol_mm: float,
    depth: int,
    samples: list[tuple[float, np.ndarray]],
) -> None:
    mid_t = 0.5 * (left_t + right_t)
    mid_p = segment.evaluate(np.array([mid_t]))[0]
    chord = float(np.linalg.norm(right_p - left_p))
    poly = float(np.linalg.norm(mid_p - left_p) + np.linalg.norm(right_p - mid_p))
    if depth <= 0 or poly - chord <= tol_mm:
        samples.append((mid_t, mid_p))
        samples.append((right_t, right_p))
        return
    _subdivide_arc(segment, left_t, mid_t, left_p, mid_p, tol_mm, depth - 1, samples)
    _subdivide_arc(segment, mid_t, right_t, mid_p, right_p, tol_mm, depth - 1, samples)


def _strictly_increasing_xy(x: np.ndarray, y: np.ndarray, eps: float = 1.0e-12) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.ones(len(x), dtype=bool)
    last = x[0]
    for idx in range(1, len(x)):
        if x[idx] <= last + eps:
            keep[idx] = False
        else:
            last = x[idx]
    return x[keep], y[keep]


def _active_segments(segments: list[SegmentFit] | TrajectoryFit, time_value: float) -> list[tuple[int, SegmentFit]]:
    if isinstance(segments, TrajectoryFit):
        fit = segments
        if fit._segment_starts.size == 0:
            return []
        left = int(np.searchsorted(fit._segment_ends, time_value - 1.0e-12, side="left"))
        right = int(np.searchsorted(fit._segment_starts, time_value + 1.0e-12, side="right"))
        active = [
            (idx, fit.segments[idx])
            for idx in range(left, right)
            if fit.segments[idx].time_start - 1.0e-12 <= time_value <= fit.segments[idx].time_end + 1.0e-12
        ]
    else:
        active = [
            (idx, segment)
            for idx, segment in enumerate(segments)
            if segment.time_start - 1.0e-12 <= time_value <= segment.time_end + 1.0e-12
        ]
    if active:
        return active
    seq = segments.segments if isinstance(segments, TrajectoryFit) else segments
    if time_value < seq[0].time_start:
        return [(0, seq[0])]
    return [(len(seq) - 1, seq[-1])]


def _blend_segments(active: list[tuple[int, SegmentFit]], time_value: float, *, derivative: bool) -> np.ndarray:
    if len(active) == 1:
        idx, segment = active[0]
        values = segment.derivative(np.array([time_value])) if derivative else segment.evaluate(np.array([time_value]))
        return values[0]
    weights = np.array([_segment_blend_weight(active, pos, time_value) for pos in range(len(active))], dtype=float)
    if np.allclose(weights.sum(), 0.0):
        weights = np.ones(len(active), dtype=float)
    weights /= weights.sum()
    result = np.zeros(3, dtype=float)
    for weight, (_, segment) in zip(weights, active):
        values = segment.derivative(np.array([time_value])) if derivative else segment.evaluate(np.array([time_value]))
        result += weight * values[0]
    return result


def _segment_blend_weight(active: list[tuple[int, SegmentFit]], pos: int, time_value: float) -> float:
    _, segment = active[pos]
    left_overlap = 0.0
    right_overlap = 0.0
    if pos > 0:
        left_overlap = max(0.0, active[pos - 1][1].time_end - segment.time_start)
    if pos + 1 < len(active):
        right_overlap = max(0.0, segment.time_end - active[pos + 1][1].time_start)

    weight = 1.0
    if left_overlap > 1.0e-12 and time_value < segment.time_start + left_overlap:
        alpha = np.clip((time_value - segment.time_start) / left_overlap, 0.0, 1.0)
        weight *= 0.5 - 0.5 * np.cos(np.pi * alpha)
    if right_overlap > 1.0e-12 and time_value > segment.time_end - right_overlap:
        beta = np.clip((segment.time_end - time_value) / right_overlap, 0.0, 1.0)
        weight *= 0.5 - 0.5 * np.cos(np.pi * beta)
    return float(weight)


def _coverage_intervals(segments: list[SegmentFit]) -> list[tuple[float, float]]:
    intervals = sorted((segment.time_start, segment.time_end) for segment in segments)
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1.0e-12:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_arc_length_table_fit(fit: TrajectoryFit, start: float, end: float, tol_mm: float, max_depth: int) -> np.ndarray:
    breakpoints = {float(start), float(end)}
    for segment in fit.segments:
        if segment.time_end < start or segment.time_start > end:
            continue
        for u_value in np.unique(segment.knots):
            knot_time = segment.time_start + float(u_value) * segment.time_scale
            if start <= knot_time <= end:
                breakpoints.add(float(knot_time))
    ordered = np.array(sorted(breakpoints), dtype=float)
    samples: list[tuple[float, np.ndarray]] = [(start, fit.evaluate(np.array([start]))[0])]
    for left_t, right_t in zip(ordered[:-1], ordered[1:]):
        if right_t <= left_t:
            continue
        left_p = fit.evaluate(np.array([left_t]))[0]
        right_p = fit.evaluate(np.array([right_t]))[0]
        _subdivide_arc_eval(fit, left_t, right_t, left_p, right_p, tol_mm, max_depth, samples)
    times = np.array([entry[0] for entry in samples], dtype=float)
    points = np.array([entry[1] for entry in samples], dtype=float)
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    points = points[order]
    unique_times, unique_idx = np.unique(times, return_index=True)
    points = points[unique_idx]
    if len(unique_times) == 1:
        return np.column_stack((unique_times, np.array([0.0], dtype=float)))
    seglen = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seglen)))
    return np.column_stack((unique_times, s))


def _subdivide_arc_eval(
    fit: TrajectoryFit,
    left_t: float,
    right_t: float,
    left_p: np.ndarray,
    right_p: np.ndarray,
    tol_mm: float,
    depth: int,
    samples: list[tuple[float, np.ndarray]],
) -> None:
    mid_t = 0.5 * (left_t + right_t)
    mid_p = fit.evaluate(np.array([mid_t]))[0]
    chord = float(np.linalg.norm(right_p - left_p))
    poly = float(np.linalg.norm(mid_p - left_p) + np.linalg.norm(right_p - mid_p))
    if depth <= 0 or poly - chord <= tol_mm:
        samples.append((mid_t, mid_p))
        samples.append((right_t, right_p))
        return
    _subdivide_arc_eval(fit, left_t, mid_t, left_p, mid_p, tol_mm, depth - 1, samples)
    _subdivide_arc_eval(fit, mid_t, right_t, mid_p, right_p, tol_mm, depth - 1, samples)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _emit_progress(progress_fn: ProgressFn | None, message: str) -> None:
    if progress_fn is not None:
        progress_fn(message)
