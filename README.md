# PEPTSpline

A standalone Python package for robust spline-based PEPT trajectory fitting.

## Install

From inside this folder:

```bash
cd PEPTSpline
pip install -e .
```

Then in Python:

```python
import PEPTSpline
```

## Simple usage

For NumPy input, pass either:

- an `N x 5` array with columns `time, x, y, z, error`
- or an `N x 4` array with columns `time, x, y, z`

For PEPT data, these columns mean:

- `time`: sample time in your native input units, for example `ms` or `s`
- `x, y, z`: particle position, typically in `mm`
- `error`: per-sample positional uncertainty, typically in the same spatial unit as `x, y, z`

Important:

- `fit_window_duration` is a duration in the same units as your `time` column
- `fit_window_overlap` is a fraction in `[0, 1)`, not a time
- `dt_out` is also in the same units as your `time` column
- `ds_mm` is a spatial resampling distance, typically in `mm`

Example:

```python
import numpy as np
import PEPTSpline

# Example PEPT-style rows:
# [time_ms, x_mm, y_mm, z_mm, error_mm]
data = np.array([
    [0.0,   0.0,  0.0,  0.0, 0.5],
    [20.0,  1.0,  0.2,  0.0, 0.5],
    [40.0,  2.0,  0.1,  0.1, 0.5],
    [60.0,  3.1,  0.0,  0.2, 0.5],
])

result = PEPTSpline.fit(
    data,
    dt_out=20.0,               # resample every 20 ms
    fit_window_duration=200.0, # each local spline sees a 200 ms time window
    fit_window_overlap=0.5,    # consecutive windows overlap by 50%
)

xyz = result.evaluate([10.0, 30.0, 50.0])   # query positions at 10, 30, 50 ms
vel = result.velocity([10.0, 30.0, 50.0])   # velocity in mm per time-unit
uniform_t = result.resample_time(20.0)       # uniform-time output, every 20 ms
uniform_s = result.resample_space(1.0)       # uniform-space output, every 1 mm
```

## Public API

- `PEPTSpline.fit(data, **kwargs)`
- `PEPTSpline.fit_csv(path, **kwargs)`

The returned `FitResult` exposes:

- `evaluate(time)`
- `velocity(time)`
- `resample_time(dt_out=None)`
- `resample_space(ds_mm)`
- `diagnostics`

## Layout

This folder is a self-contained Python project:

- `pyproject.toml`
- `PEPTSpline/`
- `examples/`
- `tests/`

It does not import code from the repository root package.
