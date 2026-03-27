# PEPTSpline

Small Python wrapper around the trajectory denoiser in this repository.

## Install

From the repository root:

```bash
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

Example:

```python
import numpy as np
import PEPTSpline

data = np.array([
    [0.0,  0.0, 0.0, 0.0, 0.5],
    [0.1,  1.0, 0.2, 0.0, 0.5],
    [0.2,  2.0, 0.1, 0.1, 0.5],
    [0.3,  3.1, 0.0, 0.2, 0.5],
])

result = PEPTSpline.fit(
    data,
    fit_window_duration=0.2,
    fit_window_overlap=0.5,
)

xyz = result.evaluate([0.05, 0.15, 0.25])
vel = result.velocity([0.05, 0.15, 0.25])
uniform_t = result.resample_time(0.05)
uniform_s = result.resample_space(1.0)
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
