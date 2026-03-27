from __future__ import annotations

import numpy as np

import PEPTSpline


def test_fit_accepts_five_column_numpy_array():
    times = np.linspace(0.0, 1.0, 30)
    data = np.column_stack(
        [
            times,
            10.0 * times,
            np.sin(times),
            np.cos(times),
            np.full_like(times, 0.5),
        ]
    )

    result = PEPTSpline.fit(data, dt_out=0.05)
    values = result.evaluate([0.1, 0.5, 0.9])
    uniform = result.resample_time(0.05)

    assert values.shape == (3, 3)
    assert list(uniform.columns) == ["time", "x", "y", "z"]
    assert len(result.segments) >= 1


def test_fit_injects_default_error_for_four_column_numpy_array():
    times = np.linspace(0.0, 1.0, 25)
    data = np.column_stack(
        [
            times,
            5.0 * times,
            np.zeros_like(times),
            2.0 * times,
        ]
    )

    result = PEPTSpline.fit(data, default_error=0.7, dt_out=0.1)
    uniform = result.resample_time(0.1)

    assert result.preprocessing["fallback_sigma"] > 0.0
    assert len(uniform) > 5
