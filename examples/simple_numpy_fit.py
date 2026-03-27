from __future__ import annotations

import numpy as np

import PEPTSpline


def main() -> None:
    # Columns: time, x, y, z, error
    t = np.linspace(0.0, 2.0, 60)
    x = 20.0 * np.cos(2.0 * np.pi * t / 2.0)
    y = 20.0 * np.sin(2.0 * np.pi * t / 2.0)
    z = 5.0 * t
    error = np.full_like(t, 0.4)

    data = np.column_stack([t, x, y, z, error])

    result = PEPTSpline.fit(
        data,
        dt_out=0.05,
        fit_window_duration=0.5,
        fit_window_overlap=0.5,
    )

    print("Diagnostics keys:", sorted(result.diagnostics.keys()))
    print("Sample evaluation:\n", result.evaluate([0.25, 0.75, 1.25]))
    print("Uniform-time output head:\n", result.resample_time().head())


if __name__ == "__main__":
    main()
