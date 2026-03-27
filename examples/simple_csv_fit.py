from __future__ import annotations

from pathlib import Path

import PEPTSpline


def main() -> None:
    csv_path = Path("birmingham_points.csv")
    if not csv_path.exists():
        raise SystemExit("Run this example from the repository root so birmingham_points.csv is available.")

    result = PEPTSpline.fit_csv(
        csv_path,
        dt_out=20.0,
        fit_window_duration=200.0,
        fit_window_overlap=0.6,
    )

    print("Segments:", len(result.segments))
    print("Reduced chi-square:", result.diagnostics.get("reduced_chi_square"))
    print(result.resample_time(20.0).head())


if __name__ == "__main__":
    main()
