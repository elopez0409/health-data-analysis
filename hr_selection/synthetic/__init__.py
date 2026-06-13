"""Synthetic data generation in the exact real on-disk formats."""

from hr_selection.synthetic.calibration import (
    HRV_METRIC,
    HR_METRIC,
    MetricConfig,
    build_paired_nights,
    generate_calibration_data,
    write_calibration_dataset,
)
from hr_selection.synthetic.generate import generate_all

__all__ = [
    "generate_all",
    "generate_calibration_data",
    "write_calibration_dataset",
    "build_paired_nights",
    "MetricConfig",
    "HR_METRIC",
    "HRV_METRIC",
]
