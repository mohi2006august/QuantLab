"""Rank candidate formats by fit and estimated precision."""

from dataclasses import dataclass

from quantlab.estimator import MemoryEstimate, estimate_memory
from quantlab.formats import FORMATS, QuantFormat


@dataclass(frozen=True)
class Recommendation:
    quant_format: QuantFormat
    estimate: MemoryEstimate
    fits: bool


def recommend_formats(
    parameters_b: float,
    device_memory_gib: float,
    runtime_reserve_gib: float = 1.5,
) -> list[Recommendation]:
    if device_memory_gib <= 0:
        raise ValueError("Device memory must be greater than zero.")
    rows = []
    for item in FORMATS:
        estimate = estimate_memory(parameters_b, item, runtime_reserve_gib)
        rows.append(
            Recommendation(
                quant_format=item,
                estimate=estimate,
                fits=estimate.total_gib <= device_memory_gib,
            )
        )
    return sorted(rows, key=lambda row: row.quant_format.bits_per_parameter)
