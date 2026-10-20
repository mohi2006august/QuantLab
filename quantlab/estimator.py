"""Transparent first-pass model weight memory estimates."""

from dataclasses import dataclass
import math

from quantlab.formats import QuantFormat


GIB_BYTES = 1024**3
GB_BYTES = 1_000_000_000


@dataclass(frozen=True)
class MemoryEstimate:
    weights_gib: float
    runtime_reserve_gib: float
    total_gib: float


def estimate_memory(
    parameters_b: float,
    quant_format: QuantFormat,
    runtime_reserve_gib: float = 1.5,
) -> MemoryEstimate:
    if not math.isfinite(parameters_b) or parameters_b <= 0:
        raise ValueError("Parameter count must be greater than zero.")
    if not math.isfinite(runtime_reserve_gib) or runtime_reserve_gib < 0:
        raise ValueError("Runtime reserve cannot be negative.")

    weight_bytes = parameters_b * GB_BYTES * quant_format.bits_per_parameter / 8
    weights_gib = weight_bytes / GIB_BYTES
    return MemoryEstimate(
        weights_gib=weights_gib,
        runtime_reserve_gib=runtime_reserve_gib,
        total_gib=weights_gib + runtime_reserve_gib,
    )


def parse_parameter_count(model: str) -> float:
    """Infer a parameter count in billions from common model-name patterns."""
    import re

    matches = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[bB](?!\w)", model)
    if not matches:
        raise ValueError(
            f"Could not infer parameter count from {model!r}; pass --parameters-b instead."
        )
    return float(matches[-1])
