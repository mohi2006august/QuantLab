"""Quantization formats and their approximate effective storage costs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantFormat:
    name: str
    bits_per_parameter: float
    family: str
    note: str


FORMATS = (
    QuantFormat("Q2_K", 2.6, "GGUF", "Very small; quality can drop substantially."),
    QuantFormat("Q3_K_M", 3.9, "GGUF", "Compact mixed-bit GGUF format."),
    QuantFormat("Q4_K_M", 4.8, "GGUF", "Common quality and size trade-off."),
    QuantFormat("Q5_K_M", 5.7, "GGUF", "Higher quality with a larger footprint."),
    QuantFormat("Q6_K", 6.6, "GGUF", "Near-high precision GGUF quantization."),
    QuantFormat("Q8_0", 8.5, "GGUF", "High-quality 8-bit GGUF quantization."),
    QuantFormat("FP16", 16.0, "full precision", "Half-precision weights."),
)


def get_format(name: str) -> QuantFormat:
    normalized = name.casefold()
    for quant_format in FORMATS:
        if quant_format.name.casefold() == normalized:
            return quant_format
    choices = ", ".join(item.name for item in FORMATS)
    raise ValueError(f"Unknown format {name!r}. Choose from: {choices}.")
