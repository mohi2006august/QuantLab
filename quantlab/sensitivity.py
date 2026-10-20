"""Per-layer simulated quantization error ranking."""

from pathlib import Path
from typing import Any


def analyze_sensitivity(model_path: str, bits: tuple[int, ...] = (2, 3, 4, 8), group_size: int = 128) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("Layer sensitivity analysis requires torch and transformers. Install QuantLab with the eval extra.") from error
    if group_size < 1 or not bits or any(bit < 2 or bit > 8 for bit in bits):
        raise ValueError("Specify at least one bit width from 2 to 8 and a positive group size.")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cpu")
    layers = []
    quant_ranges = {bit: (1 << (bit - 1)) - 1 for bit in bits}
    with torch.inference_mode():
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point():
                continue
            flat = weight.detach().float().flatten()
            errors = {}
            for bit, quant_max in quant_ranges.items():
                total_squared_error = 0.0
                total_elements = 0
                for start in range(0, flat.numel(), group_size):
                    group = flat[start:start + group_size]
                    scale = group.abs().max() / quant_max
                    if scale == 0:
                        continue
                    quantized = torch.round(group / scale).clamp(-quant_max, quant_max) * scale
                    total_squared_error += float(torch.sum((group - quantized) ** 2))
                    total_elements += group.numel()
                errors[str(bit)] = total_squared_error / max(total_elements, 1)
            layers.append({"name": name, "shape": list(weight.shape), "errors": errors})
    layers.sort(key=lambda item: item["errors"].get("4", 0), reverse=True)
    suggestions = [
        {"layer": row["name"], "recommended_bits": max(bits), "reason": "highest simulated quantization error"}
        for row in layers[:max(1, len(layers) // 10)]
    ]
    return {"model": str(model_path), "group_size": group_size, "layers": layers, "mixed_precision_suggestions": suggestions}


def write_sensitivity_report(result: dict[str, Any], output_path: str | Path) -> Path:
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path
