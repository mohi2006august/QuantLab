"""Per-layer quantization error ranking with optional activation weighting."""

from pathlib import Path
from typing import Any


def analyze_sensitivity(
    model_path: str,
    bits: tuple[int, ...] = (2, 3, 4, 8),
    group_size: int = 128,
    calibration_file: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Layer sensitivity analysis requires torch and transformers. Install QuantLab with the eval extra.") from error
    if group_size < 1 or not bits or any(bit < 2 or bit > 8 for bit in bits):
        raise ValueError("Specify at least one bit width from 2 to 8 and a positive group size.")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cpu")
    activation_moments = {}
    if calibration_file:
        from quantlab.runner import _read_calibration_data

        calibration_text = _read_calibration_data(calibration_file)
        if not calibration_text:
            raise ValueError("Calibration file contains no text samples.")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                activation_moments[name] = [None, 0]

                def collect_activation(current_module, inputs, layer_name=name):
                    if not inputs or not isinstance(inputs[0], torch.Tensor):
                        return
                    values = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
                    moments, count = activation_moments[layer_name]
                    current = values.square().sum(dim=0).cpu()
                    activation_moments[layer_name] = [current if moments is None else moments + current, count + values.shape[0]]

                handles.append(module.register_forward_pre_hook(collect_activation))
        try:
            model.eval()
            with torch.inference_mode():
                for sample in calibration_text[:512]:
                    tokens = tokenizer(sample, return_tensors="pt", truncation=True, max_length=2048)
                    if tokens.input_ids.shape[1]:
                        model(**tokens)
        finally:
            for handle in handles:
                handle.remove()
    layers = []
    quant_ranges = {bit: (1 << (bit - 1)) - 1 for bit in bits}
    with torch.inference_mode():
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point():
                continue
            matrix = weight.detach().float()
            errors = {}
            for bit, quant_max in quant_ranges.items():
                total_squared_error = 0.0
                total_elements = 0
                moments, observation_count = activation_moments.get(name, (None, 0))
                activation_mean_square = moments / observation_count if moments is not None and observation_count else None
                for start in range(0, matrix.shape[1], group_size):
                    group = matrix[:, start:start + group_size]
                    scale = group.abs().amax(dim=1, keepdim=True) / quant_max
                    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                    quantized = torch.round(group / scale).clamp(-quant_max, quant_max) * scale
                    squared_error = (group - quantized).square()
                    if activation_mean_square is not None:
                        squared_error = squared_error * activation_mean_square[start:start + group.shape[1]].unsqueeze(0)
                    total_squared_error += float(torch.sum(squared_error))
                    total_elements += group.numel()
                errors[str(bit)] = total_squared_error / max(total_elements, 1)
            layers.append({
                "name": name,
                "shape": list(weight.shape),
                "errors": errors,
                "sensitivity_method": "activation_weighted" if name in activation_moments else "weight_only",
            })
    layers.sort(key=lambda item: item["errors"].get("4", 0), reverse=True)
    suggestions = [
        {"layer": row["name"], "recommended_bits": max(bits), "reason": "highest simulated quantization error"}
        for row in layers[:max(1, len(layers) // 10)]
    ]
    return {
        "model": str(model_path),
        "group_size": group_size,
        "calibration_file": calibration_file,
        "method": "activation_weighted_diagonal" if calibration_file else "weight_only_simulation",
        "layers": layers,
        "mixed_precision_suggestions": suggestions,
    }


def write_sensitivity_report(result: dict[str, Any], output_path: str | Path) -> Path:
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path
