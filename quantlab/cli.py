"""Command-line interface for QuantLab."""

import argparse
import shutil
import sys

from quantlab import __version__
from quantlab.devices import DEVICES, get_device
from quantlab.estimator import estimate_memory, parse_parameter_count
from quantlab.formats import FORMATS, get_format
from quantlab.recommend import recommend_formats


def _parameter_count(args: argparse.Namespace) -> float:
    return args.parameters_b if args.parameters_b is not None else parse_parameter_count(args.model)


def _memory_budget(args: argparse.Namespace) -> float:
    memory_gb = args.device_memory_gb
    if memory_gb is None:
        memory_gb = get_device(args.device).memory_gb
    if memory_gb <= 0:
        raise ValueError("Device memory must be greater than zero.")
    return memory_gb * 1_000_000_000 / 1024**3


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model name containing its size, such as Qwen2.5-14B")
    group.add_argument("--parameters-b", type=float, help="Model parameter count in billions")


def _add_device_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--device", choices=sorted(DEVICES), help="Built-in device profile")
    group.add_argument("--device-memory-gb", type=float, help="Device memory budget in GB")
    parser.add_argument("--runtime-reserve-gib", type=float, default=1.5, help="Memory reserved for runtime overhead (default: 1.5 GiB)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab", description="Plan LLM quantization for local hardware.")
    parser.add_argument("--version", action="version", version=f"quantlab {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Check Python and optional inference tools")
    doctor.set_defaults(handler=_doctor)

    estimate = commands.add_parser("estimate", help="Estimate memory for one quantization format")
    _add_model_args(estimate)
    estimate.add_argument("--format", required=True, choices=[item.name for item in FORMATS])
    _add_device_args(estimate)
    estimate.set_defaults(handler=_estimate)

    recommend = commands.add_parser("recommend", help="Compare quantization formats for a device")
    _add_model_args(recommend)
    _add_device_args(recommend)
    recommend.set_defaults(handler=_recommend)
    return parser


def _doctor(_: argparse.Namespace) -> int:
    print(f"QuantLab {__version__}")
    print(f"Python {sys.version.split()[0]}")
    for tool in ("llama-cli", "llama-quantize", "llama-server", "nvidia-smi"):
        location = shutil.which(tool)
        status = location if location else "not found (optional)"
        print(f"{tool}: {status}")
    return 0


def _estimate(args: argparse.Namespace) -> int:
    parameters_b = _parameter_count(args)
    estimate = estimate_memory(parameters_b, get_format(args.format), args.runtime_reserve_gib)
    budget = _memory_budget(args)
    print(f"Model parameters: {parameters_b:g}B")
    print(f"Format: {args.format}")
    print(f"Estimated weights: {estimate.weights_gib:.2f} GiB")
    print(f"Runtime reserve: {estimate.runtime_reserve_gib:.2f} GiB")
    print(f"Estimated minimum: {estimate.total_gib:.2f} GiB")
    print(f"Device memory: {budget:.2f} GiB")
    print(f"Fits estimate: {'yes' if estimate.total_gib <= budget else 'no'}")
    print("Estimate excludes KV cache and runtime-specific allocations.")
    return 0


def _recommend(args: argparse.Namespace) -> int:
    parameters_b = _parameter_count(args)
    budget = _memory_budget(args)
    recommendations = recommend_formats(parameters_b, budget, args.runtime_reserve_gib)
    print(f"Model: {args.model or f'{parameters_b:g}B parameters'}")
    print(f"Device memory: {budget:.2f} GiB | runtime reserve: {args.runtime_reserve_gib:.2f} GiB")
    print(f"{'FORMAT':<12} {'WEIGHTS':>10} {'TOTAL':>10} {'STATUS':>10}  NOTES")
    best_fit = max(
        (row.quant_format.bits_per_parameter for row in recommendations if row.fits),
        default=None,
    )
    for row in recommendations:
        marker = "recommended" if row.fits and row.quant_format.bits_per_parameter == best_fit else "fits" if row.fits else "too large"
        print(
            f"{row.quant_format.name:<12} {row.estimate.weights_gib:>8.2f} GiB "
            f"{row.estimate.total_gib:>8.2f} GiB {marker:>10}  {row.quant_format.note}"
        )
    print("Estimates exclude KV cache; verify fit with your inference runtime.")
    if not any(row.fits for row in recommendations):
        print("No listed format fits this estimate. Try a smaller model or larger device.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except ValueError as error:
        parser.error(str(error))
    return 2
