"""Command-line interface for QuantLab."""

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

from quantlab import __version__
from quantlab.devices import DEVICES, get_device
from quantlab.config import load_run_config
from quantlab.estimator import estimate_memory, parse_parameter_count
from quantlab.formats import FORMATS, get_format
from quantlab.recommend import recommend_formats
from quantlab.orchestrate import parse_methods, run_matrix
from quantlab.benchmark import benchmark_gguf, benchmark_vllm
from quantlab.evaluate import evaluate_perplexity, evaluate_prompt_suite, evaluate_tasks
from quantlab.publish import publish_model
from quantlab.report import write_html_report
from quantlab.sensitivity import analyze_sensitivity, write_sensitivity_report
from quantlab.storage import RunStore, validate_run_id


def _parameter_count(args: argparse.Namespace) -> float:
    return args.parameters_b if args.parameters_b is not None else parse_parameter_count(args.model)


def _memory_budget(args: argparse.Namespace) -> float:
    memory_gb = args.device_memory_gb
    if memory_gb is None:
        memory_gb = get_device(args.device).memory_gb
    if not math.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("Device memory must be greater than zero.")
    return memory_gb * 1_000_000_000 / 1024**3


def _add_model_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--model", help="Model name containing its size, such as Qwen2.5-14B")
    group.add_argument("--parameters-b", type=float, help="Model parameter count in billions")


def _add_device_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
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
    _add_model_args(recommend, required=False)
    _add_device_args(recommend, required=False)
    recommend.add_argument("--run-id", help="Choose from measured results in an existing run")
    recommend.add_argument("--results-db", help="SQLite database created by quantlab run")
    recommend.add_argument("--min-tok-s", type=float, help="Minimum measured generation throughput")
    recommend.add_argument("--max-perplexity", type=float, help="Maximum measured perplexity")
    recommend.add_argument("--min-custom-score", type=float, help="Minimum measured prompt exact-match score (0..1)")
    recommend.set_defaults(handler=_recommend)

    run = commands.add_parser("run", help="Run a resumable quantization matrix")
    run.add_argument("--config", help="JSON or YAML run configuration")
    run.add_argument("--model", help="Local model directory, GGUF file, or Hugging Face model ID")
    run.add_argument("--methods", help="Comma-separated method:variant matrix, e.g. gguf:Q4_K_M,bnb:4")
    run.add_argument("--device", help="Device profile label saved with the run")
    run.add_argument("--output-dir", help="Run and SQLite results directory")
    run.add_argument("--eval-text", help="UTF-8 corpus file for perplexity evaluation")
    run.add_argument("--prompt-suite", help="JSON custom prompt suite")
    run.add_argument("--judge-model", help="Optional local/Hugging Face model used to score expected answers")
    run.add_argument("--tasks", help="Comma-separated lm-evaluation-harness tasks, e.g. mmlu,gsm8k")
    run.add_argument("--calibration-file", help="Calibration text file or JSON/JSONL samples for GPTQ and AWQ")
    run.add_argument("--benchmark", choices=("llama.cpp", "vllm"), help="Optional throughput benchmark backend")
    run.add_argument("--kv-cache-dtype", default="auto", choices=("auto", "f16", "q8_0", "q5_0", "q4_0", "fp8", "fp8_e4m3", "fp8_e5m2"))
    run.add_argument("--batch-size", type=int)
    run.add_argument("--prompt-tokens", type=int)
    run.add_argument("--generation-tokens", type=int)
    run.add_argument("--repeats", type=int)
    run.add_argument("--seed", type=int, help="Seed for task benchmarks (default: 1234)")
    run.add_argument("--resume", metavar="RUN_ID", help="Resume a run ID from the selected output directory")
    run.set_defaults(handler=_run)

    evaluate = commands.add_parser("evaluate", help="Evaluate perplexity or a custom prompt suite")
    evaluate.add_argument("--model-path", required=True, help="Local model or quantized artifact path")
    evaluate.add_argument("--text-file", help="UTF-8 evaluation corpus for perplexity")
    evaluate.add_argument("--prompt-suite", help="JSON custom prompt suite")
    evaluate.add_argument("--judge-model", help="Optional local/Hugging Face model for reference-based judging")
    evaluate.add_argument("--tasks", help="Comma-separated lm-evaluation-harness tasks")
    evaluate.add_argument("--limit", type=int, help="Limit evaluation examples for a smoke-sized benchmark")
    evaluate.add_argument("--seed", type=int, default=1234)
    evaluate.add_argument("--output", help="Write full evaluation results to JSON")
    evaluate.add_argument("--stride", type=int, default=512)
    evaluate.add_argument("--max-length", type=int)
    evaluate.set_defaults(handler=_evaluate)

    benchmark = commands.add_parser("benchmark", help="Measure generation throughput")
    benchmark.add_argument("--model-path", required=True)
    benchmark.add_argument("--backend", choices=("llama.cpp", "vllm"), default="llama.cpp")
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--prompt-tokens", type=int, default=512)
    benchmark.add_argument("--generation-tokens", type=int, default=128)
    benchmark.add_argument("--batch-size", type=int, help="Concurrent requests (default: 2048 llama.cpp / 1 vLLM)")
    benchmark.add_argument("--kv-cache-dtype", default="f16", choices=("auto", "f16", "q8_0", "q5_0", "q4_0", "fp8", "fp8_e4m3", "fp8_e5m2"))
    benchmark.add_argument("--output", help="Write benchmark metrics to JSON")
    benchmark.set_defaults(handler=_benchmark)

    report = commands.add_parser("report", help="Build a Pareto HTML report for a run")
    report.add_argument("run_id")
    report.add_argument("--output-dir", default=os.environ.get("QL_CACHE_DIR", "quantlab-runs"))
    report.add_argument("--output")
    report.set_defaults(handler=_report)

    sensitivity = commands.add_parser("sensitivity", help="Rank layers by simulated quantization error")
    sensitivity.add_argument("--model-path", required=True)
    sensitivity.add_argument("--bits", default="2,3,4,8")
    sensitivity.add_argument("--group-size", type=int, default=128)
    sensitivity.add_argument("--output", default="sensitivity.json")
    sensitivity.set_defaults(handler=_sensitivity)

    publish = commands.add_parser("publish", help="Publish a local artifact to Hugging Face Hub")
    publish.add_argument("--model-path", required=True)
    publish.add_argument("--repo-id", required=True, help="Hugging Face repository ID, e.g. user/model-q4")
    publish.add_argument("--license", required=True, help="SPDX license identifier for the source model")
    publish.add_argument("--private", action="store_true")
    publish.set_defaults(handler=_publish)
    return parser


def _doctor(_: argparse.Namespace) -> int:
    print(f"QuantLab {__version__}")
    print(f"Python {sys.version.split()[0]}")
    for tool in ("llama-cli", "llama-quantize", "llama-perplexity", "llama-bench", "llama-server", "nvidia-smi"):
        location = shutil.which(tool)
        status = location if location else "not found (optional)"
        print(f"{tool}: {status}")
    for package in ("torch", "transformers", "bitsandbytes", "gptqmodel", "awq", "llmcompressor", "vllm", "lm_eval", "huggingface_hub", "psutil"):
        status = "available" if importlib.util.find_spec(package) else "not installed (optional)"
        print(f"{package}: {status}")
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
    if args.run_id:
        validate_run_id(args.run_id)
        if not args.results_db:
            raise ValueError("Measured recommendations require --results-db.")
        if args.device is None and args.device_memory_gb is None:
            raise ValueError("Measured recommendations require --device or --device-memory-gb.")
        budget = _memory_budget(args)
        available_budget = budget - args.runtime_reserve_gib
        if available_budget <= 0:
            raise ValueError("Runtime reserve must be smaller than device memory.")
        if args.min_tok_s is not None and args.min_tok_s < 0:
            raise ValueError("--min-tok-s cannot be negative.")
        if args.max_perplexity is not None and args.max_perplexity <= 0:
            raise ValueError("--max-perplexity must be greater than zero.")
        if args.min_custom_score is not None and not 0 <= args.min_custom_score <= 1:
            raise ValueError("--min-custom-score must be between zero and one.")
        store = RunStore(args.results_db)
        if not store.run_exists(args.run_id):
            raise ValueError(f"Run {args.run_id!r} was not found in {args.results_db}.")
        rows = [
            row for row in store.results(args.run_id)
            if row["status"] == "completed"
            and row.get("size_gib") is not None
            and row["size_gib"] <= available_budget
            and (args.min_tok_s is None or (row.get("tok_s") is not None and row["tok_s"] >= args.min_tok_s))
            and (args.max_perplexity is None or (row.get("perplexity") is not None and row["perplexity"] <= args.max_perplexity))
            and (args.min_custom_score is None or (row.get("custom_score") is not None and row["custom_score"] >= args.min_custom_score))
        ]
        if not rows:
            print("No measured configuration meets the memory and throughput constraints.")
            return 1
        rows.sort(key=lambda row: (
            row.get("custom_score") is None,
            -(row.get("custom_score") or 0),
            row.get("perplexity") is None,
            row.get("perplexity") or float("inf"),
            row.get("size_gib") or float("inf"),
            -(row.get("tok_s") or 0),
        ))
        winner = rows[0]
        print(f"Measured recommendation: {winner['method']}:{winner['variant']}")
        print(f"Artifact: {winner['artifact_path']}")
        print(f"Size: {winner['size_gib']:.2f} GiB | perplexity: {winner.get('perplexity') or 'not measured'} | throughput: {winner.get('tok_s') or 'not measured'} tok/s")
        return 0
    if args.model is None and args.parameters_b is None:
        raise ValueError("Provide --model or --parameters-b, or select measured results with --run-id.")
    if args.device is None and args.device_memory_gb is None:
        raise ValueError("Provide --device or --device-memory-gb.")
    if args.min_tok_s is not None:
        raise ValueError("--min-tok-s can only be used with a measured --run-id.")
    if args.max_perplexity is not None or args.min_custom_score is not None:
        raise ValueError("Measured quality constraints require --run-id and --results-db.")
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


def _run(args: argparse.Namespace) -> int:
    config = load_run_config(args.config) if args.config else {}
    model = args.model or config.get("model")
    methods_spec = args.methods or config.get("methods")
    if not model or not methods_spec:
        raise ValueError("Provide --model and --methods, or set both in --config.")
    methods = parse_methods(methods_spec)
    repeats = args.repeats if args.repeats is not None else config.get("repeats", 3)
    if repeats < 1:
        raise ValueError("Benchmark repeats must be at least one.")
    benchmark_backend = args.benchmark or config.get("benchmark")
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 2048 if benchmark_backend == "llama.cpp" else 1)
    prompt_tokens = args.prompt_tokens if args.prompt_tokens is not None else config.get("prompt_tokens", 512)
    generation_tokens = args.generation_tokens if args.generation_tokens is not None else config.get("generation_tokens", 128)
    if batch_size < 1 or prompt_tokens < 1 or generation_tokens < 1:
        raise ValueError("Batch size and prompt/generation token counts must be at least one.")
    output_root = Path(args.output_dir or config.get("output_dir") or os.environ.get("QL_CACHE_DIR", "quantlab-runs")).expanduser()
    store = RunStore(output_root / "results.sqlite")
    if args.resume and not store.run_exists(args.resume):
        raise ValueError(f"Run {args.resume!r} was not found in {output_root}.")
    run_id, rows = run_matrix(
        model=model,
        methods=methods,
        device=args.device or config.get("device", "unknown"),
        output_root=output_root,
        eval_text=args.eval_text or config.get("eval_text"),
        prompt_suite=args.prompt_suite or config.get("prompt_suite"),
        judge_model=args.judge_model or config.get("judge_model"),
        calibration_file=args.calibration_file or config.get("calibration_file"),
        tasks=[task.strip() for task in (args.tasks or config.get("tasks") or "").split(",") if task.strip()] or None,
        benchmark_backend=benchmark_backend,
        kv_cache_dtype=args.kv_cache_dtype if args.kv_cache_dtype != "auto" else config.get("kv_cache_dtype", "auto"),
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        repeats=repeats,
        seed=args.seed if args.seed is not None else config.get("seed", 1234),
        run_id=args.resume,
    )
    report_path = write_html_report(run_id, rows, output_root / f"{run_id}.html")
    print(f"Run ID: {run_id}")
    print(f"Report: {report_path}")
    print(f"Results database: {output_root / 'results.sqlite'}")
    return 1 if any(row["status"] == "failed" for row in rows) else 0


def _evaluate(args: argparse.Namespace) -> int:
    if not args.text_file and not args.prompt_suite and not args.tasks:
        raise ValueError("Provide --text-file, --prompt-suite, --tasks, or a combination.")
    result = {}
    if args.text_file:
        result["perplexity"] = evaluate_perplexity(args.model_path, args.text_file, args.stride, args.max_length)
    if args.prompt_suite:
        result["custom"] = evaluate_prompt_suite(args.model_path, args.prompt_suite, judge_model_path=args.judge_model)
    if args.tasks:
        result["tasks"] = evaluate_tasks(args.model_path, [task.strip() for task in args.tasks.split(",")], args.limit, args.seed)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("Benchmark batch size must be at least one.")
    benchmark_fn = benchmark_gguf if args.backend == "llama.cpp" else benchmark_vllm
    if args.backend == "llama.cpp":
        cache_dtype = "f16" if args.kv_cache_dtype == "auto" else args.kv_cache_dtype
        metrics = benchmark_fn(
            args.model_path, args.repeats, args.prompt_tokens, args.generation_tokens,
            cache_dtype, args.batch_size or 2048,
        )
    else:
        metrics = benchmark_fn(
            args.model_path, args.repeats, args.prompt_tokens, args.generation_tokens,
            args.kv_cache_dtype, args.batch_size or 1,
        )
    metrics["batch_size"] = args.batch_size or (2048 if args.backend == "llama.cpp" else 1)
    metrics["prompt_tokens"] = args.prompt_tokens
    metrics["generation_tokens"] = args.generation_tokens
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


def _report(args: argparse.Namespace) -> int:
    validate_run_id(args.run_id)
    output_root = Path(args.output_dir).expanduser()
    store = RunStore(output_root / "results.sqlite")
    if not store.run_exists(args.run_id):
        raise ValueError(f"Run {args.run_id!r} was not found in {output_root}.")
    path = args.output or output_root / f"{args.run_id}.html"
    print(write_html_report(args.run_id, store.results(args.run_id), path))
    return 0


def _sensitivity(args: argparse.Namespace) -> int:
    bits = tuple(int(value.strip()) for value in args.bits.split(",") if value.strip())
    result = analyze_sensitivity(args.model_path, bits, args.group_size)
    path = write_sensitivity_report(result, args.output)
    print(f"Sensitivity report: {path} ({len(result['layers'])} layers)")
    return 0


def _publish(args: argparse.Namespace) -> int:
    url = publish_model(args.model_path, args.repo_id, args.license, args.private)
    print(url)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 2
