"""Resumable method matrix orchestration."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantlab.benchmark import benchmark_gguf, benchmark_vllm
from quantlab.evaluate import evaluate_perplexity, evaluate_prompt_suite, evaluate_tasks
from quantlab.runner import quantize_model
from quantlab.storage import RunStore, validate_run_id


def parse_methods(specification: str) -> list[tuple[str, str]]:
    supported = {"gguf", "bnb", "gptq", "awq", "fp8"}
    methods = []
    for item in specification.split(","):
        if ":" not in item:
            raise ValueError(f"Invalid method {item!r}; use method:variant, for example gguf:Q4_K_M.")
        method, variant = (part.strip() for part in item.split(":", 1))
        method = method.casefold()
        if not method or not variant:
            raise ValueError(f"Invalid method {item!r}; method and variant are both required.")
        if method not in supported:
            raise ValueError(f"Unsupported method {method!r}; choose from {', '.join(sorted(supported))}.")
        if method == "gguf":
            variant = variant.upper()
        elif method in {"bnb", "fp8"}:
            variant = variant.casefold()
        methods.append((method, variant))
    if not methods:
        raise ValueError("Specify at least one method.")
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate method and variant entries are not allowed.")
    return methods


def run_matrix(
    model: str,
    methods: list[tuple[str, str]],
    device: str,
    output_root: str | Path,
    eval_text: str | None = None,
    prompt_suite: str | None = None,
    judge_model: str | None = None,
    calibration_file: str | None = None,
    tasks: list[str] | None = None,
    benchmark_backend: str | None = None,
    kv_cache_dtype: str = "auto",
    batch_size: int = 1,
    prompt_tokens: int = 512,
    generation_tokens: int = 128,
    repeats: int = 3,
    seed: int = 1234,
    run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    database = output_root / "results.sqlite"
    store = RunStore(database)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    validate_run_id(run_id)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model,
        "device": device,
        "methods": methods,
        "eval_text": eval_text,
        "prompt_suite": prompt_suite,
        "judge_model": judge_model,
        "calibration_file": calibration_file,
        "tasks": tasks,
        "benchmark_backend": benchmark_backend,
        "kv_cache_dtype": kv_cache_dtype,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "repeats": repeats,
        "seed": seed,
    }
    if store.run_exists(run_id):
        existing = store.results(run_id)
        configured = next((row for row in existing), None)
        if configured:
            print(f"Resuming run {run_id}; completed artifacts will be reused where available.")
    else:
        store.create_run(run_id, model, device, config)

    for method, variant in methods:
        previous = store.get_result(run_id, method, variant)
        if previous and previous["status"] == "completed" and previous.get("artifact_path") and Path(previous["artifact_path"]).exists():
            print(f"Skip {method}:{variant} (already complete)")
            continue
        result: dict[str, Any] = dict(previous or {})
        result.update({
            "run_id": run_id,
            "method": method,
            "variant": variant,
            "status": "running",
            "error": None,
            "size_gib": None,
            "perplexity": None,
            "custom_score": None,
            "task_scores_json": None,
            "kv_cache_dtype": None,
            "batch_size": None,
            "tok_s": None,
            "tok_s_std": None,
            "prefill_tok_s": None,
            "prefill_ms_estimate": None,
            "ttft_ms": None,
            "ttft_ms_std": None,
            "peak_mem_gib": None,
        })
        store.save_result(result)
        artifact_dir = run_dir / f"{method}-{variant.replace('/', '-') }"
        try:
            cached_path = Path(result["artifact_path"]) if result.get("artifact_path") else None
            if cached_path and cached_path.exists():
                artifact_path = cached_path
                print(f"Reusing artifact for {method}:{variant}; resuming evaluation stages.")
            else:
                print(f"Quantizing {method}:{variant} …")
                artifact_path = quantize_model(model, method, variant, artifact_dir, calibration_file)
            result["artifact_path"] = str(artifact_path)
            result["size_gib"] = _artifact_size(artifact_path)
            if eval_text:
                result.update(evaluate_perplexity(str(artifact_path), eval_text))
            if prompt_suite:
                prompt_results = evaluate_prompt_suite(str(artifact_path), prompt_suite, judge_model_path=judge_model)
                result["custom_score"] = prompt_results["judge_score"] if prompt_results["judge_score"] is not None else prompt_results["exact_match_score"]
                (artifact_dir / "prompt-results.json").write_text(json.dumps(prompt_results, indent=2), encoding="utf-8")
            if tasks:
                task_results = evaluate_tasks(str(artifact_path), tasks, seed=seed)
                result["task_scores_json"] = json.dumps(task_results["results"], sort_keys=True)
            if benchmark_backend:
                if benchmark_backend == "llama.cpp":
                    llama_cache_dtype = "f16" if kv_cache_dtype == "auto" else kv_cache_dtype
                    metrics = benchmark_gguf(
                        artifact_path,
                        repeats=repeats,
                        prompt_tokens=prompt_tokens,
                        generation_tokens=generation_tokens,
                        kv_cache_dtype=llama_cache_dtype,
                        batch_size=batch_size,
                    )
                elif benchmark_backend == "vllm":
                    metrics = benchmark_vllm(
                        str(artifact_path),
                        repeats=repeats,
                        prompt_tokens=prompt_tokens,
                        generation_tokens=generation_tokens,
                        kv_cache_dtype=kv_cache_dtype,
                        batch_size=batch_size,
                    )
                else:
                    raise ValueError(f"Unknown benchmark backend {benchmark_backend!r}.")
                result.update(metrics)
                result["kv_cache_dtype"] = llama_cache_dtype if benchmark_backend == "llama.cpp" else kv_cache_dtype
                result["batch_size"] = batch_size
            result["status"] = "completed"
            print(f"Completed {method}:{variant} → {artifact_path}")
        except (OSError, RuntimeError, ValueError, ImportError) as error:
            result["status"] = "failed"
            result["error"] = str(error)
            print(f"Failed {method}:{variant}: {error}")
        store.save_result(result)
    return run_id, store.results(run_id)


def _artifact_size(path: Path) -> float:
    if path.is_file():
        byte_count = path.stat().st_size
    else:
        byte_count = sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())
    return byte_count / 1024**3
