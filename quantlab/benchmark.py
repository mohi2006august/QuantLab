"""llama.cpp benchmark integration."""

import shutil
import subprocess
import tempfile
import time
import os
from pathlib import Path
from statistics import mean, stdev


def _gpu_memory_gib(process_id: int) -> float | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    process_ids = {process_id}
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            process_ids.update(process.pid for process in psutil.Process(process_id).children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    memory_mib = 0.0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in process_ids and fields[1].isdigit():
            memory_mib += float(fields[1])
    return memory_mib / 1024 if memory_mib else None


def benchmark_gguf(
    model_path: str | Path,
    repeats: int = 3,
    prompt_tokens: int = 512,
    generation_tokens: int = 128,
    kv_cache_dtype: str = "f16",
    batch_size: int = 2048,
) -> dict[str, float]:
    executable = shutil.which("llama-bench")
    if not executable:
        raise RuntimeError("llama-bench was not found on PATH. Install llama.cpp and add its build/bin directory to PATH.")
    if repeats < 1 or batch_size < 1 or prompt_tokens < 1 or generation_tokens < 1:
        raise ValueError("Benchmark repeats, batch size, prompt tokens, and generation tokens must be at least one.")
    if kv_cache_dtype not in {"f16", "q8_0", "q5_0", "q4_0"}:
        raise ValueError("llama.cpp KV cache dtype must be f16, q8_0, q5_0, or q4_0.")
    command = [
        executable, "-m", str(model_path), "-o", "json", "-r", str(repeats),
        "-p", str(prompt_tokens), "-n", str(generation_tokens), "-b", str(batch_size),
        "-ctk", kv_cache_dtype, "-ctv", kv_cache_dtype,
    ]
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, text=True)
        peak_memory = None
        while process.poll() is None:
            measured = _gpu_memory_gib(process.pid)
            if measured is not None:
                peak_memory = max(peak_memory or 0.0, measured)
            time.sleep(0.25)
        return_code = process.wait()
        measured = _gpu_memory_gib(process.pid)
        if measured is not None:
            peak_memory = max(peak_memory or 0.0, measured)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    if return_code:
        raise RuntimeError(f"llama-bench failed ({return_code}): {stderr[-4000:]}")
    import json

    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("llama-bench did not return valid JSON output.") from error
    records = rows if isinstance(rows, list) else rows.get("results", [])
    generation = [row for row in records if int(row.get("n_gen", 0)) > 0]
    prompt_processing = [row for row in records if int(row.get("n_prompt", 0)) > 0 and int(row.get("n_gen", 0)) == 0]
    if not generation:
        raise RuntimeError("No generation throughput result was found in llama-bench output.")
    metrics = {
        "tok_s": float(generation[0].get("avg_ts", 0)),
        "tok_s_std": float(generation[0].get("stddev_ts", 0)),
        "repeats": float(repeats),
    }
    if prompt_processing:
        prefill_rate = float(prompt_processing[0].get("avg_ts", 0))
        if prefill_rate > 0:
            metrics["prefill_tok_s"] = prefill_rate
            metrics["prefill_ms_estimate"] = 1000 * prompt_tokens / prefill_rate
    if peak_memory is not None:
        metrics["peak_mem_gib"] = peak_memory
    return metrics


def benchmark_vllm(
    model_path: str,
    repeats: int = 3,
    prompt_tokens: int = 512,
    generation_tokens: int = 128,
    kv_cache_dtype: str = "auto",
    batch_size: int = 1,
) -> dict[str, float]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("vLLM benchmark requires the vllm optional dependency and a supported NVIDIA GPU.") from error
    if repeats < 1 or batch_size < 1 or prompt_tokens < 1 or generation_tokens < 1:
        raise ValueError("Benchmark repeats, batch size, prompt tokens, and generation tokens must be at least one.")
    engine = LLM(model=model_path, kv_cache_dtype=kv_cache_dtype, enable_prefix_caching=False)
    prompt = "Quantization benchmark prompt. " * max(1, prompt_tokens // 4)
    params = SamplingParams(max_tokens=generation_tokens, temperature=0)
    durations = []
    ttft_values = []
    peak_memory = None
    import time

    for _ in range(repeats + 1):
        peak_before = _gpu_memory_gib(os.getpid())
        if peak_before is not None:
            peak_memory = max(peak_memory or 0.0, peak_before)
        started = time.perf_counter()
        prompts = [f"{prompt} Benchmark item {index}." for index in range(batch_size)]
        result = engine.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        peak_after = _gpu_memory_gib(os.getpid())
        if peak_after is not None:
            peak_memory = max(peak_memory or 0.0, peak_after)
        if durations:
            generated = sum(len(item.outputs[0].token_ids) for item in result)
            durations.append(generated / elapsed)
            for item in result:
                request_metrics = getattr(item, "metrics", None)
                arrival_time = getattr(request_metrics, "arrival_time", None)
                first_token_time = getattr(request_metrics, "first_token_time", None)
                if arrival_time is not None and first_token_time is not None:
                    ttft_values.append((first_token_time - arrival_time) * 1000)
        else:
            durations.append(0.0)
    rates = durations[1:]
    metrics = {
        "tok_s": mean(rates),
        "tok_s_std": stdev(rates) if len(rates) > 1 else 0.0,
        "repeats": float(len(rates)),
    }
    measured_ttft = [value for value in ttft_values if value is not None]
    if measured_ttft:
        metrics["ttft_ms"] = mean(measured_ttft)
        metrics["ttft_ms_std"] = stdev(measured_ttft) if len(measured_ttft) > 1 else 0.0
    if peak_memory is not None:
        metrics["peak_mem_gib"] = peak_memory
    return metrics
