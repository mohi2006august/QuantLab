# QuantLab

QuantLab runs reproducible local LLM quantization matrices, measures quality and throughput, and builds Pareto reports with hardware-aware recommendations. The Rust CLI is the primary entry point. GPU operations use optional backends; install only the extras your hardware and workflow need.

## Rust CLI

Install the Rust toolchain, then run `cargo run --release -- estimate --model Qwen2.5-14B --format Q4_K_M --device rtx4090` or build an executable with `cargo build --release`. Rust implements `doctor`, planning, GGUF runs, SQLite reports, llama.cpp benchmarks, GGUF evaluation, Hub publishing, flat YAML/JSON configs, and weight-only safetensors sensitivity. BnB/GPTQ/AWQ/FP8 quantization, Transformers/lm-eval, activation-weighted sensitivity, and vLLM still use Python ML adapters. Both Rust and Python test workflows run in GitHub Actions.

## Quick start

```powershell
cargo run --release -- doctor
cargo run --release -- recommend --model Qwen2.5-14B --device m3pro18
cargo run --release -- estimate --parameters-b 14 --format Q4_K_M --device-memory-gb 18
cargo run --release -- run --model .\models\qwen.gguf --methods gguf:Q4_K_M --device rtx4090 --benchmark llama.cpp
```

Native Rust GGUF workflows need llama.cpp executables on `PATH`; converting a Hugging Face model folder also needs `LLAMA_CPP_DIR`. The Python compatibility package (`python -m pip install -e .`) remains available for optional GPU backends and evaluations not yet ported to Rust.

You can pass a parameter count directly when it cannot be inferred from the model name:

```powershell
cargo run --release -- recommend --parameters-b 8 --device rtx4090
```

Built-in device profiles are stored in `devices/`. Use `--device-memory-gb` to estimate for another device without editing a profile. Override the runtime reserve with `--runtime-reserve-gib`.

## Quantization matrix

The Rust `run` command accepts local Hugging Face model folders, GGUF files, or Hub model IDs (downloaded with the `hf` CLI) and runs GGUF quantization. The Python compatibility command also supports the other quantizers. GGUF conversion requires llama.cpp (`llama-quantize`, `llama-perplexity`, `llama-bench`) and `LLAMA_CPP_DIR` pointing to its source checkout for the converter. bitsandbytes can load Hugging Face IDs directly. GPTQ and AWQ runs require a calibration file containing one sample per line, or JSON/JSONL text records. The FP8 backend uses llm-compressor's dynamic FP8 recipe and does not require calibration text. Rust run configs support flat YAML mappings with scalar values and simple lists, plus JSON.

```powershell
quantlab run --model .\models\my-model --methods gguf:Q4_K_M,gguf:Q5_K_M --device rtx4090 --eval-text .\data\wiki.test.raw --benchmark llama.cpp --kv-cache-dtype q8_0 --batch-size 2048 --prompt-tokens 512 --generation-tokens 128
quantlab run --model org/model --methods bnb:nf4,bnb:int8 --eval-text .\data\heldout.txt --prompt-suite .\examples\prompts.json --tasks mmlu,gsm8k
quantlab run --model org/model --methods gptq:4,awq:4,fp8:fp8 --calibration-file .\data\calibration.jsonl --resume RUN_ID
quantlab run --config .\examples\qwen7b.yaml
quantlab run --config .\examples\qwen7b.json
```

Each matrix run records states and metrics in `quantlab-runs/results.sqlite` and writes a self-contained HTML report. Resume an interrupted run with its printed ID and the same `--output-dir`; completed artifacts are reused for later evaluation stages. Missing optional tools are reported as actionable job failures, so the rest of the matrix can continue.

## Evaluation and analysis

Rust `evaluate` supports GGUF perplexity and JSON prompt suites through llama.cpp. The Python compatibility command additionally supports Transformers sliding-window perplexity and lm-evaluation-harness tasks. Prompt suites accept objects with `prompt`, optional `expected`, and optional `id`; exact match is reported only for entries with an expected answer. Python-backed evaluation can pass `--judge-model` for local judging. Rust `benchmark` measures llama.cpp throughput; the Python command additionally supports vLLM. Use `--batch-sizes`, `--prompt-token-sizes`, `--generation-token-sizes`, and `--kv-cache-dtypes` for a Cartesian parameter sweep. Rust `sensitivity` analyzes safetensors weights without loading the full model; `--calibration-file` enables the Python activation-weighted analysis.

Rust and Python `publish` upload a local artifact folder to Hugging Face Hub using `HF_TOKEN` and require the source model's license identifier to create a model card. Check the model and dataset licenses before publishing. Publishing is an explicit separate command and is not invoked by `run`.

## Optional dependencies

- `pip install -e ".[test]"` and `python -m pytest`: run the unit test suite. GitHub Actions runs it on Python 3.10, 3.11, and 3.12 for pushes and pull requests.
- `pip install -e ".[gpu]"`: PyTorch, Transformers, and bitsandbytes.
- `pip install -e ".[gptq]"`, `.[awq]`, or `.[fp8]`: quantization backends.
- `pip install -e ".[eval]"`, `.[lm-eval]`, or `.[vllm]`: model evaluation and serving benchmarks.
- `pip install -e ".[hub]"`: Hugging Face Hub downloads and publishing.
- `pip install -e ".[config]"`: YAML run configuration support.

QuantLab's memory recommender estimates weight bytes from effective bits per parameter and adds a configurable runtime reserve. It does not predict KV cache use, CUDA/Metal allocations, fragmentation, or measured quality/speed. Use actual evaluation and benchmarks before choosing a production format.

## Current commands

- `quantlab doctor`: inspect optional runtimes and libraries.
- `quantlab run`: quantize/evaluate/benchmark a resumable method matrix.
- `quantlab evaluate` and `quantlab benchmark`: run measurement stages independently.
- `quantlab report`: rebuild a Pareto report from SQLite results.
- `quantlab sensitivity`: analyze simulated layer quantization error.
- `quantlab recommend` and `quantlab estimate`: plan formats against device memory.
- `quantlab publish`: upload a model artifact explicitly to the Hub.
