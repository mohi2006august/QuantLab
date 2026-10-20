# QuantLab

QuantLab runs reproducible local LLM quantization matrices, measures quality and throughput, and builds Pareto reports with hardware-aware recommendations. The Rust CLI is the primary entry point. GPU operations use optional backends; install only the extras your hardware and workflow need.

## Rust CLI

Install the Rust toolchain, then run `cargo run --release -- estimate --model Qwen2.5-14B --format Q4_K_M --device rtx4090` or build an executable with `cargo build --release`. Rust owns the CLI, orchestration, persistence, and reporting. Local Llama safetensors evaluation uses Candle in Rust. BnB/GPTQ/AWQ/FP8 quantization, lm-eval, vLLM, judging, and activation-captured sensitivity retain narrow Python adapters for their upstream ML implementations; they no longer re-enter the Python CLI. Both Rust and Python test workflows run in GitHub Actions.

## Quick start

```powershell
cargo run --release -- doctor
cargo run --release -- recommend --model Qwen2.5-14B --device m3pro18
cargo run --release -- estimate --parameters-b 14 --format Q4_K_M --device-memory-gb 18
cargo run --release -- run --model .\models\qwen.gguf --methods gguf:Q4_K_M --device rtx4090 --benchmark llama.cpp
```

GGUF workflows need llama.cpp executables on `PATH`; converting a Hugging Face model folder also needs `LLAMA_CPP_DIR`. To enable Candle CUDA, build with `--features cuda` and a compatible CUDA toolchain; otherwise Candle runs on CPU. `QUANTLAB_CANDLE_DTYPE` selects `f16` (default), `bf16`, or `f32`. Python plus the installed QuantLab package and matching optional dependencies are required only when a workflow invokes one of the remaining ML adapters. From this checkout, install an adapter extra with `python -m pip install -e ".[gpu]"` (choose the relevant extra below).

You can pass a parameter count directly when it cannot be inferred from the model name:

```powershell
cargo run --release -- recommend --parameters-b 8 --device rtx4090
```

Built-in device profiles are stored in `devices/`. Use `--device-memory-gb` to estimate for another device without editing a profile. Override the runtime reserve with `--runtime-reserve-gib`.

## Quantization matrix

The Rust `run` command accepts local Hugging Face model folders, GGUF files, or Hub model IDs (downloaded with the `hf` CLI), and owns the full method matrix and evaluation/benchmark stages. GGUF quantization uses llama.cpp; BnB, GPTQ, AWQ, and FP8 quantization invoke only their corresponding Python library operation. GPTQ and AWQ require a calibration file containing representative text. Rust run configs support flat YAML mappings with scalar values and simple lists, plus JSON.

```powershell
quantlab run --model .\models\my-model --methods gguf:Q4_K_M,gguf:Q5_K_M --device rtx4090 --eval-text .\data\wiki.test.raw --benchmark llama.cpp --kv-cache-dtype q8_0 --batch-size 2048 --prompt-tokens 512 --generation-tokens 128
quantlab run --model org/model --methods bnb:nf4,bnb:int8 --eval-text .\data\heldout.txt --prompt-suite .\examples\prompts.json --tasks mmlu,gsm8k
quantlab run --model org/model --methods gptq:4,awq:4,fp8:fp8 --calibration-file .\data\calibration.jsonl --resume RUN_ID
quantlab run --config .\examples\qwen7b.yaml
quantlab run --config .\examples\qwen7b.json
```

Each matrix run records states and metrics in `quantlab-runs/results.sqlite` and writes a self-contained HTML report. Resume an interrupted run with its printed ID and the same `--output-dir`; completed artifacts are reused for later evaluation stages. Missing optional tools are reported as actionable job failures, so the rest of the matrix can continue.

## Evaluation and analysis

Rust `evaluate` uses llama.cpp for GGUF and Candle for local Llama-family safetensors models; unsupported architectures/formats use the Transformers adapter. Prompt suites accept objects with `prompt`, optional `expected`, and optional `id`; exact match is reported only for entries with an expected answer. `--judge-model` uses Candle natively when both the evaluated and judge models are supported local Llama safetensors models, otherwise it uses the optional Python adapter. Rust `benchmark` supports llama.cpp, vLLM, and Candle. Candle benchmarks use sequential generation on local Llama safetensors models; they report throughput but do not provide vLLM's continuous batching, KV-cache dtype controls, or TTFT metrics. All backends support Cartesian sweeps through `--batch-sizes`, `--prompt-token-sizes`, and `--generation-token-sizes`; llama.cpp and vLLM also support `--kv-cache-dtypes`. Rust `sensitivity` analyzes safetensors weights without loading the full model; `--calibration-file` still uses Python to capture model activations.

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
