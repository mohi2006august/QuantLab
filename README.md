# QuantLab

QuantLab runs reproducible local LLM quantization matrices, measures quality and throughput, and builds Pareto reports with hardware-aware recommendations. GPU operations use optional backends; install only the extras your hardware and workflow need.

## Quick start

```powershell
python -m pip install -e .
quantlab doctor
quantlab recommend --model Qwen2.5-14B --device m3pro18
quantlab estimate --parameters-b 14 --format Q4_K_M --device-memory-gb 18
quantlab run --model Qwen/Qwen2.5-0.5B --methods gguf:Q4_K_M --device rtx4090 --benchmark llama.cpp
```

You can pass a parameter count directly when it cannot be inferred from the model name:

```powershell
quantlab recommend --parameters-b 8 --device rtx4090
```

Built-in device profiles are stored in `devices/`. Use `--device-memory-gb` to estimate for another device without editing a profile. Override the runtime reserve with `--runtime-reserve-gib`.

## Quantization matrix

`run` accepts a local Hugging Face model directory, a GGUF file, or a Hugging Face model ID. GGUF conversion requires llama.cpp (`llama-quantize`, `llama-perplexity`, `llama-bench`) and `LLAMA_CPP_DIR` pointing to its source checkout for the converter. bitsandbytes can load Hugging Face IDs directly. GPTQ and AWQ runs require a calibration file containing one sample per line, or JSON/JSONL text records. The FP8 backend uses llm-compressor's dynamic FP8 recipe and does not require calibration text. YAML/JSON configuration files let you keep a run matrix reproducible.

```powershell
quantlab run --model .\models\my-model --methods gguf:Q4_K_M,gguf:Q5_K_M --device rtx4090 --eval-text .\data\wiki.test.raw --benchmark llama.cpp --kv-cache-dtype q8_0 --batch-size 2048 --prompt-tokens 512 --generation-tokens 128
quantlab run --model org/model --methods bnb:nf4,bnb:int8 --eval-text .\data\heldout.txt --prompt-suite .\examples\prompts.json --tasks mmlu,gsm8k
quantlab run --model org/model --methods gptq:4,awq:4,fp8:fp8 --calibration-file .\data\calibration.jsonl --resume RUN_ID
quantlab run --config .\examples\qwen7b.yaml
quantlab run --config .\examples\qwen7b.json
```

Each matrix run records states and metrics in `quantlab-runs/results.sqlite` and writes a self-contained HTML report. Resume an interrupted run with its printed ID and the same `--output-dir`; completed artifacts are reused for later evaluation stages. Missing optional tools are reported as actionable job failures, so the rest of the matrix can continue.

## Evaluation and analysis

`evaluate` supports sliding-window perplexity, custom JSON prompt suites, and tasks from lm-evaluation-harness. Prompt suites accept objects with `prompt`, optional `expected`, and optional `id`; exact match is reported only for entries with an expected answer. Pass `--judge-model` to run an optional local model judge against expected answers. `benchmark` measures repeated generation throughput with llama.cpp or vLLM and can vary KV cache precision. It samples per-process NVIDIA memory when `nvidia-smi` exposes it; vLLM versions that expose request metrics report TTFT, while llama.cpp reports a prefill-time estimate separately. `sensitivity` ranks linear layers by simulated groupwise quantization error and writes mixed-precision suggestions. These simulated weight errors are not a substitute for activation-aware GPTQ/AWQ calibration.

`publish` uploads a local artifact folder to Hugging Face Hub using `HF_TOKEN` and requires the source model's license identifier to create a model card. Check the model and dataset licenses before publishing. It is an explicit separate command and is not invoked by `run`.

## Optional dependencies

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
