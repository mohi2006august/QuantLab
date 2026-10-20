# QuantLab

QuantLab helps you compare local LLM quantization options against a device's memory budget. The first release provides a transparent memory estimator and a hardware-aware format recommender. Estimates are planning aids; they are not measured quality or speed results.

## Quick start

```powershell
python -m pip install -e .
quantlab doctor
quantlab recommend --model Qwen2.5-14B --device m3pro18
quantlab estimate --parameters-b 14 --format Q4_K_M --device-memory-gb 18
```

You can pass a parameter count directly when it cannot be inferred from the model name:

```powershell
quantlab recommend --parameters-b 8 --device rtx4090
```

Built-in device profiles are stored in `devices/`. Use `--device-memory-gb` to estimate for another device without editing a profile. Override the runtime reserve with `--runtime-reserve-gib`.

## What the estimate means

QuantLab estimates weight storage from the format's effective bits per parameter, then adds a configurable runtime reserve. The result excludes KV cache, CUDA/Metal allocations, fragmentation, and runtime-specific buffers. Actual artifacts and peak memory vary by model architecture, quantizer, and inference engine. No conversion, evaluation, or speed benchmark is run by this estimator.

## Current commands

- `quantlab doctor`: report Python and optional inference tools available on `PATH`.
- `quantlab estimate`: estimate weight and runtime memory for one format.
- `quantlab recommend`: list formats that fit a device budget, preferring higher precision.

The project roadmap is described in the QuantLab product documents alongside this repository.
