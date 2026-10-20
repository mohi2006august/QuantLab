"""Quantization backend orchestration."""

import os
import shutil
import subprocess
import json
import sys
from pathlib import Path


GGUF_VARIANTS = {"Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}


def _run(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{tail}")


def _find_converter() -> Path | None:
    configured = os.environ.get("LLAMA_CPP_DIR")
    candidates = []
    if configured:
        candidates.extend((Path(configured) / "convert_hf_to_gguf.py", Path(configured) / "convert" / "convert_hf_to_gguf.py"))
    executable = shutil.which("convert_hf_to_gguf.py")
    if executable:
        candidates.append(Path(executable))
    return next((path for path in candidates if path.is_file()), None)


def quantize_gguf(model: str, variant: str, output_dir: str | Path) -> Path:
    normalized = variant.upper()
    if normalized not in GGUF_VARIANTS:
        raise ValueError(f"Unsupported GGUF quantization {variant!r}; choose from {', '.join(sorted(GGUF_VARIANTS))}.")
    quantize_binary = shutil.which("llama-quantize")
    if not quantize_binary:
        raise RuntimeError("llama-quantize was not found on PATH. Install llama.cpp and add its build/bin directory to PATH.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(model).expanduser()
    if source.is_file() and source.suffix.casefold() == ".gguf":
        fp16_path = source
    elif source.is_dir():
        fp16_path = output_dir / "model-f16.gguf"
        converter = _find_converter()
        if not converter:
            raise RuntimeError("Could not find convert_hf_to_gguf.py. Set LLAMA_CPP_DIR to a llama.cpp checkout.")
        _run([sys.executable, str(converter), str(source), "--outfile", str(fp16_path), "--outtype", "f16"], output_dir / "convert.log")
    elif "/" in model:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("A Hugging Face model ID requires huggingface_hub. Install QuantLab with the hub extra.") from error
        downloaded_model = Path(snapshot_download(repo_id=model))
        fp16_path = output_dir / "model-f16.gguf"
        converter = _find_converter()
        if not converter:
            raise RuntimeError("Could not find convert_hf_to_gguf.py. Set LLAMA_CPP_DIR to a llama.cpp checkout.")
        _run([sys.executable, str(converter), str(downloaded_model), "--outfile", str(fp16_path), "--outtype", "f16"], output_dir / "convert.log")
    else:
        raise ValueError("GGUF input must be a local .gguf file or a local Hugging Face model directory.")

    quantized_path = output_dir / f"model-{normalized}.gguf"
    _run([quantize_binary, str(fp16_path), str(quantized_path), normalized], output_dir / "quantize.log")
    if not quantized_path.is_file():
        raise RuntimeError("llama-quantize completed without creating the expected output file.")
    return quantized_path


def quantize_model(model: str, method: str, variant: str, output_dir: str | Path, calibration_file: str | None = None) -> Path:
    method = method.casefold()
    if method == "gguf":
        return quantize_gguf(model, variant, output_dir)
    if method == "bnb":
        return quantize_bnb(model, variant, output_dir)
    if method in {"gptq", "awq"}:
        if not calibration_file:
            raise ValueError(f"{method} requires --calibration-file containing representative calibration text.")
        return quantize_transformers_backend(model, method, variant, output_dir, calibration_file)
    if method == "fp8":
        return quantize_fp8(model, output_dir)
    raise ValueError(f"Unknown quantization method {method!r}.")


def quantize_bnb(model: str, variant: str, output_dir: str | Path) -> Path:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("bitsandbytes quantization requires torch, transformers, and bitsandbytes. Install QuantLab with the gpu extra.") from error

    normalized_variant = variant.casefold()
    bits = 4 if normalized_variant == "nf4" else 8 if normalized_variant == "int8" else int(normalized_variant.removesuffix("bit").removesuffix("bits"))
    if bits not in (4, 8):
        raise ValueError("bitsandbytes supports the configured 4-bit and 8-bit presets.")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    quantized = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant_config, device_map="auto")
    quantized.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def _read_calibration_data(path: str) -> list[str]:
    calibration_path = Path(path).expanduser()
    if not calibration_path.is_file():
        raise ValueError(f"Calibration file does not exist: {calibration_path}")
    if calibration_path.suffix.casefold() == ".json":
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("texts", data.get("data"))
        if not isinstance(data, list) or any(not isinstance(item, str) for item in data):
            raise ValueError("Calibration JSON must be an array of strings or an object with a 'texts' array.")
        return data
    if calibration_path.suffix.casefold() == ".jsonl":
        records = [json.loads(line) for line in calibration_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = [record if isinstance(record, str) else record.get("text") if isinstance(record, dict) else None for record in records]
        if any(not isinstance(item, str) for item in texts):
            raise ValueError("Each calibration JSONL row must be text or an object with a string 'text' field.")
        return texts
    return [line.strip() for line in calibration_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantize_transformers_backend(model: str, method: str, variant: str, output_dir: str | Path, calibration_file: str) -> Path:
    calibration_text = _read_calibration_data(calibration_file)[:512]
    if not calibration_text:
        raise ValueError("Calibration file contains no text samples.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if method == "gptq":
        try:
            from gptqmodel import GPTQModel, QuantizeConfig
        except ImportError as error:
            raise RuntimeError("GPTQ requires GPTQModel. Install the GPTQ extra for a compatible GPU.") from error
        bits = int(variant.removesuffix("bit").removesuffix("bits"))
        quantized = GPTQModel.load(model, QuantizeConfig(bits=bits, group_size=128))
        quantized.quantize(calibration_text)
        quantized.save(str(output_dir))
        return output_dir
    if method == "awq":
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("AWQ requires AutoAWQ. Install the AWQ extra for a compatible GPU.") from error
        bits = int(variant.removesuffix("bit").removesuffix("bits"))
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        quantized = AutoAWQForCausalLM.from_pretrained(model, low_cpu_mem_usage=True, use_cache=False)
        quantized.quantize(
            tokenizer,
            quant_config={"zero_point": True, "q_group_size": 128, "w_bit": bits, "version": "GEMM"},
            calib_data=calibration_text,
        )
        quantized.save_quantized(str(output_dir))
        tokenizer.save_pretrained(output_dir)
        return output_dir
    raise ValueError(f"Unknown quantization method {method!r}.")


def quantize_fp8(model: str, output_dir: str | Path) -> Path:
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("FP8 quantization requires llm-compressor and transformers on a supported GPU. Install the fp8 extra.") from error
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    quantized = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=quantized, recipe=recipe)
    quantized.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
