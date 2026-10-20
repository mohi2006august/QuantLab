"""Load validated JSON or YAML run configurations."""

import json
from pathlib import Path
from typing import Any


CONFIG_KEYS = {
    "model", "methods", "device", "output_dir", "eval_text", "prompt_suite", "judge_model",
    "calibration_file", "tasks", "benchmark", "kv_cache_dtype", "repeats", "seed",
    "batch_size", "prompt_tokens", "generation_tokens",
}


def load_run_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ValueError(f"Configuration file does not exist: {config_path}")
    source = config_path.read_text(encoding="utf-8")
    if config_path.suffix.casefold() == ".json":
        config = json.loads(source)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("YAML configuration requires PyYAML. Install QuantLab with the config extra.") from error
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Run configuration must be a YAML or JSON object.")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}.")
    if "methods" in config and isinstance(config["methods"], list):
        config["methods"] = ",".join(str(item) for item in config["methods"])
    if "tasks" in config and isinstance(config["tasks"], list):
        config["tasks"] = ",".join(str(item) for item in config["tasks"])
    for key in ("model", "methods", "device", "output_dir", "eval_text", "prompt_suite", "judge_model", "calibration_file", "benchmark", "kv_cache_dtype"):
        if key in config and config[key] is not None and not isinstance(config[key], str):
            raise ValueError(f"Configuration value {key!r} must be a string.")
    for key in ("repeats", "seed", "batch_size", "prompt_tokens", "generation_tokens"):
        if key in config and (not isinstance(config[key], int) or isinstance(config[key], bool)):
            raise ValueError(f"Configuration value {key!r} must be an integer.")
    if "repeats" in config and config["repeats"] < 1:
        raise ValueError("Configuration repeats must be at least one.")
    for key in ("batch_size", "prompt_tokens", "generation_tokens"):
        if key in config and config[key] < 1:
            raise ValueError(f"Configuration {key} must be at least one.")
    if config.get("benchmark") not in {None, "llama.cpp", "vllm"}:
        raise ValueError("Configuration benchmark must be 'llama.cpp' or 'vllm'.")
    return config
