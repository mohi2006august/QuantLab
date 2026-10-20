"""Explicit opt-in Hugging Face Hub publishing."""

import os
import re
from io import BytesIO
from pathlib import Path


def publish_model(model_path: str | Path, repo_id: str, license_id: str, private: bool = False) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in the environment before publishing.")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("Hub publishing requires huggingface_hub. Install QuantLab with the hub extra.") from error
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists() or not (model_path.is_dir() or model_path.is_file()):
        raise ValueError("Model path must be an existing local file or directory.")
    if not re.fullmatch(r"[A-Za-z0-9.-]+", license_id.strip()):
        raise ValueError("Provide a simple SPDX license identifier for the model card.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True)
    model_card = model_path / "README.md" if model_path.is_dir() else None
    has_model_card = bool(model_card and model_card.exists())
    if not has_model_card:
        card = (
            "---\n"
            f"license: {license_id.strip()}\n"
            "library_name: quantlab\n"
            "tags:\n  - quantization\n  - quantlab\n"
            "---\n\n"
            f"# {repo_id.rsplit('/', 1)[-1]}\n\n"
            "Quantized model artifact published with QuantLab.\n\n"
            "Confirm the upstream model and calibration data licenses permit this redistribution. "
            "Add the source model, quantization method, settings, evaluation results, and hardware measurements before sharing.\n"
        )
        api.upload_file(
            path_or_fileobj=BytesIO(card.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add QuantLab model card",
        )
        if model_path.is_dir():
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload quantized model files",
                ignore_patterns=["README.md"],
            )
        else:
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload quantized model file",
            )
    else:
        api.upload_folder(folder_path=str(model_path), repo_id=repo_id, repo_type="model", commit_message="Publish quantized model with QuantLab")
    return api.model_info(repo_id, repo_type="model").url
