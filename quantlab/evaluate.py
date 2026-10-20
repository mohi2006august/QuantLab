"""Perplexity and prompt-suite evaluation using Transformers."""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Evaluation requires torch and transformers. Install QuantLab with the eval extra.") from error
    return torch, AutoModelForCausalLM, AutoTokenizer


def evaluate_perplexity(model_path: str, text_path: str, stride: int = 512, max_length: int | None = None) -> dict[str, float]:
    if Path(model_path).suffix.casefold() == ".gguf":
        executable = shutil.which("llama-perplexity")
        if not executable:
            raise RuntimeError("GGUF perplexity evaluation requires llama-perplexity on PATH (from llama.cpp).")
        completed = subprocess.run(
            [executable, "-m", model_path, "-f", text_path, "-c", str(max_length or 2048)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"llama-perplexity failed ({completed.returncode}): {completed.stderr[-4000:]}")
        output = completed.stdout + "\n" + completed.stderr
        matches = re.findall(r"(?:perplexity|ppl)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", output, re.IGNORECASE)
        if not matches:
            raise RuntimeError("Could not find a perplexity value in llama-perplexity output.")
        return {"perplexity": float(matches[-1]), "tokens": 0.0}
    torch, AutoModelForCausalLM, AutoTokenizer = _transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.eval()
    text = Path(text_path).read_text(encoding="utf-8")
    token_ids = tokenizer(text, return_tensors="pt").input_ids
    sequence_length = token_ids.size(1)
    context_length = max_length or getattr(model.config, "max_position_embeddings", 2048)
    if stride < 1 or context_length < 2 or stride > context_length:
        raise ValueError("Stride must be between one and the model context length; context length must be at least two.")
    if sequence_length < 2:
        raise ValueError("The evaluation text must contain at least two tokens.")

    device = next(model.parameters()).device
    negative_log_likelihood = 0.0
    target_count = 0
    previous_end = 0
    with torch.inference_mode():
        for begin in range(0, sequence_length, stride):
            end = min(begin + context_length, sequence_length)
            target_length = end - previous_end
            inputs = token_ids[:, begin:end].to(device)
            labels = inputs.clone()
            labels[:, :-target_length] = -100
            loss = model(inputs, labels=labels).loss
            scored_tokens = max(1, target_length - (1 if previous_end == 0 else 0))
            negative_log_likelihood += float(loss) * scored_tokens
            target_count += scored_tokens
            previous_end = end
            if end >= sequence_length:
                break
    mean_loss = negative_log_likelihood / target_count
    return {"perplexity": math.exp(mean_loss), "tokens": float(target_count), "mean_nll": mean_loss}


def evaluate_prompt_suite(
    model_path: str,
    suite_path: str,
    max_new_tokens: int = 128,
    judge_model_path: str | None = None,
) -> dict[str, Any]:
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    if not isinstance(suite, list) or not suite:
        raise ValueError("Prompt suite must be a non-empty JSON array of prompt objects.")
    if Path(model_path).suffix.casefold() == ".gguf":
        return _evaluate_gguf_prompt_suite(model_path, suite, max_new_tokens, judge_model_path)
    torch, AutoModelForCausalLM, AutoTokenizer = _transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.eval()
    judge_tokenizer = None
    judge_model = None
    if judge_model_path:
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_path, use_fast=True)
        judge_model = AutoModelForCausalLM.from_pretrained(judge_model_path, device_map="auto")
        judge_model.eval()
        judge_model.eval()
    results = []
    for index, item in enumerate(suite):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"Prompt item {index} must be an object with a string 'prompt'.")
        encoded = tokenizer(item["prompt"], return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
        completion = tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = item.get("expected")
        exact_match = None
        if isinstance(expected, str):
            exact_match = completion.casefold() == expected.strip().casefold()
        judge_pass = None
        if judge_model is not None and isinstance(expected, str):
            judge_prompt = (
                "Decide whether the answer is correct and satisfies the reference. "
                "Reply with exactly PASS or FAIL.\n"
                f"Question: {item['prompt']}\nReference: {expected}\nAnswer: {completion}\nDecision:"
            )
            judge_device = next(judge_model.parameters()).device
            judge_inputs = judge_tokenizer(judge_prompt, return_tensors="pt").to(judge_device)
            with torch.inference_mode():
                judged = judge_model.generate(**judge_inputs, max_new_tokens=4, do_sample=False)
            verdict = judge_tokenizer.decode(judged[0, judge_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().casefold()
            judge_pass = verdict.startswith("pass")
        results.append({"id": item.get("id", str(index)), "completion": completion, "exact_match": exact_match, "judge_pass": judge_pass})
    scored = [row["exact_match"] for row in results if row["exact_match"] is not None]
    judged = [row["judge_pass"] for row in results if row["judge_pass"] is not None]
    return {
        "count": len(results),
        "exact_match_score": sum(scored) / len(scored) if scored else None,
        "scored_count": len(scored),
        "judge_score": sum(judged) / len(judged) if judged else None,
        "judge_scored_count": len(judged),
        "results": results,
    }


def _evaluate_gguf_prompt_suite(model_path: str, suite: list[Any], max_new_tokens: int, judge_model_path: str | None) -> dict[str, Any]:
    executable = shutil.which("llama-cli")
    if not executable:
        raise RuntimeError("GGUF prompt evaluation requires llama-cli on PATH (from llama.cpp).")
    results = []
    judge_tokenizer = None
    judge_model = None
    if judge_model_path:
        torch, AutoModelForCausalLM, AutoTokenizer = _transformers()
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_path, use_fast=True)
        judge_model = AutoModelForCausalLM.from_pretrained(judge_model_path, device_map="auto")
        judge_model.eval()
    for index, item in enumerate(suite):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"Prompt item {index} must be an object with a string 'prompt'.")
        completed = subprocess.run(
            [executable, "-m", model_path, "-p", item["prompt"], "-n", str(max_new_tokens), "--simple-io", "--no-display-prompt"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"llama-cli prompt evaluation failed ({completed.returncode}): {completed.stderr[-2000:]}")
        completion = completed.stdout.strip()
        expected = item.get("expected")
        exact_match = completion.casefold() == expected.strip().casefold() if isinstance(expected, str) else None
        judge_pass = None
        if judge_model is not None and isinstance(expected, str):
            judge_prompt = f"Decide whether the answer is correct and satisfies the reference. Reply exactly PASS or FAIL.\nQuestion: {item['prompt']}\nReference: {expected}\nAnswer: {completion}\nDecision:"
            judge_device = next(judge_model.parameters()).device
            judge_inputs = judge_tokenizer(judge_prompt, return_tensors="pt").to(judge_device)
            with torch.inference_mode():
                judged = judge_model.generate(**judge_inputs, max_new_tokens=4, do_sample=False)
            verdict = judge_tokenizer.decode(judged[0, judge_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().casefold()
            judge_pass = verdict.startswith("pass")
        results.append({"id": item.get("id", str(index)), "completion": completion, "exact_match": exact_match, "judge_pass": judge_pass})
    scored = [row["exact_match"] for row in results if row["exact_match"] is not None]
    judged = [row["judge_pass"] for row in results if row["judge_pass"] is not None]
    return {
        "count": len(results),
        "exact_match_score": sum(scored) / len(scored) if scored else None,
        "scored_count": len(scored),
        "judge_score": sum(judged) / len(judged) if judged else None,
        "judge_scored_count": len(judged),
        "results": results,
    }


def evaluate_tasks(model_path: str, tasks: list[str], limit: int | None = None, seed: int = 1234) -> dict[str, Any]:
    try:
        from lm_eval import simple_evaluate
    except ImportError as error:
        raise RuntimeError("Task benchmarks require lm-evaluation-harness. Install QuantLab with the lm-eval extra.") from error
    if not tasks or any(not task.strip() for task in tasks):
        raise ValueError("Specify at least one non-empty lm-evaluation-harness task.")
    raw = simple_evaluate(
        model="hf",
        model_args={"pretrained": model_path},
        tasks=[task.strip() for task in tasks],
        num_fewshot=0,
        batch_size="auto",
        limit=limit,
        random_seed=seed,
    )
    return {"results": raw.get("results", {}), "versions": raw.get("versions", {}), "seed": seed}
