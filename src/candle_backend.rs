use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub fn supports_local_model(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let config_path = path.join("config.json");
    let tokenizer_path = path.join("tokenizer.json");
    if !config_path.is_file() || !tokenizer_path.is_file() {
        return false;
    }
    let config = fs::read(config_path).ok().and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    config.as_ref().and_then(|value| value["model_type"].as_str()) == Some("llama")
        && collect_safetensors(path).is_ok_and(|files| !files.is_empty())
}

pub fn evaluate_perplexity(path: &Path, text_path: &Path, stride: usize, max_length: Option<usize>) -> Result<Value> {
    let model = CandleLlama::load(path)?;
    let text = fs::read_to_string(text_path).with_context(|| format!("Could not read evaluation text {}", text_path.display()))?;
    let tokens = model.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    if tokens.len() < 2 {
        bail!("Evaluation text must encode to at least two tokens.");
    }
    let context_length = max_length.unwrap_or(model.config.max_position_embeddings);
    if context_length < 2 || context_length > model.config.max_position_embeddings || stride == 0 || stride > context_length {
        bail!("Stride must be positive and no greater than a context length between 2 and the model maximum ({}).", model.config.max_position_embeddings);
    }
    let mut previous_end = 0usize;
    let mut begin = 0usize;
    let mut negative_log_likelihood = 0.0f64;
    let mut target_count = 0usize;
    while begin < tokens.len() - 1 {
        let end = (begin + context_length).min(tokens.len());
        let target_start = previous_end.max(begin + 1);
        let mut cache = Cache::new(true, model.dtype, &model.config, &model.device)?;
        for input_index in begin..end - 1 {
            let input = Tensor::new(&[tokens[input_index]], &model.device)?.reshape((1, 1))?;
            let logits = model.model.forward(&input, input_index - begin, &mut cache)?.squeeze(0)?.to_vec1::<f32>()?;
            let target_index = input_index + 1;
            if target_index >= target_start {
                negative_log_likelihood += negative_log_probability(&logits, tokens[target_index] as usize)?;
                target_count += 1;
            }
        }
        previous_end = end;
        if end == tokens.len() { break; }
        begin += stride;
    }
    if target_count == 0 {
        bail!("Evaluation text did not produce any scored tokens.");
    }
    let mean_nll = negative_log_likelihood / target_count as f64;
    Ok(json!({
        "perplexity": mean_nll.exp(),
        "tokens": target_count,
        "mean_nll": mean_nll,
        "runtime": "candle",
        "architecture": "llama",
    }))
}

pub fn evaluate_prompt_suite(path: &Path, suite: &Value, max_new_tokens: usize) -> Result<Value> {
    let items = suite.as_array().context("Prompt suite must be a JSON array")?;
    if items.is_empty() || max_new_tokens == 0 {
        bail!("Prompt suite must not be empty and max_new_tokens must be positive.");
    }
    let mut model = CandleLlama::load(path)?;
    let mut outputs = Vec::with_capacity(items.len());
    let mut exact_matches = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let prompt = item["prompt"].as_str().context("Each prompt-suite item needs a string 'prompt'")?;
        let completion = model.generate(prompt, max_new_tokens)?;
        let expected = item["expected"].as_str();
        let exact_match = expected.map(|answer| completion.trim().eq_ignore_ascii_case(answer.trim()));
        if let Some(matched) = exact_match {
            exact_matches.push(matched);
        }
        outputs.push(json!({
            "id": item.get("id").cloned().unwrap_or_else(|| json!(index.to_string())),
            "completion": completion,
            "exact_match": exact_match,
            "judge_pass": null,
        }));
    }
    let exact_match_score = if exact_matches.is_empty() { Value::Null } else {
        json!(exact_matches.iter().filter(|matched| **matched).count() as f64 / exact_matches.len() as f64)
    };
    Ok(json!({
        "count": outputs.len(),
        "exact_match_score": exact_match_score,
        "scored_count": exact_matches.len(),
        "judge_score": null,
        "judge_scored_count": 0,
        "results": outputs,
        "runtime": "candle",
        "architecture": "llama",
    }))
}

struct CandleLlama {
    model: Llama,
    tokenizer: Tokenizer,
    config: candle_transformers::models::llama::Config,
    device: Device,
    dtype: DType,
}

impl CandleLlama {
    fn load(path: &Path) -> Result<Self> {
        if !supports_local_model(path) {
            bail!("Candle currently supports local Llama-family models with config.json, tokenizer.json, and safetensors weights.");
        }
        let config_bytes = fs::read(path.join("config.json"))?;
        let hf_config: LlamaConfig = serde_json::from_slice(&config_bytes).context("Unsupported or invalid Llama config.json")?;
        let config = hf_config.into_config(false);
        let dtype = match std::env::var("QUANTLAB_CANDLE_DTYPE").as_deref() {
            Ok("f32") => DType::F32,
            Ok("bf16") => DType::BF16,
            Ok("f16") | Err(_) => DType::F16,
            Ok(other) => bail!("Unsupported QUANTLAB_CANDLE_DTYPE {other:?}; choose f16, bf16, or f32."),
        };
        let device = default_device()?;
        let weights = collect_safetensors(path)?;
        let var_builder = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
        let model = Llama::load(var_builder, &config)?;
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        Ok(Self { model, tokenizer, config, device, dtype })
    }

    fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        if input_ids.is_empty() { bail!("Prompt must encode to at least one token."); }
        if input_ids.len() >= self.config.max_position_embeddings {
            bail!("Prompt exceeds the model context window.");
        }
        let mut cache = Cache::new(true, self.dtype, &self.config, &self.device)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut previous = Vec::new();
        for index in 0..max_new_tokens {
            let (token_slice, offset) = if index == 0 {
                (input_ids.as_slice(), 0)
            } else {
                previous.clear();
                previous.push(*generated.last().unwrap());
                (previous.as_slice(), input_ids.len() + index - 1)
            };
            let input = Tensor::new(token_slice, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, offset, &mut cache)?.squeeze(0)?;
            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
            let is_eos = match &self.config.eos_token_id {
                Some(candle_transformers::models::llama::LlamaEosToks::Single(eos)) => next_token == *eos,
                Some(candle_transformers::models::llama::LlamaEosToks::Multiple(eos)) => eos.contains(&next_token),
                None => false,
            };
            if is_eos { break; }
            generated.push(next_token);
        }
        self.tokenizer.decode(&generated, true).map_err(anyhow::Error::msg)
    }
}

fn default_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return Ok(device);
    }
    Ok(Device::Cpu)
}

fn collect_safetensors(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(directory)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn negative_log_probability(logits: &[f32], target: usize) -> Result<f64> {
    let value = *logits.get(target).context("Token id is outside the model vocabulary")? as f64;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let normalizer = logits.iter().map(|logit| (*logit as f64 - maximum).exp()).sum::<f64>();
    Ok(maximum + normalizer.ln() - value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_stable_token_negative_log_probability() {
        let loss = negative_log_probability(&[0.0, 0.0], 0).unwrap();
        assert!((loss - std::f64::consts::LN_2).abs() < 1e-12);
        let peaked = negative_log_probability(&[1000.0, 0.0], 0).unwrap();
        assert!(peaked.abs() < 1e-12);
        assert!(negative_log_probability(&[0.0], 1).is_err());
    }

    #[test]
    fn only_claims_supported_local_llama_directories() {
        let directory = tempfile::tempdir().unwrap();
        assert!(!supports_local_model(directory.path()));
        fs::write(directory.path().join("config.json"), r#"{"model_type":"other"}"#).unwrap();
        fs::write(directory.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(directory.path().join("model.safetensors"), []).unwrap();
        assert!(!supports_local_model(directory.path()));
    }
}
