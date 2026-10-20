use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use candle_transformers::models::{mistral, qwen2};
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
    config.as_ref().and_then(|value| value["model_type"].as_str())
        .is_some_and(|model_type| matches!(model_type, "llama" | "mistral" | "qwen2"))
        && collect_safetensors(path).is_ok_and(|files| !files.is_empty())
}

pub fn evaluate_perplexity(path: &Path, text_path: &Path, stride: usize, max_length: Option<usize>) -> Result<Value> {
    let mut model = CandleLlama::load(path)?;
    let text = fs::read_to_string(text_path).with_context(|| format!("Could not read evaluation text {}", text_path.display()))?;
    let tokens = model.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    if tokens.len() < 2 {
        bail!("Evaluation text must encode to at least two tokens.");
    }
    let model_max_length = model.max_position_embeddings;
    let context_length = max_length.unwrap_or(model_max_length);
    if context_length < 2 || context_length > model_max_length || stride == 0 || stride > context_length {
        bail!("Stride must be positive and no greater than a context length between 2 and the model maximum ({model_max_length}).");
    }
    let mut previous_end = 0usize;
    let mut begin = 0usize;
    let mut negative_log_likelihood = 0.0f64;
    let mut target_count = 0usize;
    while begin < tokens.len() - 1 {
        let end = (begin + context_length).min(tokens.len());
        let target_start = previous_end.max(begin + 1);
        model.reset();
        let mut cache = None;
        for input_index in begin..end - 1 {
            let input = Tensor::new(&[tokens[input_index]], &model.device)?.reshape((1, 1))?;
            let logits = model.forward(&input, input_index - begin, &mut cache)?.to_vec1::<f32>()?;
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
        "architecture": model.architecture,
    }))
}

pub fn benchmark(path: &Path, repeats: usize, prompt_tokens: usize, generation_tokens: usize, batch_size: usize) -> Result<Value> {
    if repeats == 0 || prompt_tokens == 0 || generation_tokens == 0 || batch_size == 0 {
        bail!("Benchmark repeats, batch size, and token counts must be positive.");
    }
    let mut model = CandleLlama::load(path)?;
    let prompt_source = "QuantLab benchmark prompt. ".repeat(prompt_tokens);
    let encoded = model.tokenizer.encode(prompt_source, true).map_err(anyhow::Error::msg)?;
    if encoded.get_ids().len() < prompt_tokens {
        bail!("Could not construct the requested number of benchmark prompt tokens.");
    }
    let prompt_ids = &encoded.get_ids()[..prompt_tokens];
    if prompt_tokens + generation_tokens > model.max_position_embeddings {
        bail!("Benchmark prompt and generation exceed the model context window.");
    }
    for _ in 0..batch_size {
        model.generate_tokens(prompt_ids, generation_tokens)?;
    }
    let mut rates = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = std::time::Instant::now();
        let mut generated_tokens = 0usize;
        for _ in 0..batch_size {
            generated_tokens += model.generate_tokens(prompt_ids, generation_tokens)?.len();
        }
        rates.push(generated_tokens as f64 / started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE));
    }
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let variance = if rates.len() > 1 {
        rates.iter().map(|rate| (rate - mean).powi(2)).sum::<f64>() / (rates.len() - 1) as f64
    } else {
        0.0
    };
    Ok(json!({
        "tok_s": mean,
        "tok_s_std": variance.sqrt(),
        "repeats": repeats,
        "runtime": "candle",
        "architecture": model.architecture,
        "measurement": "sequential_generation",
    }))
}

pub fn evaluate_prompt_suite(path: &Path, suite: &Value, max_new_tokens: usize, judge_path: Option<&Path>) -> Result<Value> {
    let items = suite.as_array().context("Prompt suite must be a JSON array")?;
    if items.is_empty() || max_new_tokens == 0 {
        bail!("Prompt suite must not be empty and max_new_tokens must be positive.");
    }
    let mut model = CandleLlama::load(path)?;
    let mut judge = judge_path.map(CandleLlama::load).transpose()?;
    let mut outputs = Vec::with_capacity(items.len());
    let mut exact_matches = Vec::new();
    let mut judge_scores = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let prompt = item["prompt"].as_str().context("Each prompt-suite item needs a string 'prompt'")?;
        let completion = model.generate(prompt, max_new_tokens)?;
        let expected = item["expected"].as_str();
        let exact_match = expected.map(|answer| completion.trim().eq_ignore_ascii_case(answer.trim()));
        if let Some(matched) = exact_match {
            exact_matches.push(matched);
        }
        let judge_pass = if let (Some(judge_model), Some(reference)) = (judge.as_mut(), expected) {
            let judge_prompt = format!(
                "Decide whether the answer is correct and satisfies the reference. Reply with exactly PASS or FAIL.\nQuestion: {prompt}\nReference: {reference}\nAnswer: {completion}\nDecision:"
            );
            let verdict = judge_model.generate(&judge_prompt, 4)?;
            let passed = judge_verdict(&verdict);
            judge_scores.push(passed);
            Some(passed)
        } else {
            None
        };
        outputs.push(json!({
            "id": item.get("id").cloned().unwrap_or_else(|| json!(index.to_string())),
            "completion": completion,
            "exact_match": exact_match,
            "judge_pass": judge_pass,
        }));
    }
    let exact_match_score = if exact_matches.is_empty() { Value::Null } else {
        json!(exact_matches.iter().filter(|matched| **matched).count() as f64 / exact_matches.len() as f64)
    };
    let judge_score = if judge_scores.is_empty() { Value::Null } else {
        json!(judge_scores.iter().filter(|passed| **passed).count() as f64 / judge_scores.len() as f64)
    };
    Ok(json!({
        "count": outputs.len(),
        "exact_match_score": exact_match_score,
        "scored_count": exact_matches.len(),
        "judge_score": judge_score,
        "judge_scored_count": judge_scores.len(),
        "results": outputs,
        "runtime": "candle",
        "architecture": model.architecture,
    }))
}

struct CandleLlama {
    model: DecoderModel,
    tokenizer: Tokenizer,
    architecture: &'static str,
    max_position_embeddings: usize,
    eos_token_ids: Vec<u32>,
    device: Device,
    dtype: DType,
}

enum DecoderModel {
    Llama(Llama, candle_transformers::models::llama::Config),
    Mistral(mistral::Model),
    Qwen2(qwen2::ModelForCausalLM),
}

impl CandleLlama {
    fn load(path: &Path) -> Result<Self> {
        if !supports_local_model(path) {
            bail!("Candle supports local Llama, Mistral, and Qwen2 models with config.json, tokenizer.json, and safetensors weights.");
        }
        let config_bytes = fs::read(path.join("config.json"))?;
        let config_json: Value = serde_json::from_slice(&config_bytes).context("Invalid model config.json")?;
        let architecture = config_json["model_type"].as_str().context("Model config is missing model_type")?;
        let eos_token_ids = match &config_json["eos_token_id"] {
            Value::Number(value) => value.as_u64().map(|id| vec![id as u32]).unwrap_or_default(),
            Value::Array(values) => values.iter().filter_map(Value::as_u64).map(|id| id as u32).collect(),
            _ => Vec::new(),
        };
        let dtype = match std::env::var("QUANTLAB_CANDLE_DTYPE").as_deref() {
            Ok("f32") => DType::F32,
            Ok("bf16") => DType::BF16,
            Ok("f16") | Err(_) => DType::F16,
            Ok(other) => bail!("Unsupported QUANTLAB_CANDLE_DTYPE {other:?}; choose f16, bf16, or f32."),
        };
        let device = default_device()?;
        let weights = collect_safetensors(path)?;
        let var_builder = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
        let (model, max_position_embeddings, architecture) = match architecture {
            "llama" => {
                let hf_config: LlamaConfig = serde_json::from_value(config_json).context("Unsupported or invalid Llama config.json")?;
                let config = hf_config.into_config(false);
                let max_position_embeddings = config.max_position_embeddings;
                (DecoderModel::Llama(Llama::load(var_builder, &config)?, config), max_position_embeddings, "llama")
            }
            "mistral" => {
                let config: mistral::Config = serde_json::from_value(config_json).context("Unsupported or invalid Mistral config.json")?;
                let max_position_embeddings = config.max_position_embeddings;
                (DecoderModel::Mistral(mistral::Model::new(&config, var_builder)?), max_position_embeddings, "mistral")
            }
            "qwen2" => {
                let config: qwen2::Config = serde_json::from_value(config_json).context("Unsupported or invalid Qwen2 config.json")?;
                let max_position_embeddings = config.max_position_embeddings;
                (DecoderModel::Qwen2(qwen2::ModelForCausalLM::new(&config, var_builder)?), max_position_embeddings, "qwen2")
            }
            _ => bail!("Unsupported Candle architecture {architecture:?}"),
        };
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        Ok(Self { model, tokenizer, architecture, max_position_embeddings, eos_token_ids, device, dtype })
    }

    fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let generated = self.generate_tokens(&input_ids, max_new_tokens)?;
        self.tokenizer.decode(&generated, true).map_err(anyhow::Error::msg)
    }

    fn generate_tokens(&mut self, input_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        if input_ids.is_empty() { bail!("Prompt must encode to at least one token."); }
        if input_ids.len() >= self.max_position_embeddings {
            bail!("Prompt exceeds the model context window.");
        }
        self.reset();
        let mut cache = None;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut previous = Vec::new();
        for index in 0..max_new_tokens {
            let (token_slice, offset) = if index == 0 {
                (input_ids, 0)
            } else {
                previous.clear();
                previous.push(*generated.last().unwrap());
                (previous.as_slice(), input_ids.len() + index - 1)
            };
            let input = Tensor::new(token_slice, &self.device)?.unsqueeze(0)?;
            let logits = self.forward(&input, offset, &mut cache)?;
            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
            if self.eos_token_ids.contains(&next_token) { break; }
            generated.push(next_token);
        }
        Ok(generated)
    }

    fn reset(&mut self) {
        match &mut self.model {
            DecoderModel::Llama(_, _) => {}
            DecoderModel::Mistral(model) => model.clear_kv_cache(),
            DecoderModel::Qwen2(model) => model.clear_kv_cache(),
        }
    }

    fn forward(&mut self, input: &Tensor, offset: usize, cache: &mut Option<Cache>) -> Result<Tensor> {
        let logits = match &mut self.model {
            DecoderModel::Llama(model, config) => {
                if cache.is_none() {
                    *cache = Some(Cache::new(true, self.dtype, config, &self.device)?);
                }
                model.forward(input, offset, cache.as_mut().unwrap())?
            }
            DecoderModel::Mistral(model) => model.forward(input, offset)?,
            DecoderModel::Qwen2(model) => model.forward(input, offset)?,
        };
        match logits.dims().len() {
            3 => Ok(logits.squeeze(0)?.squeeze(0)?),
            2 => Ok(logits.squeeze(0)?),
            _ => bail!("Candle model returned unexpected logits shape {:?}", logits.dims()),
        }
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

fn judge_verdict(verdict: &str) -> bool {
    verdict.trim().to_ascii_lowercase().starts_with("pass")
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
    fn only_claims_supported_local_transformer_directories() {
        let directory = tempfile::tempdir().unwrap();
        assert!(!supports_local_model(directory.path()));
        fs::write(directory.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(directory.path().join("model.safetensors"), []).unwrap();
        for model_type in ["llama", "mistral", "qwen2"] {
            fs::write(directory.path().join("config.json"), format!(r#"{{"model_type":"{model_type}"}}"#)).unwrap();
            assert!(supports_local_model(directory.path()));
        }
        fs::write(directory.path().join("config.json"), r#"{"model_type":"other"}"#).unwrap();
        assert!(!supports_local_model(directory.path()));
    }

    #[test]
    fn parses_native_judge_verdicts() {
        assert!(judge_verdict("PASS"));
        assert!(judge_verdict(" pass: correct"));
        assert!(!judge_verdict("FAIL"));
        assert!(!judge_verdict("uncertain"));
    }
}
