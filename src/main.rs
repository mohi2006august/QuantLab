use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rusqlite::{Connection, OptionalExtension};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};

mod candle_backend;

#[derive(Parser)]
#[command(name = "quantlab", version, about = "Plan and compare local LLM quantization")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Doctor,
    Run {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        methods: Option<String>,
        #[arg(long)]
        device: Option<String>,
        #[arg(long)]
        output_dir: Option<PathBuf>,
        #[arg(long)]
        eval_text: Option<PathBuf>,
        #[arg(long)]
        prompt_suite: Option<PathBuf>,
        #[arg(long)]
        judge_model: Option<String>,
        #[arg(long)]
        tasks: Option<String>,
        #[arg(long)]
        calibration_file: Option<PathBuf>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, value_parser = ["llama.cpp", "vllm"])]
        benchmark: Option<String>,
        #[arg(long, default_value = "auto")]
        kv_cache_dtype: String,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long, default_value_t = 512)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 128)]
        generation_tokens: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long)]
        resume: Option<String>,
    },
    Estimate {
        #[arg(long, conflicts_with = "parameters_b")]
        model: Option<String>,
        #[arg(long, conflicts_with = "model")]
        parameters_b: Option<f64>,
        #[arg(long)]
        format: String,
        #[arg(long, conflicts_with = "device_memory_gb")]
        device: Option<String>,
        #[arg(long, conflicts_with = "device")]
        device_memory_gb: Option<f64>,
        #[arg(long, default_value_t = 1.5)]
        runtime_reserve_gib: f64,
    },
    Recommend {
        #[arg(long, conflicts_with = "parameters_b")]
        model: Option<String>,
        #[arg(long, conflicts_with = "model")]
        parameters_b: Option<f64>,
        #[arg(long, conflicts_with = "device_memory_gb")]
        device: Option<String>,
        #[arg(long, conflicts_with = "device")]
        device_memory_gb: Option<f64>,
        #[arg(long, default_value_t = 1.5)]
        runtime_reserve_gib: f64,
    },
    Benchmark {
        #[arg(long)]
        model_path: String,
        #[arg(long, default_value = "llama.cpp", value_parser = ["llama.cpp", "vllm"])]
        backend: String,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long, default_value_t = 512)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 128)]
        generation_tokens: usize,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long, default_value = "auto")]
        kv_cache_dtype: String,
        #[arg(long)]
        batch_sizes: Option<String>,
        #[arg(long)]
        prompt_token_sizes: Option<String>,
        #[arg(long)]
        generation_token_sizes: Option<String>,
        #[arg(long)]
        kv_cache_dtypes: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Report {
        run_id: String,
        #[arg(long, default_value = "quantlab-runs")]
        output_dir: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Evaluate {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        text_file: Option<PathBuf>,
        #[arg(long)]
        prompt_suite: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 512)]
        stride: usize,
        #[arg(long)]
        max_length: Option<usize>,
        #[arg(long)]
        tasks: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 1234)]
        seed: u64,
        #[arg(long)]
        judge_model: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Publish {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        repo_id: String,
        #[arg(long)]
        license: String,
        #[arg(long)]
        private: bool,
    },
    Sensitivity {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long, default_value = "2,3,4,8")]
        bits: String,
        #[arg(long, default_value_t = 128)]
        group_size: usize,
        #[arg(long)]
        calibration_file: Option<PathBuf>,
        #[arg(long, default_value = "sensitivity.json")]
        output: PathBuf,
    },
}

#[derive(Clone, Copy)]
struct QuantFormat {
    name: &'static str,
    bits_per_parameter: f64,
    note: &'static str,
}

struct Estimate {
    weights_gib: f64,
    runtime_reserve_gib: f64,
    total_gib: f64,
}

const FORMATS: [QuantFormat; 7] = [
    QuantFormat { name: "Q2_K", bits_per_parameter: 2.6, note: "Very small; quality can drop substantially." },
    QuantFormat { name: "Q3_K_M", bits_per_parameter: 3.9, note: "Compact mixed-bit GGUF format." },
    QuantFormat { name: "Q4_K_M", bits_per_parameter: 4.8, note: "Common quality and size trade-off." },
    QuantFormat { name: "Q5_K_M", bits_per_parameter: 5.7, note: "Higher quality with a larger footprint." },
    QuantFormat { name: "Q6_K", bits_per_parameter: 6.6, note: "Near-high precision GGUF format." },
    QuantFormat { name: "Q8_0", bits_per_parameter: 8.5, note: "High-quality 8-bit GGUF format." },
    QuantFormat { name: "FP16", bits_per_parameter: 16.0, note: "Half-precision weights." },
];

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        process::exit(2);
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Some(Commands::Doctor) => doctor(),
        Some(Commands::Run { config, model, methods, device, output_dir, eval_text, prompt_suite, judge_model, tasks, calibration_file, seed, benchmark, kv_cache_dtype, batch_size, prompt_tokens, generation_tokens, repeats, resume }) => {
            let configuration = config.as_deref().map(load_run_config).transpose()?;
            let model = model.or_else(|| config_string(configuration.as_ref(), "model"))
                .context("Provide --model or set model in the run config")?;
            let methods = methods.or_else(|| config_string(configuration.as_ref(), "methods"))
                .context("Provide --methods or set methods in the run config")?;
            let configured_benchmark = config_string(configuration.as_ref(), "benchmark");
            let output_from_config = config_string(configuration.as_ref(), "output_dir").map(PathBuf::from);
            let output_root = output_dir.or(output_from_config).unwrap_or_else(|| PathBuf::from("quantlab-runs"));
            let device = device.or_else(|| config_string(configuration.as_ref(), "device")).unwrap_or_else(|| "unknown".to_owned());
            let eval_text = eval_text.or_else(|| config_string(configuration.as_ref(), "eval_text").map(PathBuf::from));
            let prompt_suite = prompt_suite.or_else(|| config_string(configuration.as_ref(), "prompt_suite").map(PathBuf::from));
            let judge_model = judge_model.or_else(|| config_string(configuration.as_ref(), "judge_model"));
            let tasks = tasks.or_else(|| config_string(configuration.as_ref(), "tasks"));
            let calibration_file = calibration_file.or_else(|| config_string(configuration.as_ref(), "calibration_file").map(PathBuf::from));
            let seed = seed.or_else(|| config_usize(configuration.as_ref(), "seed").map(|value| value as u64)).unwrap_or(1234);
            let benchmark = benchmark.or(configured_benchmark);
            let kv_cache_dtype = if kv_cache_dtype == "auto" { config_string(configuration.as_ref(), "kv_cache_dtype").unwrap_or(kv_cache_dtype) } else { kv_cache_dtype };
            let batch_size = batch_size.or_else(|| config_usize(configuration.as_ref(), "batch_size")).unwrap_or(if benchmark.as_deref() == Some("llama.cpp") { 2048 } else { 1 });
            let prompt_tokens = if prompt_tokens == 512 { config_usize(configuration.as_ref(), "prompt_tokens").unwrap_or(prompt_tokens) } else { prompt_tokens };
            let generation_tokens = if generation_tokens == 128 { config_usize(configuration.as_ref(), "generation_tokens").unwrap_or(generation_tokens) } else { generation_tokens };
            let repeats = if repeats == 3 { config_usize(configuration.as_ref(), "repeats").unwrap_or(repeats) } else { repeats };
            run_matrix(
                &model, &methods, &device, &output_root,
                eval_text.as_deref(), prompt_suite.as_deref(), judge_model.as_deref(), tasks.as_deref(), calibration_file.as_deref(), seed,
                benchmark.as_deref(), &kv_cache_dtype, batch_size, prompt_tokens, generation_tokens, repeats, resume.as_deref(),
            )
        }
        Some(Commands::Estimate { model, parameters_b, format, device, device_memory_gb, runtime_reserve_gib }) => {
            let parameters = parameter_count(model.as_deref(), parameters_b)?;
            let memory = memory_budget(device.as_deref(), device_memory_gb)?;
            let quant_format = find_format(&format)?;
            let estimate = estimate_memory(parameters, quant_format, runtime_reserve_gib)?;
            println!("Model parameters: {parameters}B");
            println!("Format: {}", quant_format.name);
            println!("Estimated weights: {:.2} GiB", estimate.weights_gib);
            println!("Runtime reserve: {:.2} GiB", estimate.runtime_reserve_gib);
            println!("Estimated minimum: {:.2} GiB", estimate.total_gib);
            println!("Device memory: {memory:.2} GiB");
            println!("Fits estimate: {}", if estimate.total_gib <= memory { "yes" } else { "no" });
            println!("Estimate excludes KV cache and runtime-specific allocations.");
            Ok(())
        }
        Some(Commands::Recommend { model, parameters_b, device, device_memory_gb, runtime_reserve_gib }) => {
            let parameters = parameter_count(model.as_deref(), parameters_b)?;
            let memory = memory_budget(device.as_deref(), device_memory_gb)?;
            if !runtime_reserve_gib.is_finite() || runtime_reserve_gib < 0.0 {
                bail!("Runtime reserve cannot be negative.");
            }
            println!("Model: {}", model.unwrap_or_else(|| format!("{parameters}B parameters")));
            println!("Device memory: {memory:.2} GiB | runtime reserve: {runtime_reserve_gib:.2} GiB");
            println!("{:<12} {:>10} {:>10} {:>10}  NOTES", "FORMAT", "WEIGHTS", "TOTAL", "STATUS");
            let rows = FORMATS.iter().map(|format| {
                let estimate = estimate_memory(parameters, *format, runtime_reserve_gib).expect("validated inputs");
                let fits = estimate.total_gib <= memory;
                (*format, estimate, fits)
            }).collect::<Vec<_>>();
            let best_fit = rows.iter().filter(|(_, _, fits)| *fits).map(|(format, _, _)| format.bits_per_parameter).max_by(f64::total_cmp);
            for (format, estimate, fits) in rows {
                let status = if !fits { "too large" } else if Some(format.bits_per_parameter) == best_fit { "recommended" } else { "fits" };
                println!("{:<12} {:>8.2} GiB {:>8.2} GiB {:>10}  {}", format.name, estimate.weights_gib, estimate.total_gib, status, format.note);
            }
            println!("Estimates exclude KV cache; verify fit with your inference runtime.");
            Ok(())
        }
        Some(Commands::Benchmark {
            model_path, backend, repeats, prompt_tokens, generation_tokens, batch_size,
            kv_cache_dtype, batch_sizes, prompt_token_sizes, generation_token_sizes,
            kv_cache_dtypes, output,
        }) => benchmark_command(
            &model_path, &backend, repeats, prompt_tokens, generation_tokens, batch_size,
            &kv_cache_dtype, batch_sizes.as_deref(), prompt_token_sizes.as_deref(),
            generation_token_sizes.as_deref(), kv_cache_dtypes.as_deref(), output.as_deref(),
        ),
        Some(Commands::Report { run_id, output_dir, output }) => {
            report_command(&run_id, &output_dir, output.as_deref())
        }
        Some(Commands::Evaluate { model_path, text_file, prompt_suite, max_new_tokens, stride, max_length, tasks, limit, seed, judge_model, output }) => {
            evaluate_command(&model_path, text_file.as_deref(), prompt_suite.as_deref(), max_new_tokens, stride, max_length, tasks.as_deref(), limit, seed, judge_model.as_deref(), output.as_deref())
        }
        Some(Commands::Publish { model_path, repo_id, license, private }) => {
            publish_with_hf_cli(&model_path, &repo_id, &license, private)
        }
        Some(Commands::Sensitivity { model_path, bits, group_size, calibration_file, output }) => {
            if let Some(calibration) = calibration_file {
                let result = call_python_adapter("sensitivity", &[
                    model_path.into_os_string(), bits.into(), group_size.to_string().into(),
                    calibration.into_os_string(), output.clone().into_os_string(),
                ])?;
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            sensitivity_safetensors(&model_path, &bits, group_size, &output)
        }
        None => Ok(()),
    }
}

fn doctor() -> Result<()> {
    println!("QuantLab {}", env!("CARGO_PKG_VERSION"));
    println!("Rust toolchain: {}", option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("stable"));
    for tool in ["llama-cli", "llama-quantize", "llama-perplexity", "llama-bench", "llama-server", "nvidia-smi", "hf", "python"] {
        let path = env::split_paths(&env::var_os("PATH").unwrap_or_default())
            .map(|directory| directory.join(if cfg!(windows) { format!("{tool}.exe") } else { tool.to_owned() }))
            .find(|candidate| candidate.is_file());
        println!("{tool}: {}", path.map(|value| value.display().to_string()).unwrap_or_else(|| "not found (optional)".to_owned()));
    }
    Ok(())
}

fn benchmark_command(
    model_path: &str,
    backend: &str,
    repeats: usize,
    prompt_tokens: usize,
    generation_tokens: usize,
    batch_size: Option<usize>,
    kv_cache_dtype: &str,
    batch_sizes: Option<&str>,
    prompt_token_sizes: Option<&str>,
    generation_token_sizes: Option<&str>,
    kv_cache_dtypes: Option<&str>,
    output: Option<&std::path::Path>,
) -> Result<()> {
    if backend == "vllm" {
        let batches = parse_usize_list(batch_sizes, batch_size.unwrap_or(1))?;
        let prompts = parse_usize_list(prompt_token_sizes, prompt_tokens)?;
        let generations = parse_usize_list(generation_token_sizes, generation_tokens)?;
        let dtypes = parse_string_list(kv_cache_dtypes, kv_cache_dtype)?;
        let mut measurements = Vec::new();
        for batch in batches {
            for prompt in &prompts {
                for generation in &generations {
                    for dtype in &dtypes {
                        let result = call_python_adapter("vllm_benchmark", &[
                            model_path.into(), repeats.to_string().into(), prompt.to_string().into(),
                            generation.to_string().into(), dtype.clone().into(), batch.to_string().into(),
                        ])?;
                        measurements.push(json!({"batch_size": batch, "prompt_tokens": prompt,
                            "generation_tokens": generation, "kv_cache_dtype": dtype, "metrics": result}));
                    }
                }
            }
        }
        let rendered = serde_json::to_string_pretty(&json!({"backend": "vllm", "measurements": measurements}))?;
        println!("{rendered}");
        if let Some(path) = output { fs::write(path, format!("{rendered}\n"))?; }
        return Ok(());
    }
    if repeats == 0 {
        bail!("Benchmark repeats must be at least one.");
    }
    let batch_sizes = parse_usize_list(batch_sizes, batch_size.unwrap_or(2048))?;
    let prompt_sizes = parse_usize_list(prompt_token_sizes, prompt_tokens)?;
    let generation_sizes = parse_usize_list(generation_token_sizes, generation_tokens)?;
    let dtypes = parse_string_list(kv_cache_dtypes, kv_cache_dtype)?;
    let mut measurements = Vec::new();
    for batch in &batch_sizes {
        for prompt in &prompt_sizes {
            for generation in &generation_sizes {
                for requested_dtype in &dtypes {
                    let dtype = if requested_dtype == "auto" { "f16" } else { requested_dtype };
                    if !["f16", "q8_0", "q5_0", "q4_0"].contains(&dtype) {
                        bail!("llama.cpp KV cache dtype must be auto, f16, q8_0, q5_0, or q4_0.");
                    }
                    let executable = find_executable("llama-bench")
                        .context("llama-bench was not found on PATH; install llama.cpp")?;
                    let result = Command::new(executable)
                        .args(["-m", model_path, "-o", "json", "-r", &repeats.to_string(),
                            "-p", &prompt.to_string(), "-n", &generation.to_string(),
                            "-b", &batch.to_string(), "-ctk", dtype, "-ctv", dtype])
                        .output()
                        .context("Could not execute llama-bench")?;
                    if !result.status.success() {
                        bail!("llama-bench failed: {}", String::from_utf8_lossy(&result.stderr));
                    }
                    let raw: Value = serde_json::from_slice(&result.stdout)
                        .context("llama-bench returned invalid JSON")?;
                    let records = raw.as_array().or_else(|| raw.get("results").and_then(Value::as_array))
                        .context("llama-bench output did not contain a results array")?;
                    let generation_row = records.iter().find(|row| row["n_gen"].as_u64().unwrap_or(0) > 0)
                        .context("No generation throughput result in llama-bench output")?;
                    let prompt_row = records.iter().find(|row| row["n_prompt"].as_u64().unwrap_or(0) > 0 && row["n_gen"].as_u64().unwrap_or(0) == 0);
                    let mut metrics = json!({
                        "backend": backend,
                        "tok_s": generation_row["avg_ts"],
                        "tok_s_std": generation_row["stddev_ts"],
                        "repeats": repeats,
                        "batch_size": batch,
                        "prompt_tokens": prompt,
                        "generation_tokens": generation,
                        "kv_cache_dtype": dtype,
                    });
                    if let Some(prefill_rate) = prompt_row.and_then(|row| row["avg_ts"].as_f64()) {
                        if prefill_rate > 0.0 {
                            metrics["prefill_tok_s"] = json!(prefill_rate);
                            metrics["prefill_ms_estimate"] = json!(1000.0 * *prompt as f64 / prefill_rate);
                        }
                    }
                    measurements.push(metrics);
                }
            }
        }
    }
    let rendered = serde_json::to_string_pretty(&json!({"results": measurements}))?;
    println!("{rendered}");
    if let Some(path) = output {
        fs::write(path, format!("{rendered}\n"))
            .with_context(|| format!("Could not write benchmark output to {}", path.display()))?;
    }
    Ok(())
}

fn parse_usize_list(value: Option<&str>, default: usize) -> Result<Vec<usize>> {
    let parsed = if let Some(value) = value {
        value.split(',').map(|item| item.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        vec![default]
    };
    if parsed.is_empty() || parsed.iter().any(|value| *value == 0) {
        bail!("Benchmark sweep sizes must be positive comma-separated integers.");
    }
    Ok(parsed)
}

fn parse_string_list(value: Option<&str>, default: &str) -> Result<Vec<String>> {
    let values = value.map(|value| value.split(',').map(|item| item.trim().to_owned()).collect())
        .unwrap_or_else(|| vec![default.to_owned()]);
    if values.is_empty() || values.iter().any(String::is_empty) {
        bail!("KV-cache dtype sweep values cannot be empty.");
    }
    Ok(values)
}

fn report_command(run_id: &str, output_dir: &std::path::Path, output: Option<&std::path::Path>) -> Result<()> {
    validate_run_id(run_id)?;
    let database = Connection::open(output_dir.join("results.sqlite"))
        .with_context(|| format!("Could not open results database in {}", output_dir.display()))?;
    let mut statement = database.prepare(
        "SELECT method, variant, status, perplexity, custom_score, size_gib, batch_size, kv_cache_dtype, tok_s, tok_s_std, prefill_ms_estimate, ttft_ms, error FROM results WHERE run_id = ?1 ORDER BY method, variant"
    )?;
    let rows = statement.query_map([run_id], |row| {
        Ok(ReportRow {
            method: row.get(0)?, variant: row.get(1)?, status: row.get(2)?,
            perplexity: row.get(3)?, custom_score: row.get(4)?, size_gib: row.get(5)?,
            batch_size: row.get(6)?, kv_cache_dtype: row.get(7)?, tok_s: row.get(8)?,
            tok_s_std: row.get(9)?, prefill_ms_estimate: row.get(10)?, ttft_ms: row.get(11)?, error: row.get(12)?,
        })
    })?.collect::<rusqlite::Result<Vec<_>>>()?;
    if rows.is_empty() {
        bail!("No results found for run {run_id:?}.");
    }
    let frontier = pareto_keys(&rows);
    let table = rows.iter().map(|row| {
        let key = (row.method.as_str(), row.variant.as_str());
        let is_frontier = frontier.contains(&key);
        let columns = [
            row.method.clone(), row.variant.clone(), row.status.clone(), fmt_opt(row.perplexity),
            fmt_opt(row.custom_score), fmt_opt(row.size_gib), row.batch_size.map(|value| value.to_string()).unwrap_or_else(|| "—".into()),
            row.kv_cache_dtype.clone().unwrap_or_else(|| "—".into()), fmt_opt(row.tok_s), fmt_opt(row.tok_s_std),
            fmt_opt(row.prefill_ms_estimate), fmt_opt(row.ttft_ms), row.error.clone().unwrap_or_else(|| "—".into()),
            if is_frontier { "yes".into() } else { "".into() },
        ];
        format!("<tr>{}</tr>", columns.iter().map(|value| format!("<td>{}</td>", escape_html(value))).collect::<String>())
    }).collect::<String>();
    let html = format!("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>QuantLab {}</title><style>body{{font:15px system-ui;background:#10131a;color:#e5e7eb;padding:2rem}}table{{border-collapse:collapse}}td,th{{padding:.6rem;border-bottom:1px solid #394150;text-align:left}}th{{color:#a3e635}}</style><h1>QuantLab run {}</h1><table><thead><tr><th>Method</th><th>Variant</th><th>Status</th><th>PPL</th><th>Custom score</th><th>Size GiB</th><th>Batch</th><th>KV dtype</th><th>tok/s</th><th>tok/s std</th><th>Prefill ms</th><th>TTFT ms</th><th>Error</th><th>Pareto</th></tr></thead><tbody>{}</tbody></table></html>", escape_html(run_id), escape_html(run_id), table);
    let path = output.map(PathBuf::from).unwrap_or_else(|| output_dir.join(format!("{run_id}.html")));
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, html).with_context(|| format!("Could not write report to {}", path.display()))?;
    println!("{}", path.display());
    Ok(())
}

struct ReportRow {
    method: String,
    variant: String,
    status: String,
    perplexity: Option<f64>,
    custom_score: Option<f64>,
    size_gib: Option<f64>,
    batch_size: Option<i64>,
    kv_cache_dtype: Option<String>,
    tok_s: Option<f64>,
    tok_s_std: Option<f64>,
    prefill_ms_estimate: Option<f64>,
    ttft_ms: Option<f64>,
    error: Option<String>,
}

fn pareto_keys(rows: &[ReportRow]) -> std::collections::HashSet<(&str, &str)> {
    rows.iter().filter(|candidate| candidate.status == "completed").filter(|candidate| {
        let (Some(ppl), Some(size), Some(speed)) = (candidate.perplexity, candidate.size_gib, candidate.tok_s) else { return true; };
        !rows.iter().any(|other| {
            if other.method == candidate.method && other.variant == candidate.variant { return false; }
            let (Some(other_ppl), Some(other_size), Some(other_speed)) = (other.perplexity, other.size_gib, other.tok_s) else { return false; };
            other.status == "completed" && other_ppl <= ppl && other_size <= size && other_speed >= speed
                && (other_ppl < ppl || other_size < size || other_speed > speed)
        })
    }).map(|row| (row.method.as_str(), row.variant.as_str())).collect()
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.3}")).unwrap_or_else(|| "—".into())
}

fn escape_html(value: &str) -> String {
    value.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#39;")
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    env::split_paths(&env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join(format!("{name}{suffix}")))
        .find(|candidate| candidate.is_file())
}

fn evaluate_command(
    model: &std::path::Path, text_file: Option<&std::path::Path>, prompt_suite: Option<&std::path::Path>,
    max_new_tokens: usize, stride: usize, max_length: Option<usize>, tasks: Option<&str>, limit: Option<usize>,
    seed: u64, judge_model: Option<&str>, output_path: Option<&std::path::Path>,
) -> Result<()> {
    if text_file.is_none() && prompt_suite.is_none() && tasks.is_none() {
        bail!("Provide --text-file, --prompt-suite, --tasks, or a combination.");
    }
    let gguf = model.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
    let mut result = json!({});
    if gguf {
        result = evaluate_gguf(model, text_file, if judge_model.is_none() { prompt_suite } else { None }, max_new_tokens, max_length)?;
        if let (Some(suite), Some(judge)) = (prompt_suite, judge_model) {
            result["custom"] = call_python_adapter("prompt_suite", &[
                model.as_os_str().to_owned(), suite.as_os_str().to_owned(), max_new_tokens.to_string().into(), judge.into(),
            ])?;
        }
        if let Some(task_list) = tasks {
            result["tasks"] = call_python_adapter("tasks", &[
                model.as_os_str().to_owned(), task_list.into(), limit.map(|value| value.to_string()).unwrap_or_default().into(), seed.to_string().into(),
            ])?;
        }
        let rendered = serde_json::to_string_pretty(&result)?;
        println!("{rendered}");
        if let Some(path) = output_path { fs::write(path, format!("{rendered}\n"))?; }
        return Ok(());
    }
    if candle_backend::supports_local_model(model) {
        if let Some(corpus) = text_file {
            result["perplexity"] = candle_backend::evaluate_perplexity(model, corpus, stride, max_length)?;
        }
        if let Some(suite) = prompt_suite {
            if let Some(judge) = judge_model.filter(|judge| !candle_backend::supports_local_model(std::path::Path::new(judge))) {
                result["custom"] = call_python_adapter("prompt_suite", &[
                    model.as_os_str().to_owned(), suite.as_os_str().to_owned(), max_new_tokens.to_string().into(), judge.into(),
                ])?;
            } else {
                let prompts: Value = serde_json::from_slice(&fs::read(suite).context("Could not read prompt suite")?)?;
                result["custom"] = candle_backend::evaluate_prompt_suite(
                    model,
                    &prompts,
                    max_new_tokens,
                    judge_model.map(std::path::Path::new),
                )?;
            }
        }
        if let Some(task_list) = tasks {
            result["tasks"] = call_python_adapter("tasks", &[
                model.as_os_str().to_owned(), task_list.into(), limit.map(|value| value.to_string()).unwrap_or_default().into(), seed.to_string().into(),
            ])?;
        }
        let rendered = serde_json::to_string_pretty(&result)?;
        println!("{rendered}");
        if let Some(path) = output_path { fs::write(path, format!("{rendered}\n"))?; }
        return Ok(());
    }
    if let Some(corpus) = text_file {
        result["perplexity"] = call_python_adapter("perplexity", &[
            model.as_os_str().to_owned(), corpus.as_os_str().to_owned(), stride.to_string().into(),
            max_length.map(|value| value.to_string()).unwrap_or_default().into(),
        ])?;
    }
    if let Some(suite) = prompt_suite {
        result["custom"] = call_python_adapter("prompt_suite", &[
            model.as_os_str().to_owned(), suite.as_os_str().to_owned(), max_new_tokens.to_string().into(),
            judge_model.unwrap_or_default().into(),
        ])?;
    }
    if let Some(task_list) = tasks {
        result["tasks"] = call_python_adapter("tasks", &[
            model.as_os_str().to_owned(), task_list.into(), limit.map(|value| value.to_string()).unwrap_or_default().into(), seed.to_string().into(),
        ])?;
    }
    let rendered = serde_json::to_string_pretty(&result)?;
    println!("{rendered}");
    if let Some(path) = output_path { fs::write(path, format!("{rendered}\n"))?; }
    Ok(())
}

fn evaluate_gguf(
    model: &std::path::Path,
    text_file: Option<&std::path::Path>,
    prompt_suite: Option<&std::path::Path>,
    max_new_tokens: usize,
    max_length: Option<usize>,
) -> Result<Value> {
    if text_file.is_none() && prompt_suite.is_none() {
        bail!("Provide --text-file, --prompt-suite, or both.");
    }
    if max_new_tokens == 0 {
        bail!("max-new-tokens must be at least one.");
    }
    let mut results = json!({});
    if let Some(corpus) = text_file {
        let executable = find_executable("llama-perplexity").context("llama-perplexity was not found on PATH")?;
        let measured = Command::new(executable).args(["-m", &model.to_string_lossy(), "-f", &corpus.to_string_lossy(), "-c", &max_length.unwrap_or(2048).to_string()]).output()?;
        if !measured.status.success() {
            bail!("llama-perplexity failed: {}", String::from_utf8_lossy(&measured.stderr));
        }
        let text = format!("{}\n{}", String::from_utf8_lossy(&measured.stdout), String::from_utf8_lossy(&measured.stderr));
        let perplexity = parse_perplexity(&text).context("Could not find perplexity in llama-perplexity output")?;
        results["perplexity"] = json!({"perplexity": perplexity, "tokens": 0});
    }
    if let Some(suite_path) = prompt_suite {
        let suite: Value = serde_json::from_slice(&fs::read(suite_path).context("Could not read prompt suite")?)?;
        let items = suite.as_array().context("Prompt suite must be a JSON array")?;
        if items.is_empty() { bail!("Prompt suite must contain at least one prompt."); }
        let executable = find_executable("llama-cli").context("llama-cli was not found on PATH")?;
        let mut outputs = Vec::new();
        let mut exact_matches = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let prompt = item.get("prompt").and_then(Value::as_str).context("Each prompt-suite item needs a string 'prompt'")?;
            let generated = Command::new(&executable).args(["-m", &model.to_string_lossy(), "-p", prompt, "-n", &max_new_tokens.to_string(), "--simple-io", "--no-display-prompt"]).output()?;
            if !generated.status.success() {
                bail!("llama-cli failed for prompt {index}: {}", String::from_utf8_lossy(&generated.stderr));
            }
            let completion = String::from_utf8_lossy(&generated.stdout).trim().to_owned();
            let expected = item.get("expected").and_then(Value::as_str);
            if let Some(expected) = expected {
                exact_matches.push(completion.trim().eq_ignore_ascii_case(expected.trim()));
            }
            let exact_match = expected.map(|value| completion.trim().eq_ignore_ascii_case(value.trim()));
            outputs.push(json!({
                "id": item.get("id").cloned().unwrap_or_else(|| json!(index.to_string())),
                "completion": completion,
                "exact_match": exact_match,
                "judge_pass": null,
            }));
        }
        let score = if exact_matches.is_empty() { Value::Null } else { json!(exact_matches.iter().filter(|value| **value).count() as f64 / exact_matches.len() as f64) };
        results["custom"] = json!({
            "count": outputs.len(),
            "exact_match_score": score,
            "scored_count": exact_matches.len(),
            "judge_score": null,
            "judge_scored_count": 0,
            "results": outputs,
        });
    }
    Ok(results)
}

fn parse_perplexity(output: &str) -> Option<f64> {
    output.lines().rev().find_map(|line| {
        let lower = line.to_ascii_lowercase();
        let marker = lower.find("perplexity").map(|index| index + "perplexity".len())
            .or_else(|| lower.find("ppl").map(|index| index + 3))?;
        line[marker..].split(|character: char| !(character.is_ascii_digit() || character == '.' || character == '-'))
            .find_map(|value| value.parse::<f64>().ok()).filter(|value| value.is_finite() && *value > 0.0)
    })
}

fn publish_with_hf_cli(model_path: &std::path::Path, repo_id: &str, license: &str, private: bool) -> Result<()> {
    if !model_path.exists() { bail!("Model path does not exist: {}", model_path.display()); }
    if env::var_os("HF_TOKEN").is_none() { bail!("Set HF_TOKEN in the environment before publishing."); }
    if !license.chars().all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '-')) || license.is_empty() {
        bail!("Provide a simple SPDX license identifier.");
    }
    let (owner, name) = repo_id.split_once('/').context("Repository ID must have the form owner/name")?;
    if owner.is_empty() || name.is_empty() || name.contains('/') || !repo_id.chars().all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_' | '/')) {
        bail!("Invalid Hugging Face repository ID.");
    }
    let hf = find_executable("hf").context("Hugging Face CLI 'hf' was not found; install huggingface_hub")?;
    let mut create = Command::new(&hf);
    create.args(["repo", "create", repo_id, "--type", "model", "--exist-ok"]);
    if private { create.arg("--private"); } else { create.arg("--public"); }
    let created = create.output().context("Could not start Hugging Face CLI")?;
    if !created.status.success() {
        bail!("Could not create or access Hub repository: {}", String::from_utf8_lossy(&created.stderr));
    }
    let mut upload = Command::new(&hf);
    upload.arg("upload").arg(repo_id).arg(model_path);
    if model_path.is_file() {
        upload.arg(model_path.file_name().context("Model file has no filename")?);
    } else {
        upload.arg(".");
    }
    upload.arg("--commit-message").arg("Publish quantized model with QuantLab");
    let uploaded = upload.output().context("Could not start model upload")?;
    if !uploaded.status.success() {
        bail!("Hub upload failed: {}", String::from_utf8_lossy(&uploaded.stderr));
    }
    let has_card = model_path.is_dir() && model_path.join("README.md").is_file();
    if !has_card {
        let card_path = env::temp_dir().join(format!("quantlab-model-card-{}.md", process::id()));
        let card = format!("---\nlicense: {license}\nlibrary_name: quantlab\ntags:\n  - quantization\n  - quantlab\n---\n\n# {name}\n\nQuantized model artifact published with QuantLab.\n\nConfirm the upstream model and calibration data licenses permit redistribution. Document the source model, quantization settings, and evaluation results before sharing.\n");
        fs::write(&card_path, card)?;
        let card_upload = Command::new(&hf).args(["upload", repo_id]).arg(&card_path).arg("README.md").args(["--commit-message", "Add QuantLab model card"]).output()?;
        let _ = fs::remove_file(&card_path);
        if !card_upload.status.success() {
            bail!("Hub model-card upload failed: {}", String::from_utf8_lossy(&card_upload.stderr));
        }
    }
    println!("https://huggingface.co/{repo_id}");
    Ok(())
}

fn sensitivity_safetensors(model_path: &std::path::Path, bits: &str, group_size: usize, output: &std::path::Path) -> Result<()> {
    if group_size == 0 { bail!("Group size must be positive."); }
    let bits = bits.split(',').map(|item| item.trim().parse::<u8>()).collect::<std::result::Result<Vec<_>, _>>()?;
    if bits.is_empty() || bits.iter().any(|bit| !(2..=8).contains(bit)) { bail!("Specify bit widths from 2 to 8."); }
    let mut files = Vec::new();
    collect_safetensors(model_path, &mut files)?;
    if files.is_empty() { bail!("No .safetensors files found; this native analysis requires Hugging Face safetensors weights."); }
    let mut layers = Vec::new();
    for file in files { analyze_safetensors_file(&file, group_size, &bits, &mut layers)?; }
    if layers.is_empty() { bail!("No supported two-dimensional floating-point tensors were found."); }
    layers.sort_by(|left, right| {
        right["errors"]["4"].as_f64().unwrap_or_default().total_cmp(&left["errors"]["4"].as_f64().unwrap_or_default())
    });
    let suggestion_count = std::cmp::max(1, layers.len() / 10);
    let suggested_bits = *bits.iter().max().context("No precision widths supplied")?;
    let suggestions = layers.iter().take(suggestion_count).map(|layer| json!({
        "layer": layer["name"],
        "recommended_bits": suggested_bits,
        "reason": "highest simulated groupwise weight quantization error",
    })).collect::<Vec<_>>();
    let result = json!({
        "model": model_path,
        "group_size": group_size,
        "method": "weight_only_safetensors_simulation",
        "layers": layers,
        "mixed_precision_suggestions": suggestions,
    });
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) { fs::create_dir_all(parent)?; }
    fs::write(output, serde_json::to_string_pretty(&result)?)?;
    println!("Sensitivity report: {} ({} layers)", output.display(), result["layers"].as_array().map_or(0, Vec::len));
    Ok(())
}

fn collect_safetensors(path: &std::path::Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") { files.push(path.to_path_buf()); }
        return Ok(());
    }
    for item in fs::read_dir(path).with_context(|| format!("Could not read model directory {}", path.display()))? {
        let item = item?;
        if item.file_type()?.is_dir() { collect_safetensors(&item.path(), files)?; }
        else if item.path().extension().and_then(|ext| ext.to_str()) == Some("safetensors") { files.push(item.path()); }
    }
    Ok(())
}

fn analyze_safetensors_file(path: &std::path::Path, group_size: usize, bits: &[u8], layers: &mut Vec<Value>) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(path)?;
    let mut length_bytes = [0; 8];
    file.read_exact(&mut length_bytes)?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len > 128 * 1024 * 1024 { bail!("Safetensors header is unexpectedly large in {}", path.display()); }
    let mut header_bytes = vec![0; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let header: Value = serde_json::from_slice(&header_bytes).context("Invalid safetensors header")?;
    let data_start = 8 + header_len;
    let file_len = file.metadata()?.len();
    for (name, tensor) in header.as_object().context("Safetensors header must be an object")? {
        if name == "__metadata__" || tensor["shape"].as_array().map_or(true, |shape| shape.len() != 2) { continue; }
        let dtype = tensor["dtype"].as_str().unwrap_or_default();
        let element_bytes = match dtype { "F16" | "BF16" => 2usize, "F32" => 4, "F64" => 8, _ => continue };
        let shape = tensor["shape"].as_array().unwrap();
        let rows = shape[0].as_u64().context("Invalid tensor row count")?;
        let columns = shape[1].as_u64().context("Invalid tensor column count")?;
        if rows == 0 || columns == 0 { continue; }
        let offsets = tensor["data_offsets"].as_array().context("Tensor missing data_offsets")?;
        let start = offsets[0].as_u64().context("Invalid tensor offset")?;
        let end = offsets[1].as_u64().context("Invalid tensor offset")?;
        let count = rows.checked_mul(columns).context("Tensor element count overflow")?;
        let byte_count = count.checked_mul(element_bytes as u64).context("Tensor byte count overflow")?;
        if end < start || end - start != byte_count || data_start.checked_add(end).map_or(true, |position| position > file_len) {
            bail!("Invalid tensor storage range for {name} in {}", path.display());
        }
        let mut totals = bits.iter().map(|bit| (*bit, (0.0f64, 0u64))).collect::<std::collections::HashMap<_, _>>();
        let mut index = 0u64;
        while index < count {
            let take = std::cmp::min(group_size as u64, count - index) as usize;
            let mut data = vec![0u8; take * element_bytes];
            file.seek(SeekFrom::Start(data_start + start + index * element_bytes as u64))?;
            file.read_exact(&mut data)?;
            let values = decode_tensor_values(&data, dtype)?;
            let max_abs = values.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            for bit in bits {
                let quant_max = ((1u32 << (*bit - 1)) - 1) as f32;
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / quant_max };
                let error = values.iter().map(|value| {
                    let quantized = ((value / scale).round().clamp(-quant_max, quant_max)) * scale;
                    let delta = *value - quantized;
                    (delta as f64) * (delta as f64)
                }).sum::<f64>();
                let (sum, elements) = totals.get_mut(bit).unwrap();
                *sum += error;
                *elements += values.len() as u64;
            }
            index += take as u64;
        }
        let errors = totals.into_iter().map(|(bit, (sum, count))| (bit.to_string(), json!(sum / count.max(1) as f64))).collect::<serde_json::Map<_, _>>();
        layers.push(json!({
            "name": name.strip_suffix(".weight").unwrap_or(name),
            "shape": shape,
            "errors": errors,
            "sensitivity_method": "weight_only",
            "source_file": path.file_name().map(|value| value.to_string_lossy()),
        }));
    }
    Ok(())
}

fn decode_tensor_values(data: &[u8], dtype: &str) -> Result<Vec<f32>> {
    let width = match dtype { "F16" | "BF16" => 2, "F32" => 4, "F64" => 8, _ => bail!("Unsupported floating point type {dtype}") };
    if data.len() % width != 0 { bail!("Tensor data length is not aligned to its element type."); }
    Ok(data.chunks_exact(width).map(|chunk| match dtype {
        "F16" => f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
        "BF16" => f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16),
        "F32" => f32::from_le_bytes(chunk.try_into().unwrap()),
        "F64" => f64::from_le_bytes(chunk.try_into().unwrap()) as f32,
        _ => unreachable!(),
    }).collect())
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let converted = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let normalized = (fraction as f32) * 2.0f32.powi(-24);
            return if sign == 0 { normalized } else { -normalized };
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(converted)
}

fn validate_run_id(run_id: &str) -> Result<()> {
    if run_id.is_empty() || run_id.len() > 64 || !run_id.as_bytes()[0].is_ascii_alphanumeric()
        || !run_id.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)) {
        bail!("Run ID must be 1-64 letters, numbers, dots, underscores, or hyphens and start with a letter or number.");
    }
    Ok(())
}

fn load_run_config(path: &std::path::Path) -> Result<Value> {
    let source = fs::read_to_string(path).with_context(|| format!("Could not read config {}", path.display()))?;
    let value = if path.extension().and_then(|extension| extension.to_str()).is_some_and(|extension| extension.eq_ignore_ascii_case("json")) {
        serde_json::from_str::<Value>(&source)?
    } else {
        parse_simple_yaml(&source)?
    };
    let object = value.as_object().context("Run configuration must be an object")?;
    let keys = ["model", "methods", "device", "output_dir", "eval_text", "prompt_suite", "judge_model", "calibration_file", "tasks", "benchmark", "kv_cache_dtype", "repeats", "seed", "batch_size", "prompt_tokens", "generation_tokens"];
    if let Some(key) = object.keys().find(|key| !keys.contains(&key.as_str())) {
        bail!("Unknown configuration key {key:?}.");
    }
    Ok(value)
}

fn parse_simple_yaml(source: &str) -> Result<Value> {
    let mut values = serde_json::Map::new();
    let mut active_list: Option<String> = None;
    for (line_index, raw_line) in source.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        if let Some(item) = line.strip_prefix("- ") {
            let key = active_list.as_ref().context("YAML list item is not under a key")?;
            let list = values.get_mut(key).and_then(Value::as_array_mut).context("Invalid YAML list value")?;
            list.push(parse_yaml_scalar(item.trim())?);
            continue;
        }
        let (key, value) = line.split_once(':').with_context(|| format!("Invalid YAML config at line {}", line_index + 1))?;
        let key = key.trim();
        if key.is_empty() { bail!("Empty YAML key at line {}", line_index + 1); }
        let value = value.trim();
        if value.is_empty() {
            values.insert(key.to_owned(), Value::Array(Vec::new()));
            active_list = Some(key.to_owned());
        } else {
            values.insert(key.to_owned(), parse_yaml_scalar(value)?);
            active_list = None;
        }
    }
    Ok(Value::Object(values))
}

fn parse_yaml_scalar(value: &str) -> Result<Value> {
    if value.starts_with('"') || value.starts_with('[') || value.starts_with('{') {
        return Ok(serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.trim_matches('"').to_owned())));
    }
    if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
        return Ok(Value::String(value[1..value.len() - 1].replace("''", "'")));
    }
    match value {
        "true" => Ok(Value::Bool(true)),
        "false" => Ok(Value::Bool(false)),
        "null" | "~" => Ok(Value::Null),
        _ => Ok(value.parse::<i64>().map(|number| json!(number))
            .or_else(|_| value.parse::<f64>().map(|number| json!(number)))
            .unwrap_or_else(|_| Value::String(value.to_owned()))),
    }
}

fn config_string(configuration: Option<&Value>, key: &str) -> Option<String> {
    match configuration?.get(key)? {
        Value::String(value) => Some(value.clone()),
        Value::Array(values) => Some(values.iter().map(|value| value.as_str().map(str::to_owned).unwrap_or_else(|| value.to_string())).collect::<Vec<_>>().join(",")),
        _ => None,
    }
}

fn config_usize(configuration: Option<&Value>, key: &str) -> Option<usize> {
    configuration?.get(key)?.as_u64().and_then(|value| usize::try_from(value).ok())
}

fn run_matrix(
    model: &str,
    methods_spec: &str,
    device: &str,
    output_root: &std::path::Path,
    eval_text: Option<&std::path::Path>,
    prompt_suite: Option<&std::path::Path>,
    judge_model: Option<&str>,
    tasks: Option<&str>,
    calibration_file: Option<&std::path::Path>,
    seed: u64,
    benchmark: Option<&str>,
    kv_cache_dtype: &str,
    batch_size: usize,
    prompt_tokens: usize,
    generation_tokens: usize,
    repeats: usize,
    requested_run_id: Option<&str>,
) -> Result<()> {
    if batch_size == 0 || prompt_tokens == 0 || generation_tokens == 0 || repeats == 0 {
        bail!("Batch size, token counts, and repeats must be positive.");
    }
    if benchmark.is_some_and(|backend| !["llama.cpp", "vllm"].contains(&backend)) {
        bail!("Benchmark backend must be llama.cpp or vllm.");
    }
    let methods = methods_spec.split(',').map(|item| {
        let (method, variant) = item.trim().split_once(':').context("Methods must use method:variant syntax")?;
        let method = method.trim().to_ascii_lowercase();
        if variant.trim().is_empty() || !["gguf", "bnb", "gptq", "awq", "fp8"].contains(&method.as_str()) {
            bail!("Unsupported method entry {item:?}; choose gguf, bnb, gptq, awq, or fp8.");
        }
        let variant = if method == "gguf" || method == "gptq" || method == "awq" { variant.trim().to_ascii_uppercase() } else { variant.trim().to_ascii_lowercase() };
        if method == "gguf" && !["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"].contains(&variant.as_str()) {
            bail!("Unsupported GGUF quantization {variant:?}.");
        }
        Ok((method, variant))
    }).collect::<Result<Vec<_>>>()?;
    if methods.is_empty() || methods.iter().collect::<std::collections::HashSet<_>>().len() != methods.len() {
        bail!("Specify a non-empty method list without duplicate variants.");
    }
    if benchmark == Some("llama.cpp") && methods.iter().any(|(method, _)| method != "gguf") {
        bail!("The llama.cpp benchmark backend only supports GGUF artifacts; select vllm for other quantization methods.");
    }
    fs::create_dir_all(output_root)?;
    let resolved_model = if methods.iter().any(|(method, _)| method == "gguf") { Some(resolve_model_path(model, output_root)?) } else { None };
    let run_id = requested_run_id.map(str::to_owned).unwrap_or_else(|| {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        format!("run-{nanos:x}")
    });
    validate_run_id(&run_id)?;
    let run_dir = output_root.join(&run_id);
    fs::create_dir_all(&run_dir)?;
    let database = Connection::open(output_root.join("results.sqlite"))?;
    create_run_schema(&database)?;
    let config = json!({
        "model": model,
        "device": device,
        "methods": methods,
        "eval_text": eval_text.map(|path| path.to_string_lossy()),
        "benchmark_backend": benchmark,
        "kv_cache_dtype": kv_cache_dtype,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "repeats": repeats,
        "seed": seed,
        "judge_model": judge_model,
        "prompt_suite": prompt_suite.map(|path| path.to_string_lossy()),
        "tasks": tasks,
        "calibration_file": calibration_file.map(|path| path.to_string_lossy()),
    });
    let existing_config: Option<String> = database.query_row("SELECT config_json FROM runs WHERE run_id=?1", [&run_id], |row| row.get(0)).optional()?;
    if let Some(previous) = existing_config {
        let previous: Value = serde_json::from_str(&previous)?;
        if previous != config {
            bail!("Run {run_id:?} was created with a different configuration.");
        }
        println!("Resuming run {run_id}; completed artifacts will be reused where available.");
    } else {
        database.execute("INSERT INTO runs (run_id, model, device, config_json, created_at) VALUES (?1, ?2, ?3, ?4, datetime('now'))", rusqlite::params![run_id, model, device, config.to_string()])?;
    }
    let mut failures = 0;
    for (method, variant) in methods {
        let existing: Option<(String, Option<String>)> = database.query_row(
            "SELECT status, artifact_path FROM results WHERE run_id=?1 AND method=?2 AND variant=?3",
            rusqlite::params![run_id, method, variant], |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?;
        if let Some((status, Some(artifact))) = &existing {
            if status == "completed" && PathBuf::from(artifact).exists() {
                println!("Skip {method}:{variant} (already complete)");
                continue;
            }
        }
        let artifact_dir = run_dir.join(format!("{method}-{variant}"));
        let _ = fs::create_dir_all(&artifact_dir);
        store_run_result(&database, &run_id, &method, &variant, "running", None, None, None, None, None, None, None, None)?;
        let attempt = (|| -> Result<(PathBuf, Option<f64>, Option<f64>, Option<String>, Option<Value>)> {
            let artifact = if method == "gguf" && calibration_file.is_none() {
                quantize_gguf_native(&resolved_model.as_ref().unwrap().to_string_lossy(), &variant, &artifact_dir)?
            } else {
                let quantized = call_python_adapter("quantize", &[
                    model.into(), method.clone().into(), variant.clone().into(), artifact_dir.clone().into_os_string(),
                    calibration_file.map(|path| path.as_os_str().to_owned()).unwrap_or_default(),
                ])?;
                PathBuf::from(quantized["artifact_path"].as_str().context("Quantizer adapter omitted artifact_path")?)
            };
            let perplexity = if let Some(path) = eval_text {
                if method == "gguf" { Some(perplexity_gguf(&artifact, path)?) }
                else { Some(call_python_adapter("perplexity", &[artifact.clone().into_os_string(), path.as_os_str().to_owned(), "512".into(), "".into()])?["perplexity"].as_f64().context("Perplexity adapter omitted score")?) }
            } else { None };
            let prompt_result = if let Some(suite) = prompt_suite {
                if method == "gguf" && judge_model.is_none() {
                    Some(evaluate_gguf(&artifact, None, Some(suite), 128, None)?["custom"].clone())
                } else {
                    Some(call_python_adapter("prompt_suite", &[artifact.clone().into_os_string(), suite.as_os_str().to_owned(), "128".into(), judge_model.unwrap_or_default().into()])?)
                }
            } else { None };
            let task_result = if let Some(task_list) = tasks {
                Some(call_python_adapter("tasks", &[artifact.clone().into_os_string(), task_list.into(), "".into(), seed.to_string().into()])?)
            } else { None };
            let custom_score = prompt_result.as_ref().and_then(|value| value["judge_score"].as_f64().or_else(|| value["exact_match_score"].as_f64()));
            let task_scores = task_result.as_ref().map(|value| value["results"].to_string());
            let metrics = if benchmark == Some("llama.cpp") && method == "gguf" {
                Some(llama_bench_one(&artifact, repeats, prompt_tokens, generation_tokens, kv_cache_dtype, batch_size)?)
            } else if benchmark == Some("vllm") {
                Some(call_python_adapter("vllm_benchmark", &[artifact.clone().into_os_string(), repeats.to_string().into(), prompt_tokens.to_string().into(), generation_tokens.to_string().into(), kv_cache_dtype.into(), batch_size.to_string().into()])?)
            } else { None };
            Ok((artifact, perplexity, custom_score, task_scores, metrics))
        })();
        match attempt {
            Ok((artifact, perplexity, custom_score, task_scores, metrics)) => {
                let cache_dtype = if benchmark.is_some() { Some(kv_cache_dtype) } else { None };
                store_run_result(&database, &run_id, &method, &variant, "completed", Some(&artifact), Some(artifact_size(&artifact)? as f64 / 1024_f64.powi(3)), perplexity, custom_score, task_scores.as_deref(), cache_dtype, if metrics.is_some() { Some(batch_size as i64) } else { None }, metrics.as_ref())?;
                println!("Completed {method}:{variant} → {}", artifact.display());
            }
            Err(error) => {
                failures += 1;
                store_run_result(&database, &run_id, &method, &variant, "failed", None, None, None, None, None, None, None, None)?;
                database.execute("UPDATE results SET error=?1 WHERE run_id=?2 AND method=?3 AND variant=?4", rusqlite::params![format!("{error:#}"), run_id, method, variant])?;
                eprintln!("Failed {method}:{variant}: {error:#}");
            }
        }
    }
    report_command(&run_id, output_root, None)?;
    println!("Run ID: {run_id}");
    println!("Results database: {}", output_root.join("results.sqlite").display());
    if failures > 0 { bail!("{failures} quantization matrix item(s) failed."); }
    Ok(())
}

fn resolve_model_path(model: &str, output_root: &std::path::Path) -> Result<PathBuf> {
    let local = PathBuf::from(model);
    if local.is_file() || local.is_dir() { return Ok(local); }
    if !model.contains('/') { bail!("Model path does not exist: {model}"); }
    let hf = find_executable("hf").context("Model ID download requires the Hugging Face CLI ('hf')")?;
    let cache = output_root.join("model-cache").join(model.replace('/', "--"));
    fs::create_dir_all(&cache)?;
    let downloaded = Command::new(hf).args(["download", model, "--local-dir"]).arg(&cache).output()
        .context("Could not start Hugging Face model download")?;
    if !downloaded.status.success() {
        bail!("Hugging Face model download failed: {}", String::from_utf8_lossy(&downloaded.stderr));
    }
    Ok(cache)
}

fn create_run_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, model TEXT NOT NULL, device TEXT NOT NULL, config_json TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS results (run_id TEXT NOT NULL REFERENCES runs(run_id), method TEXT NOT NULL, variant TEXT NOT NULL, status TEXT NOT NULL, artifact_path TEXT, size_gib REAL, perplexity REAL, custom_score REAL, task_scores_json TEXT, kv_cache_dtype TEXT, batch_size INTEGER, tok_s REAL, tok_s_std REAL, prefill_tok_s REAL, prefill_ms_estimate REAL, ttft_ms REAL, ttft_ms_std REAL, peak_mem_gib REAL, error TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(run_id, method, variant));")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn store_run_result(
    connection: &Connection, run_id: &str, method: &str, variant: &str, status: &str,
    artifact_path: Option<&std::path::Path>, size_gib: Option<f64>, perplexity: Option<f64>,
    custom_score: Option<f64>, task_scores_json: Option<&str>, kv_cache_dtype: Option<&str>,
    batch_size: Option<i64>, metrics: Option<&Value>,
) -> Result<()> {
    let metric = |key: &str| metrics.and_then(|value| value[key].as_f64());
    connection.execute("INSERT INTO results (run_id, method, variant, status, artifact_path, size_gib, perplexity, custom_score, task_scores_json, kv_cache_dtype, batch_size, tok_s, tok_s_std, prefill_tok_s, prefill_ms_estimate, ttft_ms, ttft_ms_std, peak_mem_gib, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, datetime('now')) ON CONFLICT(run_id, method, variant) DO UPDATE SET status=excluded.status, artifact_path=excluded.artifact_path, size_gib=excluded.size_gib, perplexity=excluded.perplexity, custom_score=excluded.custom_score, task_scores_json=excluded.task_scores_json, kv_cache_dtype=excluded.kv_cache_dtype, batch_size=excluded.batch_size, tok_s=excluded.tok_s, tok_s_std=excluded.tok_s_std, prefill_tok_s=excluded.prefill_tok_s, prefill_ms_estimate=excluded.prefill_ms_estimate, ttft_ms=excluded.ttft_ms, ttft_ms_std=excluded.ttft_ms_std, peak_mem_gib=excluded.peak_mem_gib, error=NULL, updated_at=excluded.updated_at",
        rusqlite::params![run_id, method, variant, status, artifact_path.map(|path| path.to_string_lossy().into_owned()), size_gib, perplexity, custom_score, task_scores_json, kv_cache_dtype, batch_size, metric("tok_s"), metric("tok_s_std"), metric("prefill_tok_s"), metric("prefill_ms_estimate"), metric("ttft_ms"), metric("ttft_ms_std"), metric("peak_mem_gib")])?;
    Ok(())
}

fn quantize_gguf_native(model: &str, variant: &str, output_dir: &std::path::Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir)?;
    let source = PathBuf::from(model);
    let input = if source.is_file() && source.extension().and_then(|value| value.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")) {
        source
    } else if source.is_dir() {
        let llama_dir = env::var_os("LLAMA_CPP_DIR").map(PathBuf::from).context("Set LLAMA_CPP_DIR to the llama.cpp source directory to convert Hugging Face model folders")?;
        let converter = [llama_dir.join("convert_hf_to_gguf.py"), llama_dir.join("convert").join("convert_hf_to_gguf.py")].into_iter().find(|path| path.is_file()).context("convert_hf_to_gguf.py was not found under LLAMA_CPP_DIR")?;
        let fp16 = output_dir.join("model-f16.gguf");
        let python = env::var_os("PYTHON").unwrap_or_else(|| "python".into());
        run_logged(Command::new(python).arg(converter).arg(&source).arg("--outfile").arg(&fp16).arg("--outtype").arg("f16"), &output_dir.join("convert.log"))?;
        fp16
    } else {
        bail!("Model must be an existing GGUF file or local Hugging Face model folder.");
    };
    let quantizer = find_executable("llama-quantize").context("llama-quantize was not found on PATH")?;
    let destination = output_dir.join(format!("model-{variant}.gguf"));
    run_logged(Command::new(quantizer).arg(&input).arg(&destination).arg(variant), &output_dir.join("quantize.log"))?;
    if !destination.is_file() { bail!("llama-quantize completed without creating the expected output."); }
    Ok(destination)
}

fn run_logged(command: &mut Command, log_path: &std::path::Path) -> Result<()> {
    use std::process::Stdio;
    let log = fs::File::create(log_path)?;
    let stderr = log.try_clone()?;
    let status = command.stdout(Stdio::from(log)).stderr(Stdio::from(stderr)).status()
        .with_context(|| format!("Could not launch command {command:?}"))?;
    if !status.success() {
        let text = fs::read_to_string(log_path).unwrap_or_default();
        bail!("Command exited with {status}: {}", text.chars().rev().take(4000).collect::<String>().chars().rev().collect::<String>());
    }
    Ok(())
}

fn artifact_size(path: &std::path::Path) -> Result<u64> {
    if path.is_file() { return Ok(fs::metadata(path)?.len()); }
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        total += artifact_size(&entry.path())?;
    }
    Ok(total)
}

fn perplexity_gguf(model: &std::path::Path, corpus: &std::path::Path) -> Result<f64> {
    let executable = find_executable("llama-perplexity").context("llama-perplexity was not found on PATH")?;
    let output = Command::new(executable).args(["-m", &model.to_string_lossy(), "-f", &corpus.to_string_lossy(), "-c", "2048"]).output()?;
    if !output.status.success() {
        bail!("llama-perplexity failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let text = format!("{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    text.lines().filter(|line| {
        let lower = line.to_ascii_lowercase(); lower.contains("perplexity") || lower.contains("ppl")
    }).flat_map(|line| line.split(|char: char| !(char.is_ascii_digit() || char == '.' || char == '-')))
        .filter_map(|value| value.parse::<f64>().ok()).last()
        .context("Could not find perplexity value in llama-perplexity output")
}

fn llama_bench_one(model: &std::path::Path, repeats: usize, prompt: usize, generation: usize, dtype: &str, batch: usize) -> Result<Value> {
    let executable = find_executable("llama-bench").context("llama-bench was not found on PATH")?;
    let cache = if dtype == "auto" { "f16" } else { dtype };
    let output = Command::new(executable).args(["-m", &model.to_string_lossy(), "-o", "json", "-r", &repeats.to_string(), "-p", &prompt.to_string(), "-n", &generation.to_string(), "-b", &batch.to_string(), "-ctk", cache, "-ctv", cache]).output()?;
    if !output.status.success() { bail!("llama-bench failed: {}", String::from_utf8_lossy(&output.stderr)); }
    let raw: Value = serde_json::from_slice(&output.stdout)?;
    let records = raw.as_array().or_else(|| raw.get("results").and_then(Value::as_array)).context("llama-bench output missing results")?;
    let row = records.iter().find(|row| row["n_gen"].as_u64().unwrap_or_default() > 0).context("No generation metrics in llama-bench output")?;
    let mut metrics = json!({"tok_s": row["avg_ts"], "tok_s_std": row["stddev_ts"]});
    if let Some(prefill) = records.iter().find(|row| row["n_prompt"].as_u64().unwrap_or_default() > 0 && row["n_gen"].as_u64().unwrap_or_default() == 0).and_then(|row| row["avg_ts"].as_f64()) {
        if prefill > 0.0 {
            metrics["prefill_tok_s"] = json!(prefill);
            metrics["prefill_ms_estimate"] = json!(1000.0 * prompt as f64 / prefill);
        }
    }
    Ok(metrics)
}

fn call_python_adapter(operation: &str, arguments: &[std::ffi::OsString]) -> Result<Value> {
    const ADAPTER: &str = r#"
import json, sys
op, *a = sys.argv[1:]
if op == 'quantize':
    from quantlab.runner import quantize_model
    value = {'artifact_path': str(quantize_model(a[0], a[1], a[2], a[3], a[4] or None))}
elif op == 'perplexity':
    from quantlab.evaluate import evaluate_perplexity
    value = evaluate_perplexity(a[0], a[1], int(a[2]), int(a[3]) if a[3] else None)
elif op == 'prompt_suite':
    from quantlab.evaluate import evaluate_prompt_suite
    value = evaluate_prompt_suite(a[0], a[1], int(a[2]), a[3] or None)
elif op == 'tasks':
    from quantlab.evaluate import evaluate_tasks
    value = evaluate_tasks(a[0], [x for x in a[1].split(',') if x], int(a[2]) if a[2] else None, int(a[3]))
elif op == 'vllm_benchmark':
    from quantlab.benchmark import benchmark_vllm
    value = benchmark_vllm(a[0], int(a[1]), int(a[2]), int(a[3]), a[4], int(a[5]))
elif op == 'sensitivity':
    from quantlab.sensitivity import analyze_sensitivity, write_sensitivity_report
    value = analyze_sensitivity(a[0], tuple(int(x) for x in a[1].split(',') if x), int(a[2]), a[3])
    write_sensitivity_report(value, a[4])
else:
    raise ValueError('Unknown QuantLab Python ML adapter: ' + op)
print('QL_ADAPTER_RESULT=' + json.dumps(value, default=str))
"#;
    let python = env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python).arg("-c").arg(ADAPTER).arg(operation).args(arguments).output()
        .context("Could not start the optional Python ML adapter; set PYTHON or install the QuantLab compatibility package")?;
    if !output.status.success() {
        bail!("Python ML adapter {operation:?} failed: {}{}", String::from_utf8_lossy(&output.stderr), String::from_utf8_lossy(&output.stdout));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result = stdout.lines().rev().find_map(|line| line.strip_prefix("QL_ADAPTER_RESULT="))
        .context("Python ML adapter did not return a result")?;
    serde_json::from_str(result).context("Python ML adapter returned invalid JSON")
}

fn parameter_count(model: Option<&str>, explicit: Option<f64>) -> Result<f64> {
    if let Some(value) = explicit {
        if !value.is_finite() || value <= 0.0 { bail!("Parameter count must be greater than zero."); }
        return Ok(value);
    }
    let model = model.context("Provide --model or --parameters-b")?;
    let bytes = model.as_bytes();
    let mut found = None;
    for index in 0..bytes.len() {
        if !(bytes[index].is_ascii_digit() || bytes[index] == b'.') { continue; }
        if index > 0 && (bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'.') { continue; }
        let mut end = index;
        while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') { end += 1; }
        let mut suffix = end;
        while suffix < bytes.len() && bytes[suffix].is_ascii_whitespace() { suffix += 1; }
        if suffix < bytes.len() && matches!(bytes[suffix], b'b' | b'B')
            && (suffix + 1 == bytes.len() || !bytes[suffix + 1].is_ascii_alphanumeric()) {
            found = model[index..end].parse::<f64>().ok();
        }
    }
    let value = found.context("Could not infer parameter count from model name; pass --parameters-b instead")?;
    if !value.is_finite() || value <= 0.0 { bail!("Parameter count must be greater than zero."); }
    Ok(value)
}

fn find_format(name: &str) -> Result<QuantFormat> {
    FORMATS.iter().copied().find(|format| format.name.eq_ignore_ascii_case(name))
        .with_context(|| format!("Unknown format {name:?}. Choose from: {}", FORMATS.iter().map(|f| f.name).collect::<Vec<_>>().join(", ")))
}

fn memory_budget(device: Option<&str>, explicit: Option<f64>) -> Result<f64> {
    let memory_gb = if let Some(value) = explicit { value } else {
        let name = device.context("Provide --device or --device-memory-gb")?.to_ascii_lowercase();
        match name.as_str() {
            "m3pro18" | "m3-pro-18gb" => 18.0,
            "rtx4090" | "a10g" => 24.0,
            "h100" => 80.0,
            _ => bail!("Unknown device {name:?}; choose m3pro18, rtx4090, a10g, or h100"),
        }
    };
    if !memory_gb.is_finite() || memory_gb <= 0.0 { bail!("Device memory must be greater than zero."); }
    Ok(memory_gb * 1_000_000_000.0 / 1024_f64.powi(3))
}

fn estimate_memory(parameters_b: f64, format: QuantFormat, reserve: f64) -> Result<Estimate> {
    if !parameters_b.is_finite() || parameters_b <= 0.0 { bail!("Parameter count must be greater than zero."); }
    if !reserve.is_finite() || reserve < 0.0 { bail!("Runtime reserve cannot be negative."); }
    let weights_gib = parameters_b * 1_000_000_000.0 * format.bits_per_parameter / 8.0 / 1024_f64.powi(3);
    Ok(Estimate { weights_gib, runtime_reserve_gib: reserve, total_gib: weights_gib + reserve })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_parameter_counts() {
        assert_eq!(parameter_count(Some("Qwen2.5-14B-Instruct"), None).unwrap(), 14.0);
        assert_eq!(parameter_count(Some("org/model-0.5b"), None).unwrap(), 0.5);
        assert_eq!(parameter_count(Some("model 7 B"), None).unwrap(), 7.0);
    }

    #[test]
    fn calculates_memory_for_quantized_model() {
        let format = find_format("q4_k_m").unwrap();
        let estimate = estimate_memory(1.0, format, 0.0).unwrap();
        assert!((estimate.weights_gib - 4.8e9 / 8.0 / 1024_f64.powi(3)).abs() < 1e-10);
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(parameter_count(None, Some(0.0)).is_err());
        assert!(memory_budget(None, Some(f64::NAN)).is_err());
        assert!(find_format("unknown").is_err());
    }

    #[test]
    fn parses_benchmark_sweep_dimensions() {
        assert_eq!(parse_usize_list(Some("1, 4,8"), 2).unwrap(), vec![1, 4, 8]);
        assert!(parse_usize_list(Some("1,0"), 2).is_err());
        assert!(parse_string_list(Some("f16,"), "auto").is_err());
    }

    #[test]
    fn parses_perplexity_from_tool_output() {
        assert_eq!(parse_perplexity("perplexity = 2.75\n"), Some(2.75));
        assert_eq!(parse_perplexity("no metric found"), None);
    }

    #[test]
    fn decodes_half_precision_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
    }

    #[test]
    fn ranks_safetensors_weights_without_loading_full_model() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let weights = directory.path().join("model.safetensors");
        let mut header = br#"{"dense.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#.to_vec();
        while header.len() % 8 != 0 { header.push(b' '); }
        let mut file = fs::File::create(&weights).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        for value in [1.0f32, 0.5, -1.0, -0.5] { file.write_all(&value.to_le_bytes()).unwrap(); }
        let output = directory.path().join("sensitivity.json");
        sensitivity_safetensors(directory.path(), "2,4", 2, &output).unwrap();
        let result: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(result["layers"].as_array().unwrap().len(), 1);
        assert_eq!(result["layers"][0]["name"], "dense");
    }

    #[test]
    fn loads_flat_yaml_run_configs_and_method_lists() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("run.yaml");
        fs::write(&config_path, "model: org/model\nmethods:\n  - gguf:Q4_K_M\n  - gguf:Q5_K_M\nrepeats: 2\n").unwrap();
        let config = load_run_config(&config_path).unwrap();
        assert_eq!(config_string(Some(&config), "methods").as_deref(), Some("gguf:Q4_K_M,gguf:Q5_K_M"));
        assert_eq!(config_usize(Some(&config), "repeats"), Some(2));
    }

    #[test]
    fn report_reads_existing_run_database_and_escapes_html() {
        let directory = tempfile::tempdir().unwrap();
        let connection = Connection::open(directory.path().join("results.sqlite")).unwrap();
        connection.execute_batch("CREATE TABLE results (run_id TEXT, method TEXT, variant TEXT, status TEXT, perplexity REAL, custom_score REAL, size_gib REAL, batch_size INTEGER, kv_cache_dtype TEXT, tok_s REAL, tok_s_std REAL, prefill_ms_estimate REAL, ttft_ms REAL, error TEXT);").unwrap();
        connection.execute("INSERT INTO results VALUES (?1, 'gguf', '<script>', 'completed', 4.0, NULL, 3.0, 1, 'f16', 20.0, 1.0, NULL, NULL, NULL)", ["run-1"]).unwrap();
        report_command("run-1", directory.path(), None).unwrap();
        let report = fs::read_to_string(directory.path().join("run-1.html")).unwrap();
        assert!(report.contains("run-1"));
        assert!(report.contains("&lt;script&gt;"));
    }

    #[test]
    fn stores_all_benchmark_metrics_for_each_method() {
        let connection = Connection::open_in_memory().unwrap();
        create_run_schema(&connection).unwrap();
        connection.execute("INSERT INTO runs VALUES ('run-1', 'model', 'device', '{}', 'now')", []).unwrap();
        let metrics = json!({
            "tok_s": 22.0,
            "tok_s_std": 1.5,
            "prefill_tok_s": 120.0,
            "prefill_ms_estimate": 8.0,
            "ttft_ms": 45.0,
            "ttft_ms_std": 3.0,
            "peak_mem_gib": 9.5,
        });
        store_run_result(&connection, "run-1", "bnb", "nf4", "completed", None, None, None,
            None, None, Some("fp8"), Some(2), Some(&metrics)).unwrap();
        let saved: (String, f64, f64, f64, f64, f64, f64, f64) = connection.query_row(
            "SELECT method, tok_s, tok_s_std, prefill_tok_s, prefill_ms_estimate, ttft_ms, ttft_ms_std, peak_mem_gib FROM results WHERE run_id='run-1'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?)),
        ).unwrap();
        assert_eq!(saved, ("bnb".to_owned(), 22.0, 1.5, 120.0, 8.0, 45.0, 3.0, 9.5));
    }
}
