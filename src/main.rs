use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rusqlite::{Connection, OptionalExtension};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};

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
    let raw_args: Vec<_> = env::args_os().skip(1).collect();
    if let Some(first) = raw_args.first().and_then(|value| value.to_str()) {
        if ["evaluate", "sensitivity", "publish"].contains(&first) {
            return forward_legacy(&raw_args);
        }
        if first == "run" && ["--config", "--prompt-suite", "--judge-model", "--tasks", "--calibration-file", "--seed"]
            .iter().any(|flag| raw_args.iter().any(|argument| argument == flag)) {
            return forward_legacy(&raw_args);
        }
    }
    match Cli::parse().command {
        Some(Commands::Doctor) => doctor(),
        Some(Commands::Run { config, model, methods, device, output_dir, eval_text, benchmark, kv_cache_dtype, batch_size, prompt_tokens, generation_tokens, repeats, resume }) => {
            if config.is_some() || methods.as_deref().map_or(true, |spec| spec.split(',').any(|item| !item.trim().to_ascii_lowercase().starts_with("gguf:")))
                || benchmark.as_deref() == Some("vllm") {
                let raw_args: Vec<_> = env::args_os().skip(1).collect();
                return forward_legacy(&raw_args);
            }
            let model = model.context("Provide --model for the native Rust GGUF runner")?;
            let methods = methods.context("Provide --methods for the native Rust GGUF runner")?;
            run_gguf_matrix(
                &model, &methods, device.as_deref().unwrap_or("unknown"),
                output_dir.as_deref().unwrap_or_else(|| std::path::Path::new("quantlab-runs")),
                eval_text.as_deref(), benchmark.as_deref(), &kv_cache_dtype,
                batch_size.unwrap_or(2048), prompt_tokens, generation_tokens, repeats, resume.as_deref(),
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
        None => Ok(()),
    }
}

fn doctor() -> Result<()> {
    println!("QuantLab {}", env!("CARGO_PKG_VERSION"));
    println!("Rust toolchain: {}", option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("stable"));
    for tool in ["llama-cli", "llama-quantize", "llama-perplexity", "llama-bench", "llama-server", "nvidia-smi"] {
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
        let raw_args: Vec<_> = env::args_os().skip(1).collect();
        return forward_legacy(&raw_args);
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

fn validate_run_id(run_id: &str) -> Result<()> {
    if run_id.is_empty() || run_id.len() > 64 || !run_id.as_bytes()[0].is_ascii_alphanumeric()
        || !run_id.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)) {
        bail!("Run ID must be 1-64 letters, numbers, dots, underscores, or hyphens and start with a letter or number.");
    }
    Ok(())
}

fn run_gguf_matrix(
    model: &str,
    methods_spec: &str,
    device: &str,
    output_root: &std::path::Path,
    eval_text: Option<&std::path::Path>,
    benchmark: Option<&str>,
    kv_cache_dtype: &str,
    batch_size: usize,
    prompt_tokens: usize,
    generation_tokens: usize,
    repeats: usize,
    requested_run_id: Option<&str>,
) -> Result<()> {
    if benchmark == Some("vllm") {
        bail!("vLLM runs currently require the Python compatibility backend.");
    }
    if batch_size == 0 || prompt_tokens == 0 || generation_tokens == 0 || repeats == 0 {
        bail!("Batch size, token counts, and repeats must be positive.");
    }
    let methods = methods_spec.split(',').map(|item| {
        let (method, variant) = item.trim().split_once(':').context("Methods must use method:variant syntax")?;
        if !method.eq_ignore_ascii_case("gguf") || variant.trim().is_empty() {
            bail!("The native Rust runner currently supports GGUF methods only.");
        }
        let variant = variant.trim().to_ascii_uppercase();
        if !["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"].contains(&variant.as_str()) {
            bail!("Unsupported GGUF quantization {variant:?}.");
        }
        Ok(variant)
    }).collect::<Result<Vec<_>>>()?;
    if methods.is_empty() || methods.iter().collect::<std::collections::HashSet<_>>().len() != methods.len() {
        bail!("Specify a non-empty method list without duplicate variants.");
    }
    if !PathBuf::from(model).is_dir() && !PathBuf::from(model).is_file() {
        bail!("Rust-native GGUF conversion currently requires a local model directory or GGUF file.");
    }
    fs::create_dir_all(output_root)?;
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
        "methods": methods.iter().map(|variant| json!(["gguf", variant])).collect::<Vec<_>>(),
        "eval_text": eval_text.map(|path| path.to_string_lossy()),
        "benchmark_backend": benchmark,
        "kv_cache_dtype": kv_cache_dtype,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "repeats": repeats,
        "seed": 1234,
        "judge_model": null,
        "prompt_suite": null,
        "tasks": null,
        "calibration_file": null,
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
    for variant in methods {
        let existing: Option<(String, Option<String>)> = database.query_row(
            "SELECT status, artifact_path FROM results WHERE run_id=?1 AND method='gguf' AND variant=?2",
            rusqlite::params![run_id, variant], |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?;
        if let Some((status, Some(artifact))) = &existing {
            if status == "completed" && PathBuf::from(artifact).exists() {
                println!("Skip gguf:{variant} (already complete)");
                continue;
            }
        }
        let artifact_dir = run_dir.join(format!("gguf-{variant}"));
        let _ = fs::create_dir_all(&artifact_dir);
        store_run_result(&database, &run_id, &variant, "running", None, None, None, None, None, None, None, None, None, None)?;
        let attempt = (|| -> Result<(PathBuf, Option<f64>, Option<Value>)> {
            let artifact = quantize_gguf_native(model, &variant, &artifact_dir)?;
            let perplexity = eval_text.map(|path| perplexity_gguf(&artifact, path)).transpose()?;
            let metrics = if benchmark == Some("llama.cpp") {
                Some(llama_bench_one(&artifact, repeats, prompt_tokens, generation_tokens, kv_cache_dtype, batch_size)?)
            } else { None };
            Ok((artifact, perplexity, metrics))
        })();
        match attempt {
            Ok((artifact, perplexity, metrics)) => {
                let tok_s = metrics.as_ref().and_then(|value| value["tok_s"].as_f64());
                let tok_s_std = metrics.as_ref().and_then(|value| value["tok_s_std"].as_f64());
                let prefill = metrics.as_ref().and_then(|value| value["prefill_ms_estimate"].as_f64());
                let cache_dtype = if benchmark == Some("llama.cpp") { Some(if kv_cache_dtype == "auto" { "f16" } else { kv_cache_dtype }) } else { None };
                store_run_result(&database, &run_id, &variant, "completed", Some(&artifact), Some(artifact_size(&artifact)? as f64 / 1024_f64.powi(3)), perplexity, None, None, cache_dtype, if metrics.is_some() { Some(batch_size as i64) } else { None }, tok_s, tok_s_std, prefill)?;
                println!("Completed gguf:{variant} → {}", artifact.display());
            }
            Err(error) => {
                failures += 1;
                store_run_result(&database, &run_id, &variant, "failed", None, None, None, None, None, None, None, None, None, None)?;
                database.execute("UPDATE results SET error=?1 WHERE run_id=?2 AND method='gguf' AND variant=?3", rusqlite::params![format!("{error:#}"), run_id, variant])?;
                eprintln!("Failed gguf:{variant}: {error:#}");
            }
        }
    }
    report_command(&run_id, output_root, None)?;
    println!("Run ID: {run_id}");
    println!("Results database: {}", output_root.join("results.sqlite").display());
    if failures > 0 { bail!("{failures} quantization matrix item(s) failed."); }
    Ok(())
}

fn create_run_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, model TEXT NOT NULL, device TEXT NOT NULL, config_json TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS results (run_id TEXT NOT NULL REFERENCES runs(run_id), method TEXT NOT NULL, variant TEXT NOT NULL, status TEXT NOT NULL, artifact_path TEXT, size_gib REAL, perplexity REAL, custom_score REAL, task_scores_json TEXT, kv_cache_dtype TEXT, batch_size INTEGER, tok_s REAL, tok_s_std REAL, prefill_tok_s REAL, prefill_ms_estimate REAL, ttft_ms REAL, ttft_ms_std REAL, peak_mem_gib REAL, error TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(run_id, method, variant));")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn store_run_result(
    connection: &Connection, run_id: &str, variant: &str, status: &str,
    artifact_path: Option<&std::path::Path>, size_gib: Option<f64>, perplexity: Option<f64>,
    custom_score: Option<f64>, task_scores_json: Option<&str>, kv_cache_dtype: Option<&str>,
    batch_size: Option<i64>, tok_s: Option<f64>, tok_s_std: Option<f64>, prefill_ms_estimate: Option<f64>,
) -> Result<()> {
    connection.execute("INSERT INTO results (run_id, method, variant, status, artifact_path, size_gib, perplexity, custom_score, task_scores_json, kv_cache_dtype, batch_size, tok_s, tok_s_std, prefill_ms_estimate, updated_at) VALUES (?1, 'gguf', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, datetime('now')) ON CONFLICT(run_id, method, variant) DO UPDATE SET status=excluded.status, artifact_path=excluded.artifact_path, size_gib=excluded.size_gib, perplexity=excluded.perplexity, custom_score=excluded.custom_score, task_scores_json=excluded.task_scores_json, kv_cache_dtype=excluded.kv_cache_dtype, batch_size=excluded.batch_size, tok_s=excluded.tok_s, tok_s_std=excluded.tok_s_std, prefill_ms_estimate=excluded.prefill_ms_estimate, error=NULL, updated_at=excluded.updated_at",
        rusqlite::params![run_id, variant, status, artifact_path.map(|path| path.to_string_lossy().into_owned()), size_gib, perplexity, custom_score, task_scores_json, kv_cache_dtype, batch_size, tok_s, tok_s_std, prefill_ms_estimate])?;
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
        if prefill > 0.0 { metrics["prefill_ms_estimate"] = json!(1000.0 * prompt as f64 / prefill); }
    }
    Ok(metrics)
}

fn forward_legacy(args: &[std::ffi::OsString]) -> Result<()> {
    let python = env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let status = Command::new(python).arg("-m").arg("quantlab").args(args)
        .status().context("Could not start the Python compatibility backend; install the QuantLab Python package or set PYTHON")?;
    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }
    Ok(())
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
}
