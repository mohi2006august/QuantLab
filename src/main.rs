use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::env;
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
        if ["run", "evaluate", "benchmark", "report", "sensitivity", "publish"].contains(&first) {
            return forward_legacy(&raw_args);
        }
    }
    match Cli::parse().command {
        Some(Commands::Doctor) => doctor(),
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
}
