import json

import pytest

from quantlab.config import load_run_config
from quantlab.devices import get_device
from quantlab.estimator import estimate_memory, parse_parameter_count
from quantlab.formats import get_format
from quantlab.orchestrate import parse_methods, run_matrix
from quantlab.report import pareto_frontier, write_html_report
from quantlab.storage import RunStore, validate_run_id


def test_memory_estimate_and_format_lookup():
    estimate = estimate_memory(1, get_format("q4_k_m"), runtime_reserve_gib=0)
    assert estimate.weights_gib == pytest.approx(4.8 * 1_000_000_000 / 8 / 1024**3)
    assert estimate.total_gib == estimate.weights_gib


@pytest.mark.parametrize("model, expected", [("Qwen2.5-14B-Instruct", 14), ("org/model-0.5b", 0.5), ("model 7 B v2", 7)])
def test_parameter_count_parsing(model, expected):
    assert parse_parameter_count(model) == expected


def test_invalid_parameter_count_and_format_fail_clearly():
    with pytest.raises(ValueError, match="Could not infer"):
        parse_parameter_count("model-without-size")
    with pytest.raises(ValueError, match="Unknown format"):
        get_format("not-a-format")


def test_config_validation_and_list_normalization(tmp_path):
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps({"model": "org/model", "methods": ["gguf:Q4_K_M", "bnb:4"], "tasks": ["mmlu"]}))
    config = load_run_config(config_path)
    assert config["methods"] == "gguf:Q4_K_M,bnb:4"
    assert config["tasks"] == "mmlu"

    config_path.write_text('{"surprise": true}')
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        load_run_config(config_path)


def test_method_matrix_validation():
    assert parse_methods("GGUF:q4_k_m,bnb:NF4") == [("gguf", "Q4_K_M"), ("bnb", "nf4")]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_methods("gguf:Q4_K_M,GGUF:q4_k_m")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_methods("unknown:4")


def test_run_store_round_trip_and_run_id_validation(tmp_path):
    store = RunStore(tmp_path / "results.sqlite")
    store.create_run("run-1", "org/model", "test-device", {"methods": [["gguf", "Q4_K_M"]]})
    run = store.get_run("run-1")
    assert run["config"] == {"methods": [["gguf", "Q4_K_M"]]}
    assert store.run_exists("run-1")
    with pytest.raises(ValueError, match="Run ID"):
        validate_run_id("../outside")


def test_resume_rejects_different_configuration_and_reuses_matching_run(tmp_path, monkeypatch):
    def fake_quantize(model, method, variant, output_dir, calibration_file=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / "model.gguf"
        artifact.write_bytes(b"model")
        return artifact

    monkeypatch.setattr("quantlab.orchestrate.quantize_model", fake_quantize)
    options = dict(
        model="org/model",
        methods=[("gguf", "Q4_K_M")],
        device="test-device",
        output_root=tmp_path,
        repeats=1,
    )
    run_id, rows = run_matrix(**options, run_id="resume-test")
    assert run_id == "resume-test"
    assert rows[0]["status"] == "completed"

    with pytest.raises(ValueError, match="different configuration"):
        run_matrix(**{**options, "model": "org/other-model"}, run_id=run_id)

    resumed_id, resumed_rows = run_matrix(**options, run_id=run_id)
    assert resumed_id == run_id
    assert resumed_rows[0]["status"] == "completed"


def test_pareto_frontier_and_html_escaping(tmp_path):
    rows = [
        {"method": "gguf", "variant": "Q4", "status": "completed", "perplexity": 5, "size_gib": 4, "tok_s": 10},
        {"method": "gguf", "variant": "Q5", "status": "completed", "perplexity": 6, "size_gib": 5, "tok_s": 8},
        {"method": "gguf", "variant": "Q6", "status": "failed", "perplexity": 2, "size_gib": 2, "tok_s": 20},
    ]
    assert [row["variant"] for row in pareto_frontier(rows)] == ["Q4"]
    report_path = write_html_report("run<&", rows, tmp_path / "report.html")
    content = report_path.read_text(encoding="utf-8")
    assert "run&lt;&amp;" in content
    assert "Q5" in content
