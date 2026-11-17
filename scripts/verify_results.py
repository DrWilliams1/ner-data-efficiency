"""Recalculate saved NER metrics without a model, GPU, or dataset download."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ner_utils import choose_fraction, error_rows, metrics_from_arrays, nested_indices


def assert_same(actual, expected, location="value"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"Different fields at {location}"
        for key in expected:
            assert_same(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, float):
        assert np.isclose(actual, expected, rtol=0, atol=1e-10), f"Mismatch at {location}"
    else:
        assert actual == expected, f"Mismatch at {location}"


def verify_run(run_dir, require_test=True):
    root = Path(run_dir)
    read = lambda name: json.loads((root / name).read_text())
    config, data, records = read("experiment_config.json"), read("data_record.json"), read("validation_records.json")
    expected_keys = {(int(100 * f), seed) for f in config["fractions"] for seed in config["seeds"]}
    assert len(records) == len(expected_keys), "Missing or duplicate training run"
    assert {(r["fraction_percent"], r["seed"]) for r in records} == expected_keys
    first_gold, first_offsets = None, None
    for record in records:
        key = f'p{record["fraction_percent"]}_s{record["seed"]}'
        with np.load(root / f"validation_{key}.npz", allow_pickle=False) as arrays:
            assert_same(metrics_from_arrays(arrays), record["metrics"], key)
            assert arrays["label_names"].tolist() == data["label_names"]
            assert np.array_equal(arrays["row_ids"], np.arange(data["rows"]["validation"]))
            if first_gold is None:
                first_gold, first_offsets = arrays["gold_ids"].copy(), arrays["offsets"].copy()
            else:
                assert np.array_equal(first_gold, arrays["gold_ids"]), "Validation gold labels changed"
                assert np.array_equal(first_offsets, arrays["offsets"]), "Validation word boundaries changed"
        assert_same(read(f"validation_{key}.json"), record, key)
        planned = nested_indices(data["rows"]["train"], config["fractions"], record["seed"])
        ids = read(f"training_indices_{key}.json")
        assert ids == planned[str(record["fraction_percent"])], "Training subset changed"
        assert len(ids) == record["n_training_sentences"]
        assert record["training_seconds"] > 0 and record["optimizer_steps"] > 0
    for seed in config["seeds"]:
        hashes = {r["initial_classifier_sha256"] for r in records if r["seed"] == seed}
        assert len(hashes) == 1, "Classifier initialization differs between fractions of a seed"
    selected_fraction, means = choose_fraction(records)
    selection = read("selection.json")
    assert selection["fraction_percent"] == selected_fraction
    assert selection["seed"] == config["test_seed"]
    assert config["test_seed"] == config["seeds"][0]
    assert_same(selection["mean_validation_f1"], {str(k): v for k, v in means.items()})
    selected = next(r for r in records if r["fraction_percent"] == selected_fraction and r["seed"] == selection["seed"])
    with (root / "validation_error_review.csv").open(newline="") as f:
        errors = list(csv.DictReader(f))
    assert len(errors) == selected["metrics"]["fp"] + selected["metrics"]["fn"]
    assert sum(r["error_kind"] == "missed_gold_entity" for r in errors) == selected["metrics"]["fn"]
    assert sum(r["error_kind"] == "extra_predicted_entity" for r in errors) == selected["metrics"]["fp"]
    with np.load(root / f'validation_p{selected_fraction}_s{selection["seed"]}.npz', allow_pickle=False) as arrays:
        expected_errors = error_rows(arrays)
    for actual, expected in zip(errors, expected_errors):
        for key, value in expected.items():
            if key == "my_observation":
                continue  # Qualitative notes may be filled in after the run.
            converted = float(actual[key]) if isinstance(value, float) else int(actual[key]) if isinstance(value, int) else actual[key]
            assert_same(converted, value, f"error_worksheet.{key}")
    # Recalculate the comparison tables, including the sample SD across seeds.
    with (root / "validation_comparison.csv").open(newline="") as f:
        comparisons = list(csv.DictReader(f))
    assert len(comparisons) == len(records)
    indexed = {(r["fraction_percent"], r["seed"]): r for r in records}
    seen = set()
    for row in comparisons:
        key = (int(row["fraction_percent"]), int(row["seed"]))
        assert key not in seen
        seen.add(key)
        record = indexed[key]
        for column, value in {"training_sentences": record["n_training_sentences"],
                              "entity_precision": record["metrics"]["precision"],
                              "entity_recall": record["metrics"]["recall"],
                              "entity_f1": record["metrics"]["f1"],
                              "word_accuracy": record["metrics"]["word_accuracy"],
                              "training_seconds": record["training_seconds"],
                              "optimizer_steps": record["optimizer_steps"]}.items():
            assert np.isclose(float(row[column]), value, rtol=0, atol=1e-10), f"Comparison mismatch: {column}"
    with (root / "validation_summary.csv").open(newline="") as f:
        summaries = list(csv.DictReader(f))
    assert len(summaries) == len(config["fractions"])
    assert {int(row["fraction_percent"]) for row in summaries} == {int(100 * f) for f in config["fractions"]}
    for row in summaries:
        fraction_records = [r for r in records if r["fraction_percent"] == int(row["fraction_percent"])]
        f1s = [r["metrics"]["f1"] for r in fraction_records]
        seconds = [r["training_seconds"] for r in fraction_records]
        expected = {"training_sentences": fraction_records[0]["n_training_sentences"],
                    "entity_f1_mean": np.mean(f1s), "entity_f1_sd": np.std(f1s, ddof=1),
                    "precision_mean": np.mean([r["metrics"]["precision"] for r in fraction_records]),
                    "recall_mean": np.mean([r["metrics"]["recall"] for r in fraction_records]),
                    "training_seconds_mean": np.mean(seconds), "training_seconds_sd": np.std(seconds, ddof=1)}
        for column, value in expected.items():
            assert np.isclose(float(row[column]), value, rtol=0, atol=1e-10), f"Summary mismatch: {column}"
    with (root / "paired_validation_gains.csv").open(newline="") as f:
        gains = list(csv.DictReader(f))
    assert len(gains) == len(config["seeds"])
    assert {int(row["seed"]) for row in gains} == set(config["seeds"])
    for row in gains:
        seed = int(row["seed"])
        for fraction in (10, 25):
            expected = indexed[(100, seed)]["metrics"]["f1"] - indexed[(fraction, seed)]["metrics"]["f1"]
            assert np.isclose(float(row[f"100_minus_{fraction}_f1"]), expected, rtol=0, atol=1e-10)
    if require_test:
        final = read("final_test_metrics.json")
        assert final["fraction_percent"] == selected_fraction and final["seed"] == selection["seed"]
        with np.load(root / "final_test_predictions.npz", allow_pickle=False) as arrays:
            assert_same(metrics_from_arrays(arrays), final["metrics"], "final_test")
            assert arrays["label_names"].tolist() == data["label_names"]
            assert np.array_equal(arrays["row_ids"], np.arange(data["rows"]["test"]))
    manifest = read("MANIFEST_SHA256.json")
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST_SHA256.json"}
    assert set(manifest) == actual_files, "Manifest omits a file or lists a nonexistent file"
    for relative, item in manifest.items():
        target = (root / relative).resolve()
        assert target.is_relative_to(root.resolve()), "Manifest path leaves result directory"
        content = target.read_bytes()
        assert len(content) == item["size_bytes"], f"Size mismatch: {relative}"
        assert hashlib.sha256(content).hexdigest() == item["sha256"], f"Hash mismatch: {relative}"
    return {"status": "passed", "validation_runs": len(records), "test_predictions_verified": require_test,
            "checks": ["saved probabilities and labels", "strict entity metrics", "fixed validation data",
                       "nested training subsets", "paired classifier initialization", "selection rule",
                       "error worksheet rows", "comparison tables and sample SD", "artifact hashes"],
            "limits": "Checks internal consistency; does not rerun training or prove the source labels are correct."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_directory), indent=2))
