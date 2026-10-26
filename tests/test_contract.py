import json
import subprocess
import sys
from pathlib import Path
import pytest
from src.contract import LABELS, align_labels_with_tokenizer, canonical_record, evaluate_records, load_prepared, pad_model_inputs, prediction_record, write_jsonl
from src.data_prep import prepare, is_sentence_roman_only, is_sentence_majority_roman
from src.predict_flat_ner import predict_records
from scripts.bootstrap_from_hf import extract_record


class Encoding(dict):
    def __init__(self, ids, word_ids):
        super().__init__(input_ids=ids, attention_mask=[1] * len(ids))
        self.words = word_ids
    def word_ids(self):
        return self.words


class Tokenizer:
    def __init__(self, offset=0):
        self.offset = offset
    def __call__(self, tokens, **kwargs):
        ids, words = [0], [None]
        for index, token in enumerate(tokens):
            for piece in range(2 if token == "NYUAD" else 1):
                ids.append(self.offset + 10 + index * 2 + piece)
                words.append(index)
        cap = kwargs.get("max_length", 256) - 1
        return Encoding(ids[:cap] + [2], words[:cap] + [None])
    def pad(self, records, padding, return_tensors):
        assert padding is True
        width = max(len(row["input_ids"]) for row in records)
        return {key: [row[key] + [0] * (width - len(row[key])) for row in records] for key in records[0]}


def test_numeric_masks_first_subtoken_and_misc_survive():
    records = [{"id": "x", "tokens": ["NYUAD", "Delhi", "other"], "ner_tags": ["B-ORG", "I-LOC", "B-MISC"]}]
    row = align_labels_with_tokenizer(records, Tokenizer())[0]
    assert row["labels"] == [-100, 2, -100, 3, 4, -100]
    result = prediction_record(row, [0, 2, 8, 3, 4, 0], dict(enumerate(LABELS)))
    assert result["tokens"] == ["NYUAD", "Delhi", "other"]
    assert result["gold"] == result["pred"] == ["ORG", "LOC", "MISC"]


def test_ragged_batches_are_padded_and_each_model_tokenizes_its_own_words():
    records = [{"id": "a", "tokens": ["NYUAD", "Delhi"], "ner_tags": ["ORG", "LOC"]}, {"id": "b", "tokens": ["other"], "ner_tags": ["MISC"]}]
    tokenizer = Tokenizer()
    def infer_batch(encoded):
        padded = pad_model_inputs(encoded, tokenizer, return_tensors=None)
        assert len(padded["input_ids"][0]) == len(padded["input_ids"][1])
        return [[0, 2, 0, 3, 0], [0, 4, 0, 0, 0]]
    predictions = list(predict_records(records, tokenizer, dict(zip(LABELS, range(len(LABELS)))), infer_batch))
    assert [row["pred"] for row in predictions] == [["ORG", "LOC"], ["MISC"]]
    a = align_labels_with_tokenizer(records, Tokenizer(0))[0]
    b = align_labels_with_tokenizer(records, Tokenizer(100))[0]
    assert a["model_inputs"]["input_ids"] != b["model_inputs"]["input_ids"]
    tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right-padding"):
        align_labels_with_tokenizer(records, tokenizer)
    with pytest.raises(ValueError, match="right-padding"):
        pad_model_inputs([a, b], tokenizer)


def test_truncation_is_visible_and_token_metrics_include_no_entity_spans():
    row = align_labels_with_tokenizer([{"id": "a", "tokens": ["one", "two", "three"], "ner_tags": ["PER", "PER", "O"]}], Tokenizer(), max_length=3)[0]
    prediction = prediction_record(row, [0, 1, 0], dict(enumerate(LABELS)))
    report = evaluate_records([prediction])
    assert report["original_tokens"] == 3 and report["dropped_tokens"] == 2
    assert report["metrics"]["accuracy"] == 1
    assert report["metrics"]["per_class"]["PER"]["support"] == 1
    with pytest.raises(ValueError, match="aligned"):
        evaluate_records([{"id": "bad", "tokens": ["x"], "gold": ["O"], "pred": []}])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_records([prediction, prediction])


def test_prepare_preserves_scripts_and_removes_cross_split_duplicates(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    write_jsonl(raw / "train.jsonl", [{"tokens": ["shared"], "ner_tags": ["O"]}, {"tokens": ["कोड", "NYUAD"], "ner_tags": ["O", "B-ORG"]}])
    write_jsonl(raw / "validation.jsonl", [{"tokens": ["validation"], "ner_tags": ["O"]}])
    write_jsonl(raw / "test.jsonl", [{"tokens": ["shared"], "ner_tags": ["O"]}])
    manifest = prepare(raw, output)
    splits, loaded_manifest = load_prepared(output)
    assert manifest == loaded_manifest
    assert splits["train"][0]["tokens"] == ["कोड", "NYUAD"]
    assert splits["train"][0]["ner_tags"] == ["O", "ORG"]
    assert "input_ids" not in splits["train"][0]
    assert manifest["splits"]["train"]["removed_duplicates_or_filtered"] == 1
    assert is_sentence_roman_only(["hello"]) and not is_sentence_roman_only(["कोड"])
    assert is_sentence_majority_roman(["hello", "world", "कोड"])
    with pytest.raises(ValueError, match="Unsupported entity"):
        canonical_record({"tokens": ["x"], "ner_tags": ["NEW_CLASS"]}, "bad")


def test_bootstrap_parses_annotation_words_and_never_uses_predicted_tags():
    row = extract_record({"Sentences": "Delhi police", "Annotated by: Annotator 1": "[{'word': 'Delhi', 'entity': 'LOCATION'}, {'word': 'police', 'entity': 'X'}]"}, {}, "x")
    assert row["tokens"] == ["Delhi", "police"] and row["ner_tags"] == ["LOCATION", "X"]
    with pytest.raises(ValueError, match="Missing"):
        extract_record({"Sentences": "x", "Predicted tags": "O"}, {}, "x")
    class Feature:
        names = ["O", "PER"]
    assert extract_record({"tokens": ["name"], "ner_tags": [1]}, {"ner_tags": Feature()}, "ok")["ner_tags"] == ["PER"]
    with pytest.raises(ValueError, match="ClassLabel range"):
        extract_record({"tokens": ["name"], "ner_tags": [-1]}, {"ner_tags": Feature()}, "bad")


def test_documented_module_clis_accept_help_without_ml_runtime():
    root = Path(__file__).resolve().parents[1]
    for module in ["src.data_prep", "src.train_flat_ner", "src.predict_flat_ner", "src.eval_flat", "src.error_slices", "scripts.bootstrap_from_hf"]:
        result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "src.predict_flat_ner", "--model_dir", "/nonexistent/ner-checkpoint", "--text", "Delhi"], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0 and "local saved checkpoint" in result.stderr


def test_offline_preparation_prediction_evaluation_contract(tmp_path):
    record = {"id": "tiny", "tokens": ["Delhi", "NYUAD"], "ner_tags": ["LOC", "ORG"]}
    tokenizer = Tokenizer()
    def infer_batch(encoded):
        return [row["labels"] for row in encoded]
    predictions = list(predict_records([record], tokenizer, dict(zip(LABELS, range(len(LABELS)))), infer_batch))
    file = tmp_path / "predictions.jsonl"
    write_jsonl(file, predictions)
    result = subprocess.run([sys.executable, "-m", "src.eval_flat", "--pred_path", str(file), "--out_dir", str(tmp_path / "eval")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((tmp_path / "eval" / "metrics.json").read_text())
    assert metrics["metrics"]["accuracy"] == 1 and metrics["examples"] == 1


def test_conflicting_annotations_and_modified_prepared_files_fail(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    raw.mkdir()
    write_jsonl(raw / "train.jsonl", [{"tokens": ["unique"], "ner_tags": ["O"]}, {"tokens": ["same"], "ner_tags": ["ORG"]}])
    write_jsonl(raw / "validation.jsonl", [{"tokens": ["validation"], "ner_tags": ["O"]}])
    write_jsonl(raw / "test.jsonl", [{"tokens": ["same"], "ner_tags": ["PER"]}])
    with pytest.raises(ValueError, match="conflicting"):
        prepare(raw, output)
    write_jsonl(raw / "test.jsonl", [{"tokens": ["same"], "ner_tags": ["ORG"]}])
    prepare(raw, output)
    (output / "test.jsonl").unlink()
    with pytest.raises(ValueError, match="Missing prepared split"):
        load_prepared(output)
    prepare(raw, output)
    (output / "train.jsonl").write_text((output / "train.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="manifest"):
        load_prepared(output)
