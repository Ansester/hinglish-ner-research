"""Word-level flat-label contract, alignment and metrics; no ML runtime imports."""
import json
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1
LABELS = ["O", "PER", "ORG", "LOC", "MISC", "DATE", "TIME", "HASHTAG", "MENTION"]
ALIASES = {"X": "O", "NONE": "O", "PERSON": "PER", "ORGANISATION": "ORG", "ORGANIZATION": "ORG", "LOCATION": "LOC", "GPE": "LOC"}


def canonical_label(label):
    if not isinstance(label, str):
        raise ValueError(f"Expected a named entity label string, got {label!r}")
    label = label.strip().upper()
    if label.startswith(("B-", "I-")):
        label = label[2:]
    label = ALIASES.get(label, label)
    if label not in LABELS:
        raise ValueError(f"Unsupported entity label {label!r}; update the declared label contract explicitly")
    return label


def canonical_record(record, fallback_id):
    tokens, labels = record.get("tokens"), record.get("ner_tags")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(x, str) and x.strip() for x in tokens):
        raise ValueError("tokens must be a nonempty list of nonempty strings")
    if not isinstance(labels, list) or len(labels) != len(tokens):
        raise ValueError("tokens and ner_tags must have the same length")
    return {"id": str(record.get("id", fallback_id)),
            "tokens": [unicodedata.normalize("NFC", word.strip()) for word in tokens],
            "ner_tags": [canonical_label(label) for label in labels]}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("label_list") != LABELS:
        raise ValueError("Unsupported prepared dataset contract; run python -m src.data_prep again")
    expected = {"train", "validation", "test"}
    if set(manifest.get("splits", {})) != expected or set(manifest.get("prepared_sha256", {})) != expected:
        raise ValueError("Prepared manifest must declare and hash train, validation and test")
    splits = {}
    for split in ["train", "validation", "test"]:
        file = directory / f"{split}.jsonl"
        if not file.is_file():
            raise ValueError(f"Missing prepared split file: {file}")
        if manifest["prepared_sha256"][split] != hashlib.sha256(file.read_bytes()).hexdigest():
            raise ValueError(f"Prepared {split} content no longer matches its manifest; prepare the dataset again")
        splits[split] = [canonical_record(row, f"{split}:{i}") for i, row in enumerate(read_jsonl(file))]
    return splits, manifest


def align_labels_with_tokenizer(examples, tokenizer, max_length=256, add_script_id=False, label2id=None):
    """First subtoken gets the numeric flat class; all other positions get -100.

    Truncation is explicit: metadata retains original and evaluated word counts.
    Prepared files never contain tokenizer IDs, so encoders cannot reuse another
    model's vocabulary accidentally. Script IDs are not model input features.
    """
    if add_script_id:
        raise ValueError("Script embeddings are not implemented; add_script_id is unsupported")
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("Only right-padding tokenizers are supported by word-position decoding")
    label2id = label2id or {label: index for index, label in enumerate(LABELS)}
    rows = []
    for index, record in enumerate(examples):
        tokens = record["tokens"]
        gold = record.get("ner_tags")
        if gold is not None and len(tokens) != len(gold):
            raise ValueError("tokens and ner_tags length mismatch")
        encoded = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length)
        word_ids = encoded.word_ids()
        labels, positions, previous = [], [], None
        for position, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous:
                positions.append((position, word_id))
                labels.append(label2id[canonical_label(gold[word_id])] if gold is not None else -100)
            else:
                labels.append(-100)
            previous = word_id
        inputs = {key: value for key, value in dict(encoded).items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
        rows.append({"id": str(record.get("id", index)), "model_inputs": inputs, "labels": labels,
                     "word_positions": positions, "tokens": tokens, "gold": gold,
                     "original_token_count": len(tokens), "evaluated_token_count": len(positions)})
    return rows


def pad_model_inputs(encoded_rows, tokenizer, return_tensors="pt"):
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("Only right-padding tokenizers are supported by word-position decoding")
    return tokenizer.pad([row["model_inputs"] for row in encoded_rows], padding=True, return_tensors=return_tensors)


def prediction_record(encoded, prediction_ids, id2label):
    positions = encoded["word_positions"]
    if positions and max(position for position, _ in positions) >= len(prediction_ids):
        raise ValueError("Model predictions are shorter than the encoded input")
    record = {"id": encoded["id"], "tokens": [encoded["tokens"][word] for _, word in positions],
              "pred": [id2label[int(prediction_ids[position])] for position, _ in positions],
              "original_token_count": encoded["original_token_count"],
              "evaluated_token_count": encoded["evaluated_token_count"]}
    if encoded["gold"] is not None:
        record["gold"] = [canonical_label(encoded["gold"][word]) for _, word in positions]
    return record


def token_metrics(gold, predicted, labels=LABELS):
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Gold and prediction sequences must have equal, nonzero lengths")
    unknown = (set(gold) | set(predicted)) - set(labels)
    if unknown:
        raise ValueError(f"Unknown flat labels: {sorted(unknown)}")
    gold_counts, pred_counts = Counter(gold), Counter(predicted)
    correct = Counter(g for g, p in zip(gold, predicted) if g == p)
    per_class = {}
    for label in labels:
        precision = correct[label] / pred_counts[label] if pred_counts[label] else 0.0
        recall = correct[label] / gold_counts[label] if gold_counts[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": gold_counts[label]}
    entities = [label for label in labels if label != "O"]
    return {"accuracy": sum(correct.values()) / len(gold),
            "macro_f1_entities": sum(per_class[label]["f1"] for label in entities) / len(entities),
            "macro_f1_all": sum(values["f1"] for values in per_class.values()) / len(labels),
            "weighted_f1": sum(values["f1"] * values["support"] for values in per_class.values()) / len(gold),
            "token_count": len(gold), "per_class": per_class}


def evaluate_records(records):
    gold, predicted = [], []
    original_count = 0
    ids = set()
    for record in records:
        identifier = record["id"]
        if identifier in ids:
            raise ValueError(f"Duplicate prediction id {identifier!r}")
        ids.add(identifier)
        if "gold" not in record or len(record["tokens"]) != len(record["gold"]) or len(record["gold"]) != len(record["pred"]):
            raise ValueError("Every evaluated record must contain aligned tokens, gold and pred")
        original = record.get("original_token_count", len(record["tokens"]))
        if not isinstance(original, int) or original < len(record["tokens"]):
            raise ValueError("Original token count must be an integer at least as large as evaluated coverage")
        if record.get("evaluated_token_count", len(record["tokens"])) != len(record["tokens"]):
            raise ValueError("Evaluated token count does not match prediction coverage")
        gold.extend(record["gold"])
        predicted.extend(record["pred"])
        original_count += original
    metrics = token_metrics(gold, predicted)
    return {"task": "flat word-level token classification, not entity-span NER",
            "examples": len(records), "original_tokens": original_count,
            "dropped_tokens": original_count - len(gold), "metrics": metrics}
