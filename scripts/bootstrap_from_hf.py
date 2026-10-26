"""Export COMI-LINGUA NER records, retaining original script and word labels."""
import argparse
import ast
import json
from pathlib import Path
from src.contract import write_jsonl


def decode_value(value):
    if isinstance(value, str):
        for decoder in [json.loads, ast.literal_eval]:
            try:
                decoded = decoder(value)
                if isinstance(decoded, list):
                    return decoded
            except (ValueError, SyntaxError, TypeError):
                pass
    return value


def extract_record(example, features, identifier):
    token_column = next((name for name in ["tokens", "Tokens", "Sentences", "sentence", "Sentence"] if name in example), None)
    label_column = next((name for name in ["ner_tags", "labels", "BIO", "tags", "Annotated by: Annotator 1"] if name in example), None)
    if token_column is None or label_column is None:
        raise ValueError(f"Missing token/annotation columns; available: {list(example)}")
    tokens, labels = decode_value(example[token_column]), decode_value(example[label_column])
    if isinstance(labels, list) and labels and all(isinstance(row, dict) for row in labels):
        # Annotation words supply the gold segmentation in COMI-LINGUA's dict-list form.
        tokens = [row.get("word", row.get("token")) for row in labels]
        labels = [row.get("entity", row.get("label", row.get("tag"))) for row in labels]
    else:
        if isinstance(tokens, str):
            tokens = tokens.split()
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
            tokens = [row.get("word", row.get("token")) for row in tokens]
        if isinstance(labels, str):
            labels = labels.split()
        if isinstance(labels, list) and labels and isinstance(labels[0], int):
            feature = features[label_column]
            names = getattr(getattr(feature, "feature", feature), "names", None)
            if names is None:
                raise ValueError("Numeric labels need a named ClassLabel feature")
            if not all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(names) for index in labels):
                raise ValueError("Numeric annotation IDs must be in the ClassLabel range; missing/negative IDs are not entity labels")
            labels = [names[index] for index in labels]
    if not isinstance(tokens, list) or not isinstance(labels, list) or not tokens or len(tokens) != len(labels):
        raise ValueError(f"Unaligned token/label data for {identifier}")
    if not all(isinstance(word, str) and word.strip() for word in tokens) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"Invalid token/label values for {identifier}")
    return {"id": identifier, "tokens": tokens, "ner_tags": labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="LingoIITGN/COMI-LINGUA")
    parser.add_argument("--config", default="NER")
    parser.add_argument("--revision", help="Pin a dataset revision for a repeatable download")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    from datasets import load_dataset
    dataset = load_dataset(args.dataset, args.config, revision=args.revision)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"dataset": args.dataset, "config": args.config, "requested_revision": args.revision, "splits": {}}
    for source_split, rows in dataset.items():
        split = {"dev": "validation", "valid": "validation"}.get(source_split, source_split)
        if split not in {"train", "validation", "test"}:
            continue
        records = [extract_record(row, rows.features, f"{source_split}:{index}") for index, row in enumerate(rows)]
        write_jsonl(args.output / f"{split}.jsonl", records)
        report["splits"][split] = {"examples": len(records), "fingerprint": rows._fingerprint}
    (args.output / "source_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
