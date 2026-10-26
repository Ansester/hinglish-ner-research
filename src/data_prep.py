"""Prepare word-level flat labels without binding data to any model tokenizer."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from src.contract import LABELS, SCHEMA_VERSION, canonical_record, read_jsonl, write_jsonl


def is_sentence_roman_only(tokens):
    return all(not any("\u0900" <= char <= "\u097f" for char in word) for word in tokens)


def is_sentence_majority_roman(tokens):
    count = sum(not is_sentence_roman_only([word]) for word in tokens)
    return count < len(tokens) - count


def prepare(input_path, output_path, seed=42, roman_only=False, majority_roman=False):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output directories must differ")
    if input_path.is_dir():
        paths = {}
        for split, names in {"train": ["train.jsonl"], "validation": ["validation.jsonl", "dev.jsonl", "valid.jsonl"], "test": ["test.jsonl"]}.items():
            found = [input_path / name for name in names if (input_path / name).exists()]
            if len(found) > 1:
                raise ValueError(f"Ambiguous {split} files: {found}")
            if found:
                paths[split] = found[0]
        if "train" not in paths or "test" not in paths:
            raise ValueError("Input directory must contain train.jsonl and test.jsonl")
        splits = {split: [canonical_record(row, f"{split}:{i}") for i, row in enumerate(read_jsonl(path))] for split, path in paths.items()}
        if "validation" not in splits:
            rows = splits["train"][:]
            random.Random(seed).shuffle(rows)
            count = max(1, int(len(rows) * 0.1))
            splits["validation"], splits["train"] = rows[:count], rows[count:]
    else:
        paths = {"combined": input_path}
        rows = [canonical_record(row, f"combined:{i}") for i, row in enumerate(read_jsonl(input_path))]
        random.Random(seed).shuffle(rows)
        train_end, valid_end = int(len(rows) * 0.8), int(len(rows) * 0.9)
        splits = {"train": rows[:train_end], "validation": rows[train_end:valid_end], "test": rows[valid_end:]}
    # Preserve held-out examples preferentially; repeated text cannot cross splits.
    seen, removed, outputs = {}, {}, {}
    for split in ["test", "validation", "train"]:
        output, ids = [], set()
        for row in splits[split]:
            if roman_only and not is_sentence_roman_only(row["tokens"]):
                continue
            if majority_roman and not is_sentence_majority_roman(row["tokens"]):
                continue
            key = tuple(row["tokens"])
            if key in seen:
                if seen[key] != tuple(row["ner_tags"]):
                    raise ValueError("Identical token sequences have conflicting annotations; resolve these explicitly")
                continue
            if row["id"] in ids:
                raise ValueError(f"Duplicate id in {split}: {row['id']}")
            seen[key] = tuple(row["ner_tags"])
            ids.add(row["id"])
            output.append(row)
        if not output:
            raise ValueError(f"No examples remain in {split}")
        removed[split] = len(splits[split]) - len(output)
        outputs[split] = output
    output_path.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        write_jsonl(output_path / f"{split}.jsonl", rows)
    manifest = {"schema_version": SCHEMA_VERSION, "label_list": LABELS,
                "task": "flat word-level token classification; BIO prefixes collapse to entity type",
                "normalization": "NFC and surrounding whitespace only; original script/case/punctuation preserved",
                "seed": seed, "roman_only": roman_only, "majority_roman": majority_roman,
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                "prepared_sha256": {split: hashlib.sha256((output_path / f"{split}.jsonl").read_bytes()).hexdigest() for split in outputs},
                "splits": {split: {"examples": len(rows), "tokens": sum(len(row["tokens"]) for row in rows), "removed_duplicates_or_filtered": removed[split]} for split, rows in outputs.items()}}
    source_manifest = input_path / "source_manifest.json" if input_path.is_dir() else None
    if source_manifest is not None and source_manifest.exists():
        manifest["source_export"] = json.loads(source_manifest.read_text())
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument("--roman_only", action="store_true")
    filters.add_argument("--majority_roman", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(json.dumps(prepare(args.input, args.output, args.seed, args.roman_only, args.majority_roman), indent=2))


if __name__ == "__main__":
    main()
