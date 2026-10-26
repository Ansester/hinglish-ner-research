"""Batch inference using the saved encoder tokenizer and label configuration."""
import argparse
import hashlib
import json
from pathlib import Path
from src.contract import LABELS, align_labels_with_tokenizer, load_prepared, pad_model_inputs, prediction_record, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--text", help="Optional single sentence; words are separated by whitespace")
    parser.add_argument("--out_path", type=Path, default=Path("preds/xlmr_flat_test.jsonl"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    return parser.parse_args()


def predict_records(rows, tokenizer, label2id, infer_batch, batch_size=16, max_length=256):
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    id2label = {int(index): label for label, index in label2id.items()}
    for start in range(0, len(rows), batch_size):
        encoded = align_labels_with_tokenizer(rows[start:start + batch_size], tokenizer, max_length, label2id=label2id)
        predicted = infer_batch(encoded)
        if len(predicted) != len(encoded):
            raise ValueError("Model returned a different batch size")
        for row, ids in zip(encoded, predicted):
            yield prediction_record(row, ids, id2label)


def main():
    args = parse_args()
    if not args.model_dir.is_dir() or not (args.model_dir / "config.json").is_file():
        raise ValueError("model_dir must be a local saved checkpoint directory containing config.json")
    if args.text is not None:
        if not args.text.split():
            raise ValueError("text must contain at least one word")
        rows = [{"id": "text:0", "tokens": args.text.split()}]
    else:
        splits, _ = load_prepared(args.data_dir)
        rows = splits[args.split]
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with word_ids is required")
    model = AutoModelForTokenClassification.from_pretrained(str(args.model_dir), local_files_only=True)
    label2id = model.config.label2id
    if set(label2id) != set(LABELS) or set(label2id.values()) != set(range(len(LABELS))):
        raise ValueError("Checkpoint label mapping does not match this release; retrain with the current prepared-data contract")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def infer_batch(encoded):
        batch = pad_model_inputs(encoded, tokenizer).to(device)
        with torch.no_grad():
            return model(**batch).logits.argmax(dim=-1).cpu().tolist()

    predictions = list(predict_records(rows, tokenizer, label2id, infer_batch, args.batch_size, args.max_length))
    write_jsonl(args.out_path, predictions)
    report = {"model_dir": str(args.model_dir), "label_list": LABELS,
              "max_length": args.max_length, "examples": len(predictions),
              "original_tokens": sum(row["original_token_count"] for row in predictions),
              "evaluated_tokens": sum(row["evaluated_token_count"] for row in predictions),
              "prediction_sha256": hashlib.sha256(args.out_path.read_bytes()).hexdigest()}
    for name, path in {"checkpoint_config_sha256": args.model_dir / "config.json",
                       "checkpoint_run_manifest_sha256": args.model_dir / "run_manifest.json"}.items():
        if path.exists():
            report[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.text is None:
        report["dataset_manifest_sha256"] = hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest()
    args.out_path.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(predictions[0] if args.text else report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
