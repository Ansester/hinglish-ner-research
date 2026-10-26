"""Evaluate aligned flat word-level predictions; no model download is needed."""
import argparse
import hashlib
import json
from pathlib import Path
from src.contract import evaluate_records, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("eval/xlmr-flat"))
    return parser.parse_args()


def main():
    args = parse_args()
    report = evaluate_records(read_jsonl(args.pred_path))
    report["prediction_sha256"] = hashlib.sha256(args.pred_path.read_bytes()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
