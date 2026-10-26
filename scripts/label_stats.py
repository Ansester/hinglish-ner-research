"""Count flat word labels in a prepared, hash-verified dataset."""
import argparse
from collections import Counter
from src.contract import load_prepared

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="data/processed")
    args = parser.parse_args()
    splits, _ = load_prepared(args.path)
    for split, rows in splits.items():
        counts = Counter(label for row in rows for label in row["ner_tags"])
        print(f"{split}: {len(rows)} examples | labels: {dict(counts)}")

if __name__ == "__main__":
    main()
