"""Offline structural inspection for an explicitly prepared local dataset."""
import argparse
import json
from src.contract import load_prepared

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="data/processed")
    args = parser.parse_args()
    splits, manifest = load_prepared(args.path)
    print(json.dumps({"splits": {key: len(rows) for key, rows in splits.items()}, "manifest": manifest}, indent=2))

if __name__ == "__main__":
    main()
