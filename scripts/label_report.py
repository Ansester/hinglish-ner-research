#!/usr/bin/env python3
from collections import Counter
from src.contract import load_prepared

def main(path="data/processed"):
    ds, _ = load_prepared(path)
    for split, d in ds.items():
        cnt = Counter()
        longest = sorted(d, key=lambda ex: len(ex["tokens"]), reverse=True)[:5]
        for ex in d:
            cnt.update(ex["ner_tags"])
        print(f"\n[{split}] examples={len(d)} unique_labels={len(cnt)}")
        print(" top labels:", cnt.most_common(12))
        print(" longest 5 sentences (length, first 20 tokens):")
        for ex in longest:
            print("  ", len(ex["tokens"]), ex["tokens"][:20])

if __name__ == "__main__":
    main()
