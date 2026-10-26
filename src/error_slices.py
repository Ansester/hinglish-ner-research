"""List validated flat-token confusions and short examples."""
import argparse
from collections import Counter
from src.contract import evaluate_records, read_jsonl


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top_k must be positive")
    records = read_jsonl(args.pred_path)
    evaluate_records(records)

    pair_counts = Counter()
    # also collect some example sentences for each confusion
    examples = {}

    for rec in records:
        tokens = rec["tokens"]
        gold = rec["gold"]
        pred = rec["pred"]

        for t, g, p in zip(tokens, gold, pred):
            if g != p:
                key = (g, p)
                pair_counts[key] += 1
                if key not in examples:
                    examples[key] = []
                if len(examples[key]) < 3:
                    examples[key].append((t, " ".join(tokens)))

    print(f"Top {args.top_k} gold→pred confusions:")
    for (g, p), cnt in pair_counts.most_common(args.top_k):
        print(f"{g:8s} → {p:8s} : {cnt}")
        for tok, sent in examples[(g, p)]:
            print(f"   [tok={tok}] {sent}")
        print()


if __name__ == "__main__":
    main()
