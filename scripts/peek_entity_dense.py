# scripts/peek_entity_dense.py
#!/usr/bin/env python3
from src.contract import load_prepared

def non_o_share(tags): 
    n=len(tags); return 0 if n==0 else sum(t!="O" for t in tags)/n

def main(path="data/processed", top=10):
    ds, _ = load_prepared(path)
    for split in ("train","validation","test"):
        if split not in ds: continue
        d = ds[split]
        # sort by non-O share
        idx = sorted(range(len(d)), key=lambda i: non_o_share(d[i]["ner_tags"]), reverse=True)[:top]
        print(f"\n[{split}] top {top} entity-dense rows")
        for i in idx:
            ex = d[i]
            share = non_o_share(ex["ner_tags"])
            print(f"- ex#{i} share={share:.2f} len={len(ex['tokens'])}")
            print("  text:", " ".join(ex["tokens"])[:220])
            print("  tags:", ex["ner_tags"][:30])
            print()
if __name__ == "__main__":
    main()
