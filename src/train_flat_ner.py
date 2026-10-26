"""Fine-tune an encoder on flat word-level entity labels, tokenizing per model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from src.contract import LABELS, align_labels_with_tokenizer, load_prepared, token_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--revision", help="Optional pinned upstream model revision")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/xlmr-flat"))
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.batch_size, args.num_train_epochs, args.max_length, args.gradient_accumulation_steps) <= 0:
        raise ValueError("Epochs, batch size, max length and accumulation must be positive")
    splits, manifest = load_prepared(args.data_dir)
    if "train" not in splits or "validation" not in splits:
        raise ValueError("Prepared train and validation splits are required")
    from datasets import Dataset
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with word_ids is required")
    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = dict(enumerate(LABELS))
    encoded, counts = {}, {}
    for split in ["train", "validation"]:
        rows = align_labels_with_tokenizer(splits[split], tokenizer, args.max_length, label2id=label2id)
        encoded[split] = Dataset.from_list([{**row["model_inputs"], "labels": row["labels"]} for row in rows])
        counts[split] = {"original_tokens": sum(row["original_token_count"] for row in rows), "evaluated_tokens": sum(row["evaluated_token_count"] for row in rows)}
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, revision=args.revision, num_labels=len(LABELS), label2id=label2id, id2label=id2label)

    def compute_metrics(prediction):
        logits, labels = prediction
        predictions = np.argmax(logits, axis=-1)
        gold, predicted = [], []
        for gs, ps in zip(labels, predictions):
            for g, p in zip(gs, ps):
                if int(g) != -100:
                    gold.append(id2label[int(g)])
                    predicted.append(id2label[int(p)])
        metrics = token_metrics(gold, predicted)
        return {key: metrics[key] for key in ["accuracy", "macro_f1_entities", "macro_f1_all", "weighted_f1"]}

    training_args = TrainingArguments(output_dir=str(args.output_dir), num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1_entities", greater_is_better=True,
        use_cpu=args.device == "cpu", fp16=args.fp16, report_to=[],
        gradient_accumulation_steps=args.gradient_accumulation_steps, save_total_limit=2)
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
                      tokenizer=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    run = {"task": "flat word-level token classification, not span NER", "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "dataset_manifest_sha256": hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest(),
           "data_manifest": manifest, "token_coverage": counts, "label_list": LABELS,
           "resolved_model_commit": getattr(model.config, "_commit_hash", None),
           "versions": {name: importlib.metadata.version(name) for name in ["transformers", "datasets", "torch"]},
           "training_log": trainer.state.log_history}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
