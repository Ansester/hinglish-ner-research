# Hinglish Named Entity Recognition Benchmark

This repository contains code and evaluation tooling from a project on named-entity recognition in Hindi-English code-mixed text. The project fine-tunes multilingual transformer encoders and separately evaluates zero-shot general-purpose language-model baselines.

---

## Context & Provenance

* **Project Context**: Course project, New York University Abu Dhabi.
* **Team**: Akshith Karthik, Ashmit Mukherjee, Harsh Agarwal, Lovnish Julka.
* **Contributions**: Fine-tuning pipeline implementation (mBERT, XLM-RoBERTa), evaluation harness design, and error analysis.

---

## Methodology & Models

The repository evaluates named-entity recognition (PER, LOC, ORG, MISC) on Hindi-English code-mixed text from the **COMI-LINGUA** benchmark dataset.

### Evaluated Models

* **Fine-Tuned Encoder Baselines**: `xlm-roberta-base` and `bert-base-multilingual-cased` (mBERT) fine-tuned using token classification heads with subword label alignment.
* **Zero-Shot LLM Baselines**: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`, and `openai/gpt-4o` evaluated via prompt-based zero-shot extraction.

> **Evaluation Qualification**: Fine-tuned encoder models were evaluated across the complete COMI-LINGUA test split (**4,829 examples**). Zero-shot LLM baselines were evaluated on a representative **100-example sample** drawn from the test set due to API resource constraints.

---

## Key Findings & Qualitative Error Analysis

1. **Subword & Entity Boundary Errors**: Code-mixed tokens frequently cause subword tokenization splits where entity boundaries are misaligned across language boundaries.
2. **Category Ambiguity**: Frequent tag confusion between `ORG` and `MISC` labels, particularly in informal social media text containing brand names and mixed-language acronyms.
3. **Language-Switch Boundaries**: Entity spans immediately adjacent to code-switch points exhibit higher error rates across both encoder and LLM architectures.

---

## Dataset

* **Source**: COMI-LINGUA benchmark dataset.
* **Format**: BIO entity tagging for Hindi-English code-mixed text.
* **Access**: Dataset files are not redistributed in this repository. Use `scripts/bootstrap_from_hf.py` to prepare the dataset locally.

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run fine-tuning
python src/train_flat_ner.py --model xlm-roberta-base --epochs 5

# Run evaluation on test split
python src/eval_flat.py --model_path checkpoints/xlm-roberta-best --test_data data/processed/test.jsonl

# Run inference
python src/predict_flat_ner.py --model_path checkpoints/xlm-roberta-best --text "Delhi police arrested the suspect near Connaught Place"
```

---

## Repository Structure

```
hinglish-ner-research/
├── src/
│   ├── data_prep.py        # Preprocessing and label alignment
│   ├── train_flat_ner.py   # Encoder fine-tuning script
│   ├── eval_flat.py        # Evaluation pipeline
│   └── predict_flat_ner.py # Single-sentence inference script
├── scripts/
│   ├── label_stats.py      # Dataset statistics utility
│   └── bootstrap_from_hf.py# Dataset loading script
├── tests/                  # Verification tests
├── requirements.txt
├── LICENSE                 # Repository license
└── README.md
```

---

## License

This repository's source code is licensed under the [MIT License](LICENSE).
