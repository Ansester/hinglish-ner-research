# Hinglish flat-label entity tagging

Course-project code for fine-tuning multilingual encoders on Hindi-English code-mixed text, with word-level label alignment, batched prediction and token-level error analysis. Original team credit: Akshith Karthik, Ashmit Mukherjee, Harsh Agarwal and Lovnish Julka (NYU Abu Dhabi). This repository does not establish exclusive individual ownership of the team's work.

This release supplies a coherent pipeline and offline regression checks. **No trained checkpoint, full benchmark predictions, LLM-baseline implementation or measured research-results table is bundled.** Earlier README claims about complete-test evaluation, a representative 100-example LLM sample and code-switch-boundary findings are not supported by artifacts in this release and have been removed. No model benchmark was rerun for the release repair.

## Task and data contract

The adapter targets [COMI-LINGUA on Hugging Face](https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA), configuration `NER`. Dataset access and reuse terms are governed by that source. Data are not redistributed here; `tests/fixtures` contains only three hand-written contract examples.

The implemented task is **flat word-level token classification**, not entity-span/BIO NER. Supported labels are `O`, `PER`, `ORG`, `LOC`, `MISC`, `DATE`, `TIME`, `HASHTAG`, `MENTION`. Incoming B-/I- prefixes collapse to the entity type; MISC is retained. Known aliases such as PERSON/ORGANISATION/LOCATION are canonicalized, while unsupported labels fail explicitly rather than silently becoming O.

Prepared files contain IDs, original words and flat named labels. NFC normalization and surrounding-whitespace trimming preserve original script, case and punctuation. Roman-only filtering is available explicitly; transliteration is not silently applied. Exact repeated text cannot cross prepared splits: test is retained preferentially, then validation, then training. Counts removed from each split and source/prepared SHA-256 values are recorded in `manifest.json`. Conflicting annotations for the same text require explicit resolution.

Each encoder tokenizes the word-level records with **its own fast tokenizer** at training and inference. The first subtoken gets the numeric label ID; continuation/special tokens get integer `-100`. Right padding is applied per batch; left-padding tokenizers are explicitly rejected so word positions cannot silently shift. Checkpoint label mappings drive prediction. Old pretokenized Arrow data and old checkpoints require regeneration with this contract.

`--max_length` defaults to 256 subword tokens. Longer inputs are truncated; predictions record both original/evaluated word counts, and evaluation reports dropped words. Thus an evaluated split must not be described as complete-token coverage unless those counts establish it. Adjacent flat labels do not preserve entity boundaries.

## Run

Use Python 3.11. Training requirements pin a PyTorch 2.5.1 build; use the [official PyTorch installation guidance](https://pytorch.org/get-started/previous-versions/) for the appropriate CPU/CUDA wheel if needed.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Network access: export the benchmark data, then prepare model-independent words.
python -m scripts.bootstrap_from_hf
python -m src.data_prep --input data/raw --output data/processed

# Network/model compute: fine-tune and save model, tokenizer and run manifest.
python -m src.train_flat_ner --model_name xlm-roberta-base --num_train_epochs 5 --output_dir outputs/xlmr-flat

# Predict the prepared test words and evaluate their flat token labels.
python -m src.predict_flat_ner --model_dir outputs/xlmr-flat --split test --out_path preds/xlmr_flat_test.jsonl
python -m src.eval_flat --pred_path preds/xlmr_flat_test.jsonl --out_dir eval/xlmr-flat
python -m src.error_slices --pred_path preds/xlmr_flat_test.jsonl

# Single sentence: whitespace defines input words; no gold score is produced.
python -m src.predict_flat_ner --model_dir outputs/xlmr-flat --text "Delhi police arrested the suspect near Connaught Place" --out_path preds/example.jsonl
```

`bootstrap_from_hf --revision COMMIT` can pin the upstream dataset revision; the export records source fingerprints. The live dataset adapter and full transformer training/inference were not exercised during the offline release checks. Confirm exported columns/counts and preserve the manifests when running an actual experiment.

To train mBERT on the same prepared words, use `--model_name bert-base-multilingual-cased --output_dir outputs/mbert-flat`; token IDs are regenerated for that model. Training also accepts `--revision` to pin the model source and records its resolved commit when the model configuration exposes one. `make train-xlmr`, `make train-mbert`, `make predict-test`, `make eval-test` and `make errors` wrap the same module commands. Override `OUTPUT`/`PRED`/`EVAL` when using a different checkpoint. Cluster/sync scripts are optional examples and do not establish any deployed run.

## Metrics and offline checks

`eval_flat` writes `metrics.json` with per-class precision/recall/F1/support, token accuracy, weighted F1 and two macro scores. `macro_f1_entities` averages the eight declared non-O classes; `macro_f1_all` averages all nine classes. Absent classes contribute zero. Weighted F1 and accuracy include O, so interpret them alongside entity-class scores and support. These are **token metrics**, not exact-match entity-span scores. `error_slices` lists token confusions, without inferring language-switch boundaries.

The offline tests use a tiny fake tokenizer/model callback, exercise real preparation and evaluation CLIs, and require no dataset/model download, PyTorch, GPU or paid API:

```bash
pip install -r requirements-dev.txt
pytest -q
python -m src.data_prep --input tests/fixtures --output /tmp/ner-prepared
python -m tests.smoke_processed /tmp/ner-prepared
```

CI runs these contract checks. They verify alignment/masking, MISC preservation, variable-length batch padding, tokenizer separation, explicit truncation coverage, duplicate separation, annotation parsing and CLI flags. Passing them establishes the release's data/CLI contract, not model accuracy or reproduced research findings.

Source code is licensed under [MIT](LICENSE).
