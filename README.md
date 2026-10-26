# Named-entity tagging in Hinglish

An NYU Abu Dhabi course project on identifying named entities in Hindi-English code-mixed text with mBERT and XLM-RoBERTa. The pipeline prepares word-level labels, aligns them with each model's tokenizer, fine-tunes the encoder, and exports predictions and token-level error analysis.

Team: Akshith Karthik, Ashmit Mukherjee, Harsh Agarwal and Lovnish Julka.

The repository contains the pipeline and small offline test fixtures. Trained checkpoints and benchmark results are not included.

The task uses **flat word-level entity labels**. Entity-span boundaries are not preserved; evaluation measures token classification.

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

`bootstrap_from_hf --revision COMMIT` pins the upstream dataset revision; the export records source fingerprints. Confirm exported columns/counts and preserve the manifests when running an experiment.

To train mBERT on the same prepared words, use `--model_name bert-base-multilingual-cased --output_dir outputs/mbert-flat`; token IDs are regenerated for that model. Training also accepts `--revision` to pin the model source and records its resolved commit when the model configuration exposes one. `make train-xlmr`, `make train-mbert`, `make predict-test`, `make eval-test` and `make errors` wrap the same module commands. Override `OUTPUT`/`PRED`/`EVAL` when using a different checkpoint. Cluster/sync scripts are optional examples that need your own environment settings.

## Data and labels

The adapter targets [COMI-LINGUA on Hugging Face](https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA), configuration `NER`. Dataset access and reuse terms are governed by that source. Data are not redistributed here; `tests/fixtures` contains three hand-written examples.

Supported labels are `O`, `PER`, `ORG`, `LOC`, `MISC`, `DATE`, `TIME`, `HASHTAG`, `MENTION`. Incoming B-/I- prefixes collapse to the entity type. Aliases such as PERSON/ORGANISATION/LOCATION are canonicalized; unsupported labels raise an error.

Prepared files contain IDs, original words and flat named labels. NFC normalization and surrounding-whitespace trimming preserve script, case and punctuation. Roman-only filtering is optional; no transliteration is applied. Exact repeated text cannot cross prepared splits: test is retained preferentially, then validation, then training. Removed-row counts and source/prepared SHA-256 values are recorded in `manifest.json`. Conflicting annotations for the same text require explicit resolution.

Each encoder uses **its own fast tokenizer** at training and inference. The first subtoken receives the label ID; continuation and special tokens receive `-100`. Batches use right padding; left-padding tokenizers are rejected. Prediction reads the label mapping from the checkpoint. Older pretokenized data and checkpoints require regeneration with this format.

`--max_length` defaults to 256 subword tokens. Longer inputs are truncated. Predictions record original and evaluated word counts, and evaluation reports dropped words; check these counts before claiming complete-token coverage.

## Evaluation

`eval_flat` writes `metrics.json` with per-class precision/recall/F1/support, token accuracy, weighted F1 and two macro scores. `macro_f1_entities` averages the eight declared non-O classes; `macro_f1_all` averages all nine classes. Absent classes contribute zero. Weighted F1 and accuracy include O, so interpret them alongside entity-class scores and support. These are **token metrics**, not exact-match entity-span scores. `error_slices` lists token confusions, without inferring language-switch boundaries.

## Offline checks and limitations

The offline tests use a tiny fake tokenizer/model callback, exercise real preparation and evaluation CLIs, and require no dataset/model download, PyTorch, GPU or paid API:

```bash
pip install -r requirements-dev.txt
pytest -q
python -m src.data_prep --input tests/fixtures --output /tmp/ner-prepared
python -m tests.smoke_processed /tmp/ner-prepared
```

CI checks alignment/masking, MISC preservation, variable-length batch padding, tokenizer separation, truncation coverage, duplicate separation, annotation parsing and CLI flags.

The live dataset adapter and full transformer training/inference have not been exercised by these offline checks. They verify data handling and command interfaces, not model accuracy. No trained checkpoints, full benchmark predictions, LLM-baseline implementation or measured model-comparison table are bundled.

Source code is licensed under [MIT](LICENSE).
