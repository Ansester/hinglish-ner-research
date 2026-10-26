PYTHON ?= python
DATA_DIR ?= data/processed
MODEL ?= xlm-roberta-base
OUTPUT ?= outputs/xlmr-flat
PRED ?= preds/xlmr_flat_test.jsonl
EVAL ?= eval/xlmr-flat

.PHONY: help install bootstrap prep train train-xlmr train-mbert predict-test eval-test errors stats test
help:
	@echo "bootstrap → prep → train → predict-test → eval-test; test runs offline"
install:
	$(PYTHON) -m pip install -r requirements.txt
bootstrap:
	$(PYTHON) -m scripts.bootstrap_from_hf --output data/raw
prep:
	$(PYTHON) -m src.data_prep --input data/raw --output $(DATA_DIR)
train:
	$(PYTHON) -m src.train_flat_ner --model_name $(MODEL) --data_dir $(DATA_DIR) --output_dir $(OUTPUT)
train-xlmr:
	$(MAKE) train MODEL=xlm-roberta-base OUTPUT=outputs/xlmr-flat
train-mbert:
	$(MAKE) train MODEL=bert-base-multilingual-cased OUTPUT=outputs/mbert-flat
predict-test:
	$(PYTHON) -m src.predict_flat_ner --model_dir $(OUTPUT) --data_dir $(DATA_DIR) --split test --out_path $(PRED)
eval-test:
	$(PYTHON) -m src.eval_flat --pred_path $(PRED) --out_dir $(EVAL)
errors:
	$(PYTHON) -m src.error_slices --pred_path $(PRED)
stats:
	$(PYTHON) -m scripts.label_stats $(DATA_DIR)
test:
	$(PYTHON) -m pytest -q
