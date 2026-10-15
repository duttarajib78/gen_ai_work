# Common tasks. Run `make help`. Requires GNU make and the dev environment (see environment.yml).
PYTHON        ?= python
ARTIFACTS     ?= artifacts
DIST          ?= dist
DATA          ?= data/admissions.parquet
DEPLOY_CONFIG ?= configs/deploy.dev.yaml
TRAIN_CONFIG  ?= configs/train_job.dev.yaml
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)/sagemaker:$(CURDIR)/sagemaker/code

.DEFAULT_GOAL := help
.PHONY: help install test lint lock synthetic-data smoke train verify-bundle package deploy status train-job alerts clean

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

install: ## create the dev environment in the current Python
	$(PYTHON) -m pip install -r requirements/dev.in
	$(PYTHON) -m pip install -e .

test: ## run the unit tests (torch tests are skipped automatically when torch is absent)
	$(PYTHON) -m unittest discover -s tests -t . -v

lint: ## static checks including security rules
	$(PYTHON) -m ruff check src sagemaker scripts tests

lock: ## regenerate hash-pinned lock files (needs network access and pip-tools)
	$(PYTHON) -m piptools compile --generate-hashes --strip-extras requirements/train.in -o requirements/train.lock
	$(PYTHON) -m piptools compile --generate-hashes --strip-extras requirements/inference.in -o requirements/inference.lock

synthetic-data: ## write a PHI-free synthetic dataset
	$(PYTHON) scripts/make_synthetic_data.py --rows 5000 --output data/synthetic/admissions.parquet

smoke: ## end-to-end on synthetic data on CPU: train a tiny model, verify the bundle, score locally
	$(PYTHON) scripts/make_synthetic_data.py --rows 2000 --output data/synthetic/admissions.parquet
	$(PYTHON) -m drg_audit.cli train --stage all --input data/synthetic/admissions.parquet --output-dir $(ARTIFACTS)/smoke \
	  --pretrain-set runtime.require_gpu=false --pretrain-set runtime.amp=false --pretrain-set runtime.num_workers=0 \
	  --pretrain-set epochs=2 --pretrain-set min_code_frequency=3 --pretrain-set encoder.hidden_size=64 \
	  --pretrain-set encoder.num_hidden_layers=2 --pretrain-set encoder.num_attention_heads=2 \
	  --pretrain-set encoder.intermediate_size=128 --pretrain-set tokenization.max_len=48 \
	  --finetune-set runtime.require_gpu=false --finetune-set runtime.amp=false --finetune-set runtime.num_workers=0 \
	  --finetune-set epochs=2 --finetune-set classifier_hidden_dim=32
	$(PYTHON) -m drg_audit.cli verify-bundle --bundle-dir $(ARTIFACTS)/smoke/bundle
	$(PYTHON) -m drg_audit.cli score --input data/synthetic/admissions.parquet --output $(ARTIFACTS)/smoke/scores.csv \
	  --bundle-dir $(ARTIFACTS)/smoke/bundle

train: ## train both stages on a GPU machine: make train DATA=path/to/parquet
	$(PYTHON) -m drg_audit.cli train --stage all --input $(DATA) --output-dir $(ARTIFACTS)/run

verify-bundle: ## verify a bundle's integrity: make verify-bundle ARTIFACTS=artifacts
	$(PYTHON) -m drg_audit.cli verify-bundle --bundle-dir $(ARTIFACTS)/run/bundle

package: ## build dist/model.tar.gz (offline wheelhouse by default)
	$(PYTHON) sagemaker/build_model_package.py --bundle-dir $(ARTIFACTS)/run/bundle --output $(DIST)/model.tar.gz

deploy: ## deploy or blue/green update the endpoint, then smoke-test it
	$(PYTHON) sagemaker/deploy_endpoint.py deploy --config $(DEPLOY_CONFIG) --package $(DIST)/model.tar.gz \
	  --smoke-test examples/request.json

status: ## endpoint status
	$(PYTHON) sagemaker/deploy_endpoint.py status --config $(DEPLOY_CONFIG)

train-job: ## launch the SageMaker Training Job
	$(PYTHON) sagemaker/launch_training_job.py --config $(TRAIN_CONFIG) --wait

alerts: ## generate alert JSON lines from scored admissions
	$(PYTHON) -m drg_audit.cli alerts --config configs/alerts.yaml

clean: ## remove build outputs and caches
	rm -rf $(DIST) .pytest_cache .ruff_cache build src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
