# Uses .RECIPEPREFIX so recipes start with '>' instead of a tab.
.RECIPEPREFIX := >
.DEFAULT_GOAL := help
SHELL := /bin/bash

PROJECT      ?= rule-extractor
ENVIRONMENT  ?= dev
AWS_REGION   ?= $(or $(AWS_DEFAULT_REGION),)
IMAGE_TAG    ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || date +%Y%m%d%H%M%S)
PY           ?= python3

help:  ## List targets
> @grep -E '^[a-zA-Z_.-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

install:  ## Create .venv and install runtime + dev dependencies
> $(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements/dev.in && pip install -e . --no-deps

lock:  ## Compile hash-pinned requirements/prod.lock (needs index access; use your CodeArtifact/PyPI mirror)
> . .venv/bin/activate && pip-compile --generate-hashes --allow-unsafe --strip-extras -o requirements/prod.lock requirements/base.in

test:  ## Run the unit test suite (no AWS access needed)
> PYTHONPATH=src $(PY) -m unittest discover -s tests -t . -v 2>&1 | tail -n 25

lint:  ## ruff + mypy + cfn-lint
> ruff check src tests scripts && mypy src && cfn-lint infra/cloudformation/*.yaml

corpus:  ## Build and verify the page-indexed corpus from source_docs/
> PYTHONPATH=src $(PY) -m rule_extractor corpus build --workers 4 && PYTHONPATH=src $(PY) -m rule_extractor corpus verify

package:  ## Create dist/model.tar.gz (corpus + template + integrity manifest)
> PYTHONPATH=src $(PY) -m rule_extractor package --out dist/model.tar.gz

image:  ## Build and push the container image (BUILDER=docker|sm-docker; needs ECR_REPOSITORY)
> AWS_REGION=$(AWS_REGION) IMAGE_TAG=$(IMAGE_TAG) ./scripts/build_and_push.sh

foundation:  ## Deploy KMS/S3/DynamoDB/SQS/ECR/IAM
> $(PY) scripts/deploy.py stack --stack-name $(PROJECT)-$(ENVIRONMENT)-foundation --template infra/cloudformation/foundation.yaml \
>   --param ProjectName=$(PROJECT) --param Environment=$(ENVIRONMENT)

upload-model:  ## Upload dist/model.tar.gz to the artifact bucket (prints the S3 URI)
> $(PY) scripts/deploy.py upload-model --file dist/model.tar.gz --foundation-stack $(PROJECT)-$(ENVIRONMENT)-foundation

clean:  ## Remove build output
> rm -rf build dist .pytest_cache .mypy_cache .ruff_cache && find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help install lock test lint corpus package image foundation upload-model clean
