# Common tasks. Run from the repository root. Variables can be overridden: make deploy CONFIG=...
PYTHON      ?= python3
CONFIG      ?= tools/deploy/config/prod.yaml
ARTIFACT    ?= artifacts/model.tar.gz

.PHONY: help install-dev test lint artifact image deploy deploy-dry-run smoke-bedrock

help:
	@echo "install-dev      install development dependencies"
	@echo "test             run the unit tests (no AWS access needed)"
	@echo "lint             ruff + mypy (if installed)"
	@echo "artifact         build artifacts/model.tar.gz (REFERENCE_DIR=... DRG_DIR=... S3_URI=... KMS_KEY_ID=...)"
	@echo "image            build and push the container image (needs AWS_REGION, ECR_REPOSITORY)"
	@echo "deploy-dry-run   print the SageMaker/CloudWatch/Auto Scaling calls a deployment would make"
	@echo "deploy           create or update the endpoint from CONFIG"
	@echo "smoke-bedrock    check Bedrock access and request shape with the AUDIT_* environment"

install-dev:
	$(PYTHON) -m pip install -r requirements/dev.txt

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

lint:
	-ruff check src tools tests
	-mypy src

artifact:
	$(PYTHON) -m tools.deploy.package_model_artifact --reference-dir "$(REFERENCE_DIR)" \
	  $(if $(DRG_DIR),--drg-reference-dir "$(DRG_DIR)",--no-drg) --output "$(ARTIFACT)" \
	  $(if $(S3_URI),--s3-uri "$(S3_URI)") $(if $(KMS_KEY_ID),--kms-key-id "$(KMS_KEY_ID)")

image:
	bash tools/deploy/build_and_push.sh

deploy-dry-run:
	$(PYTHON) -m tools.deploy.deploy_endpoint --config "$(CONFIG)" --dry-run

deploy:
	$(PYTHON) -m tools.deploy.deploy_endpoint --config "$(CONFIG)" --smoke-test

smoke-bedrock:
	$(PYTHON) -m tools.diagnostics.smoke_test_bedrock
