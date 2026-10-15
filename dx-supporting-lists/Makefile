# Common tasks. Variables can be set on the command line:  make deploy-foundation ENV=prod REGION=ap-southeast-2 VPC_ID=vpc-... ...
SHELL := /bin/bash
PY ?= python3
ENV ?= prod
REGION ?= ap-southeast-2
FOUNDATION_STACK ?= $(ENV)-dx-support-foundation
ENDPOINT_STACK ?= $(ENV)-dx-support-endpoint
DATA_DIR ?= data
VERSION ?= $(shell date -u +%Y%m%dT%H%M%SZ)

.PHONY: help venv install test lint sample-data dry-run doctor-offline lock validate-templates \
        deploy-foundation package-data image image-codebuild deploy-endpoint describe prewarm-print clean

help:            ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

venv:            ## create .venv
	$(PY) -m venv .venv

install:         ## install the project with development tools (activate .venv first)
	pip install -e ".[dev]" pyyaml

test:            ## run every test (no AWS access needed)
	$(PY) -m unittest discover -s tests -t . -v

lint:            ## static checks
	ruff check src tests deploy

sample-data:     ## write invented, non-clinical sample reference files into ./data-sample
	dx-support make-sample-data --out data-sample

dry-run: sample-data  ## full pipeline on sample data with the offline stub: no credentials, no cost
	DX_DATA_DIR=data-sample dx-support build --dry-run --yes
	DX_DATA_DIR=data-sample dx-support compile --dry-run
	DX_DATA_DIR=data-sample dx-support status --dry-run

doctor-offline:  ## preflight without network
	dx-support doctor --offline

lock:            ## fully pinned, hash-locked dependencies for the image (commit the result)
	pip-compile --generate-hashes --output-file requirements.lock requirements.txt

validate-templates:  ## offline checks of the CloudFormation templates (also run cfn-lint in CI)
	$(PY) deploy/scripts/validate_templates.py

# ---- AWS (needs credentials) -----------------------------------------------------------------------------------------------------------
# Required: VPC_ID VPC_CIDR SUBNETS (comma-separated, one per AZ) BEDROCK_ARNS (from `dx-support bedrock-arns --account <id> --discover`)
deploy-foundation: validate-templates  ## stack 1: key, bucket, registry, role, network rules
	aws cloudformation deploy --region $(REGION) --stack-name $(FOUNDATION_STACK) --template-file deploy/cloudformation/foundation.yaml --capabilities CAPABILITY_IAM \
	  --parameter-overrides EnvironmentName=$(ENV) VpcId=$(VPC_ID) VpcCidr=$(VPC_CIDR) PrivateSubnetIds=$(SUBNETS) BedrockResourceArns='$(BEDROCK_ARNS)' $(FOUNDATION_EXTRA)
	aws cloudformation describe-stacks --region $(REGION) --stack-name $(FOUNDATION_STACK) --query 'Stacks[0].Outputs' --output table

package-data:    ## pack the reference data into the model artifact and upload it
	bash deploy/scripts/package_model_data.sh $(DATA_DIR) $$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(FOUNDATION_STACK) --query "Stacks[0].Outputs[?OutputKey=='ArtifactsBucketName'].OutputValue" --output text) $(VERSION)

image:           ## build and push the image with local Docker
	AWS_REGION=$(REGION) ECR_REPO_URI=$$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(FOUNDATION_STACK) --query "Stacks[0].Outputs[?OutputKey=='EcrRepositoryUri'].OutputValue" --output text) bash deploy/scripts/build_push_image.sh

image-codebuild: ## build and push the image with CodeBuild (no local Docker)
	STACK=$(FOUNDATION_STACK) AWS_DEFAULT_REGION=$(REGION) bash deploy/scripts/build_image_codebuild.sh

# Required: IMAGE_URI MODEL_DATA_URL (printed by the two steps above)
deploy-endpoint: validate-templates  ## stack 2: model, endpoint (blue/green, rollback alarms), autoscaling
	aws cloudformation deploy --region $(REGION) --stack-name $(ENDPOINT_STACK) --template-file deploy/cloudformation/endpoint.yaml \
	  --parameter-overrides FoundationStackName=$(FOUNDATION_STACK) EnvironmentName=$(ENV) ImageUri=$(IMAGE_URI) ModelDataUrl=$(MODEL_DATA_URL) \
	  KmsKeyArn=$$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(FOUNDATION_STACK) --query "Stacks[0].Outputs[?OutputKey=='KmsKeyArn'].OutputValue" --output text) $(ENDPOINT_EXTRA)

describe:        ## ask the deployed endpoint to describe itself
	dx-support invoke --endpoint $(ENV)-dx-support --region $(REGION) --describe

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache output compiled
