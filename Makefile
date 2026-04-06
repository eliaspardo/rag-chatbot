# Pact publishing
BROKER_URL := http://localhost:9292
PACT_DIR := ./pacts

# Defaults to current HEAD, can override: make pact-publish CONSUMER_VERSION=abc1234 CONSUMER_BRANCH=main
CONSUMER_VERSION ?= $(shell git rev-parse --short HEAD)
CONSUMER_BRANCH ?= $(shell git branch --show-current)

.PHONY: pact-publish


up:
	docker compose up -d

up-debug:
	docker compose up 

down:
	docker compose down

test: 
	pytest -s

test-unit:
	pytest -s tests/unit

test-contract:
	pytest -s tests/contract

test-integration:
	pytest -s tests/integration

test-e2e:
	@echo "Ensuring application is running for e2e tests..."
	docker compose up -d
	@echo "Running e2e tests..."
	pytest -s tests/e2e

test-eval:
	@echo "Ensuring MLflow is running for evaluation tests..."
	docker compose up -d mlflow
	@echo "Running evaluation tests..."
	pytest -s -m "deepeval" --log-cli-level=ERROR

pact-publish:
	docker run --rm \
		--network host \
		-v $(PACT_DIR):/pacts \
		pactfoundation/pact-cli \
		publish /pacts \
		--broker-base-url $(BROKER_URL) \
		--consumer-app-version $(CONSUMER_VERSION) \
		--branch $(CONSUMER_BRANCH)