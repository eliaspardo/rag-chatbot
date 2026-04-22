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
	pytest -s --cov-fail-under=70 tests/unit

test-contract:
	@echo "Ensuring Pact Broker is running for contract tests..."
	docker compose up -d pact-broker
	@echo "Running contract tests..."
	pytest -s tests/contract

test-integration:
	pytest -s tests/integration

test-e2e:
	@echo "Ensuring application is running for e2e tests..."
	docker compose up -d
	@echo "Waiting for services to be ready..."
	@for i in $$(seq 1 30); do \
		if curl -sf http://localhost:8002/health > /dev/null 2>&1; then \
			echo "Inference service is ready!"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "Timeout waiting for inference service to be ready"; \
			exit 1; \
		fi; \
		echo "Waiting for inference service... ($$i/30)"; \
		sleep 2; \
	done
	@echo "Running e2e tests..."
	pytest -s tests/e2e

test-eval:
	@echo "Ensuring MLflow is running for evaluation tests..."
	docker compose up -d mlflow
	@echo "Running evaluation tests..."
	pytest -s -m "deepeval"

pact-publish:
	docker run --rm \
		--network host \
		-v $(PACT_DIR):/pacts \
		pactfoundation/pact-cli \
		publish /pacts \
		--broker-base-url $(BROKER_URL) \
		--consumer-app-version $(CONSUMER_VERSION) \
		--branch $(CONSUMER_BRANCH)