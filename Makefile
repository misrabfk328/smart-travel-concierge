.PHONY: install lint test eval playground tf-validate clean

install:
	uv sync --extra eval --extra lint

lint:
	uv run ruff check app tests
	uv run ruff format --check app tests
	uv run ty check
	uv run codespell app tests README.md

test:
	uv run pytest tests/unit tests/integration/test_concierge_e2e.py -v

eval:
	agents-cli eval run --dataset tests/eval/datasets/travel-golden-dataset.json

playground:
	agents-cli playground

tf-validate:
	cd deployment/terraform/single-project && terraform fmt -check -recursive .. && terraform init -backend=false && terraform validate

clean:
	rm -rf .pytest_cache .ruff_cache data/*.db
