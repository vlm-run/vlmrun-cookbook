.PHONY: lint

lint:
	pre-commit run --all-files # Uses pyproject.toml
