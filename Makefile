install:
	@echo "Installing..."
	poetry config virtualenvs.prefer-active-python true
	poetry install
	poetry run pre-commit install
	poetry run pre-commit autoupdate

activate:
	@echo "Activating virtual environment..."
	poetry shell

test:
	@echo "Running tests..."
	pytest -n auto --cov=src --cov-report html

check_all:
	@echo "Checking all files with pre-commit..."
	pre-commit run --all-files