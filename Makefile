install_dev:
	@echo "Installing dev dependencies..."
	poetry config virtualenvs.prefer-active-python true
	poetry install
	poetry run pre-commit install
	poetry run pre-commit autoupdate

install_prod:
	@echo "Installing prod dependencies..."
	poetry config virtualenvs.prefer-active-python true
	poetry install

activate:
	@echo "Activating virtual environment..."
	poetry shell

test:
	@echo "Running tests..."
	pytest -n auto --cov=src --cov-report html

check_all:
	@echo "Checking all files with pre-commit..."
	poetry run pre-commit run --all-files
