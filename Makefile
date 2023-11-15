install_dev:
	@echo "Installing dev dependencies..."
	poetry config virtualenvs.prefer-active-python true
	poetry install --with dev,test,fixers,linters
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

run_all_pipelines:
	@echo "Running all pipelines..."
	@echo "Running feature pipeline..."
	poetry run python src/feature_pipeline/main.py
	@echo "Running training pipeline..."
	poetry run python src/training_pipeline/main.py
	@echo "Running inference pipeline..."
	poetry run python src/inference_pipeline/main.py
