# Makefile for Biological Hypercomputing Platform

.PHONY: help install test lint format clean run-dev run-prod docker-build docker-run docs

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run all tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean temporary files"
	@echo "  run-dev     - Run development experiment"
	@echo "  run-prod    - Run production experiment"
	@echo "  docker-build- Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo "  docs        - Generate documentation"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --cov=src/biocomputing --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ tools/ simulations/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf logs/*.log

run-dev:
	python src/main.py --experiment dual --environment development

run-prod:
	python src/main.py --experiment full --environment production

docker-build:
	docker build -t biocomputing:latest .

docker-run:
	docker run -it --rm biocomputing:latest

docker-compose-up:
	docker-compose up --build

docs:
	cd docs && make html

benchmark:
	python tools/analysis/performance_analyzer.py --benchmark

visualize:
	python tools/visualization/phenomenon_visualizer.py

simulate-molecular:
	python simulations/molecular/molecular_simulator.py

simulate-cellular:
	python simulations/cellular/cellular_simulator.py

simulate-population:
	python simulations/population/population_simulator.py
